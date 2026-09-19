#!/usr/bin/env python3
"""GCC-Enhanced Phase C: target-guided equivalence solver.

Given the near-miss compiler output and the KNOWN target bytes, search a small set of
semantics-preserving rewrites until the produced bytes equal the target, each gated by
the Phase B verifier (equiv.py). This is guided search, not blind permutation: the
target pins the answer.

Operators implemented so far (build order per the plan: reg-naming FIRST):
  * reg-realloc: derive the register correspondence sigma that the target's naming
    implies, apply it to the cc1 gas .s, and rebuild through the real maspsx+as toolchain.
  * delay-slot fill: a rule-general POST pass that fills a branch/jr delay slot with the
    immediately-preceding independent instruction (dropping the displaced nop) -- the
    SN aspsx scheduling that modern GNU as does not reproduce.

A match is accepted only when BOTH hold:
  (1) the rebuilt+normalized instruction words equal the target words, and
  (2) equiv.py proves our original cc1 function equivalent to the target under sigma and
      the declared exit live-set.

Usage:
    solve.py <cfile> <func> [--retty void|int]
"""
import os, re, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asmlib
import equiv as E

SCRATCH = os.environ.get("GCCE_SCRATCH", os.path.join(asmlib.ROOT, "gcce-scratch"))
os.makedirs(SCRATCH, exist_ok=True)

_CALLEE_SAVED = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "gp", "sp", "fp", "ra"]
# void: neither v0 nor v1 observable at return. int (32-bit): v0 only -- v1 is a
# caller-saved scratch register, DEAD at exit (only a 64-bit/struct return makes it
# live). Using the precise set matters for addr-form and other transforms that leave
# different-but-dead garbage in a scratch base register.
VOID_LIVE = list(_CALLEE_SAVED)
INT_LIVE = ["v0"] + _CALLEE_SAVED

# ---------------------------------------------------------------------------
# operand parsing on objdump / target disassembly
# ---------------------------------------------------------------------------
def split_ops(opstr):
    out, depth, cur = [], 0, ""
    for ch in opstr:
        if ch == "(":
            depth += 1; cur += ch
        elif ch == ")":
            depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out

def insn_parts(disasm):
    """'lw v1,0(a3)' -> ('lw', [regs in order], [non-reg operand skeletons])."""
    p = disasm.split(None, 1)
    mnem = p[0].lower()
    ops = split_ops(p[1]) if len(p) > 1 else []
    regs, skel = [], []
    for op in ops:
        mm = re.fullmatch(r"(.*)\((\$?\w+)\)", op)
        if mm and E.norm_reg(mm.group(2)) is not None:   # off(base): inner is a real reg
            skel.append(mm.group(1).strip() + "(#)")
            regs.append(E.norm_reg(mm.group(2)))
        else:
            # NOT a register-indexed memory operand: %hi(sym)/%lo(sym) and bare
            # symbols look like off(base) to the regex but carry no register, so they
            # stay pure skeleton (keeps operand arity aligned with the other side).
            r = E.norm_reg(op)
            if r is not None:
                regs.append(r)
                skel.append("#")
            else:
                skel.append(op)                 # immediate / symbol / label
    return mnem, regs, skel

# ---------------------------------------------------------------------------
# sigma derivation (register correspondence our -> target)
# ---------------------------------------------------------------------------
def derive_sigma(our, tgt):
    """From positions whose mnemonic + non-register skeleton already agree, unify the
    register operands to build sigma. Returns (sigma dict, conflicts list)."""
    sigma, conflicts = {}, []
    for (gw, gd), (tw, td) in zip(our, tgt):
        gm, gr, gs = insn_parts(gd)
        tm, tr, ts = insn_parts(td)
        if gm != tm or len(gr) != len(tr):
            continue                             # reordered / different form: skip here
        for a, b in zip(gr, tr):
            if a == "zero" or b == "zero":
                if a != b:
                    conflicts.append(("zero-mismatch", gd, td))
                continue
            if a in sigma and sigma[a] != b:
                conflicts.append((a, sigma[a], b))
            else:
                sigma[a] = b
    return sigma, conflicts

# ---------------------------------------------------------------------------
# apply sigma to the cc1 gas .s (numeric $N registers)
# ---------------------------------------------------------------------------
def apply_sigma_to_s(stext, sigma):
    """Rename $N registers in cc1 gas assembly simultaneously per sigma (ABI->ABI)."""
    num = {}
    for a, b in sigma.items():
        if a in E.ABI2NUM and b in E.ABI2NUM:
            num[E.ABI2NUM[a]] = E.ABI2NUM[b]
    def repl(m):
        n = int(m.group(1))
        return "$%d" % num.get(n, n)
    # only touch register operands ($<digits>), not e.g. section sizes
    return re.sub(r"\$(\d+)\b", repl, stext)

# ---------------------------------------------------------------------------
# addr-form: split-form (%lo folded into each memory operand) -> la-form
# (materialize the full address with an addiu, memory operands use 0(base)).
# Both forms compute the same effective address; the target compiler prefers
# la-form for some single-symbol globals where the near-miss compiler folds %lo. Target-guided:
# only fold the symbols the TARGET materializes with `addiu R,R,%lo(SYM)`.
# ---------------------------------------------------------------------------
def target_laform_syms(tgt):
    """Symbols the target builds via an explicit la (lui;addiu %lo) then 0(base)."""
    syms = set()
    pat = re.compile(r"addiu\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*%lo\(([\w.$]+)\)")
    for _, dis in tgt:
        m = pat.search(dis)
        if m and E.norm_reg(m.group(1)) == E.norm_reg(m.group(2)):
            syms.add(m.group(3))
    return syms

def laform_fold_s(stext, syms):
    """In the cc1 gas .s, rewrite `lui $B,%hi(SYM)` + `%lo(SYM)($B)` split-form into
    la-form: insert `addiu $B,$B,%lo(SYM)` after the lui and change each
    `%lo(SYM)($B)` operand to `0($B)`. Only fires for SYM in `syms` and only when
    every use of $B in its live range is a `%lo(SYM)($B)` memory operand of that same
    SYM (so materializing the address and zeroing the offsets preserves every access).
    Rewrites one lui/base at a time; returns the new text (unchanged if not legal)."""
    lines = stext.split("\n")
    lui_re = re.compile(r"^(\s*)lui\s+(\$\d+)\s*,\s*%hi\(([\w.$]+)\)")
    for sym in syms:
        i = 0
        while i < len(lines):
            m = lui_re.match(lines[i])
            if not m or m.group(3) != sym:
                i += 1; continue
            indent, base = m.group(1), m.group(2)
            reg_re = re.compile(r"(?<![\w$])" + re.escape(base) + r"(?![\w])")
            lo_here = re.compile(r"%lo\(" + re.escape(sym) + r"\)\(" + re.escape(base) + r"\)")
            # scan the live range of $base: from the lui to where $base is redefined
            j = i + 1
            legal = True
            hits = []
            while j < len(lines):
                ln = lines[j]
                body = ln.split("#", 1)[0]
                # base redefined (destination of an instruction) ends the range
                mdef = re.match(r"\s*[a-z][\w.]*\s+" + re.escape(base) + r"\s*,", body)
                if mdef:
                    break
                if reg_re.search(body):
                    if lo_here.search(body):
                        hits.append(j)
                    else:
                        legal = False       # $base used in some other form -> unsafe
                        break
                j += 1
            if legal and hits:
                for j in hits:
                    lines[j] = lo_here.sub("0(%s)" % base, lines[j])
                lines.insert(i + 1, "%saddiu\t%s,%s,%%lo(%s)" % (indent, base, base, sym))
                i = j + 2
            else:
                i += 1
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# un-hi-cse: rematerialization of a %hi(SYM) base per access.
#
# Some GCC 2.8.x builds CSEs the high part of a global address: it hoists ONE
# `lui $B,%hi(SYM)` and keeps $B live, reloading only `lw $D,%lo(SYM)($B)`
# before each use (the pointer value itself is correctly reloaded; only the
# %hi immediate is shared). Retail's private cc1 does NOT share the %hi: it
# REMATERIALIZES `lui R,%hi(SYM); lw R,%lo(SYM)(R)` (base == dest == R) fresh
# before every access.
#
# The transform is target-guided: for each SYM the TARGET remats (an adjacent
# `lui R,%hi(SYM)` + `lw R,%lo(SYM)(R)` with the same R), delete our hoisted
# `lui _,%hi(SYM)` lines and re-emit, at the FRONT of each access region, the
# pair `lui $D,%hi(SYM); lw $D,%lo(SYM)($D)` where $D is that load's own dest
# register (so the old shared base dies and the naming collapses onto the
# target's under sigma). Placing the pair at the region front lets the
# unshared %hi land first; the rhs value-compute then falls into the lw->store
# load-delay slot exactly as the target schedules it (the assembler inserts the
# load-delay nop where no independent instruction is available).
#
# Legality is NOT assumed here: the equiv gate proves our ORIGINAL cc1 function
# equivalent to the target. A pointer reload that an intervening aliasing store
# could invalidate would fail that proof rather than fake a match.
# ---------------------------------------------------------------------------
_STORE_MNEMS = {"sb", "sh", "sw", "swl", "swr", "swc1", "swc2", "sc1"}
_LUI_HI = re.compile(r"^(\s*)lui\s+(\$\d+)\s*,\s*%hi\(([\w.$]+)\)")
_LW_LO = re.compile(r"^(\s*)lw\s+(\$\d+)\s*,\s*%lo\(([\w.$]+)\)\((\$\d+)\)")

def target_remat_syms(tgt):
    """Symbols the target materializes with an inline `lui R,%hi(S); lw R,%lo(S)(R)`
    (base == dest). These are the globals whose %hi the target rematerializes per access."""
    syms = set()
    lui_re = re.compile(r"lui\s+(\$?\w+)\s*,\s*%hi\(([\w.$]+)\)")
    lw_re = re.compile(r"lw\s+(\$?\w+)\s*,\s*%lo\(([\w.$]+)\)\((\$?\w+)\)")
    for i in range(len(tgt) - 1):
        a = lui_re.search(tgt[i][1]); b = lw_re.search(tgt[i + 1][1])
        if not a or not b:
            continue
        rd = E.norm_reg(a.group(1)); ld = E.norm_reg(b.group(1)); lb = E.norm_reg(b.group(3))
        if a.group(2) == b.group(2) and rd == ld == lb:
            syms.add(a.group(2))
    return syms

def _is_insn_line(line):
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("."):
        return False
    if s.endswith(":"):
        return False
    return bool(re.match(r"[a-z][\w.]*", s))

def _line_mnem(line):
    return line.strip().split(None, 1)[0].lower()

def un_hi_cse_s(stext, remat_syms):
    """Rewrite the cc1 gas .s to un-CSE the %hi base of each `remat_syms` symbol.
    Returns new text (unchanged if the transform does not fire)."""
    if not remat_syms:
        return stext
    lines = stext.split("\n")
    # Only un-CSE a symbol whose %hi the near-miss compiler actually HOISTED away from its load: a
    # `lw $D,%lo(sym)($B)` whose base $B differs from its dest $D (the hoisted shared base).
    # When the near-miss compiler already emits `lw $D,%lo(sym)($D)` (base == dest, the lui adjacent), the
    # load is already in the rematerialized form -- undoing it would only duplicate the pair and
    # corrupt the schedule (this bit a real branch function whose single-use globals the target also
    # loads base == dest).
    hoisted = set()
    for l in lines:
        m = _LW_LO.match(l)
        if m and m.group(3) in remat_syms and m.group(2) != m.group(4):   # dest != base
            hoisted.add(m.group(3))
    remat_syms = hoisted
    if not remat_syms:
        return stext
    # region id per instruction line (region increments AFTER a store)
    region_of = {}
    reg = 0
    first_insn_line = {}   # region -> first original line index carrying an instruction
    saw_remat = False
    for idx, ln in enumerate(lines):
        if not _is_insn_line(ln):
            continue
        region_of[idx] = reg
        first_insn_line.setdefault(reg, idx)
        m = _LW_LO.match(ln)
        if m and m.group(3) in remat_syms:
            saw_remat = True
        if _line_mnem(ln) in _STORE_MNEMS:
            reg += 1
    if not saw_remat:
        return stext
    # per-region insertion list, built from the remat lw's in original order
    inserts = {}
    delete = set()
    for idx, ln in enumerate(lines):
        if idx not in region_of:
            continue
        mlui = _LUI_HI.match(ln)
        if mlui and mlui.group(3) in remat_syms:
            delete.add(idx)                       # drop the shared/old %hi materialization
            continue
        mlw = _LW_LO.match(ln)
        if mlw and mlw.group(3) in remat_syms:
            indent, dst, sym = mlw.group(1) or "\t", mlw.group(2), mlw.group(3)
            r = region_of[idx]
            inserts.setdefault(r, []).append(
                ["%slui\t%s,%%hi(%s)" % (indent, dst, sym),
                 "%slw\t%s,%%lo(%s)(%s)" % (indent, dst, sym, dst)])
            delete.add(idx)                       # remove the original lo-load in place
    # emit
    out = []
    emitted_region = set()
    for idx, ln in enumerate(lines):
        r = region_of.get(idx)
        if r is not None and r not in emitted_region and idx == first_insn_line[r]:
            for pair in inserts.get(r, []):
                out.extend(pair)
            emitted_region.add(r)
        if idx in delete:
            continue
        out.append(ln)
    return "\n".join(out)

# ---------------------------------------------------------------------------
# exit-merge: fold multiple `j $31` returns into one shared `jr $ra` exit.
#
# cc1 emits each return of a leaf/branch function as its own `j $31` (assembled to
# `jr $ra`) with the return value in the delay slot; the target funnels every return
# through ONE shared `jr $ra` laid last, the other paths reaching it by `j <EXIT>`.
# Rewrite the cc1 gas .s: the LAST return in layout drops its `j $31` (its delay value
# becomes a plain body insn that falls into the shared exit); every earlier return's
# `j $31` becomes `j <EXIT>` keeping its delay; append `<EXIT>: j $31 / nop` last.
#
# Semantics-preserving (a jr delay always runs before return either way) -- proven by
# equiv.canonicalize_exits and, ultimately, by hitting the known target bytes. Fires
# only when target-guided detection says the target is single-exit and ours is not.
# ---------------------------------------------------------------------------
_EXIT_LABEL = "$Lgcce_exit"
_RET_S = re.compile(r"^\s*(?:j\s+\$31|jr\s+\$(?:31|ra))\b")
def _s_is_label(l):
    s = l.strip()
    return s.endswith(":") and not s.startswith(".")
def _s_is_insn(l):
    s = l.strip()
    return bool(s) and not s.startswith("#") and not s.startswith(".") \
        and not s.endswith(":") and bool(re.match(r"[a-z]", s))
def _s_mnem(l):
    return l.strip().split(None, 1)[0].lower()
def _s_is_return(l):
    return bool(_RET_S.match(l))

def merge_exits_s(stext):
    """Fold cc1's multi `j $31` returns into one shared exit. Returns (text, n_returns)."""
    lines = stext.split("\n")
    rets = [i for i, l in enumerate(lines) if _s_is_return(l)]
    if len(rets) < 2:
        return stext, len(rets)
    last_i = rets[-1]
    out = []
    for i, l in enumerate(lines):
        if i == last_i:
            continue                       # drop the last return's `j $31` (keep its delay)
        if _s_is_return(l):
            out.append(re.sub(r"(?:j|jr)\s+\$(?:31|ra)\b", "j\t%s" % _EXIT_LABEL, l, count=1))
        else:
            out.append(l)
    ins = len(out)
    for k in range(len(out) - 1, -1, -1):
        if out[k].strip().startswith(".end"):
            ins = k; break
    exitblk = ["%s:" % _EXIT_LABEL, "\t.set\tnoreorder", "\t.set\tnomacro",
               "\tj\t$31", "\tnop", "\t.set\tmacro", "\t.set\treorder"]
    out = out[:ins] + exitblk + out[ins:]
    return "\n".join(out), len(rets)

# cond-branch delay unfill (target-guided): where the target's k-th conditional branch
# has a nop delay but the near-miss compiler filled it, move our fill instruction out to just after the
# branch's noreorder block and restore the nop, so the delay-slot placement matches.
_COND_S = re.compile(r"^(beq|bne|blez|bgtz|bltz|bgez|beqz|bnez)\b")
def _target_cond_delay_nop(tgt):
    res = []
    for i, (_, dis) in enumerate(tgt):
        if _COND_S.match(dis.split(None, 1)[0].lower()):
            nxt = tgt[i + 1][1] if i + 1 < len(tgt) else ""
            res.append(nxt.strip() in ("nop", "sll zero,zero,0", "sll zero,zero,0x0"))
    return res

def unfill_cond_delay_s(stext, tgt):
    want_nop = _target_cond_delay_nop(tgt)
    lines = stext.split("\n")
    result, idx, k = [], 0, -1
    while idx < len(lines):
        l = lines[idx]
        if _s_is_insn(l) and _COND_S.match(_s_mnem(l)):
            k += 1
            d = idx + 1
            while d < len(lines) and not _s_is_insn(lines[d]):
                if _s_is_label(lines[d]):
                    d = None; break
                d += 1
            filled = d is not None and lines[d].strip() not in (
                "nop", "sll zero,zero,0", "sll zero,zero,0x0")
            if filled and k < len(want_nop) and want_nop[k]:
                result.append(l)
                indent = re.match(r"^(\s*)", lines[d]).group(1)
                displaced = lines[d]
                for m in range(idx + 1, d):
                    result.append(lines[m])
                result.append("%snop" % indent)
                m = d + 1
                tail = []
                while m < len(lines) and lines[m].strip() in (
                        ".set\tmacro", ".set macro", ".set\treorder", ".set reorder"):
                    tail.append(lines[m]); m += 1
                result.extend(tail)
                result.append(displaced)
                idx = m
                continue
        result.append(l)
        idx += 1
    return "\n".join(result)

# ---------------------------------------------------------------------------
# delay-slot fill POST pass (rule-general, target-guided)
# ---------------------------------------------------------------------------
_BRANCHY = re.compile(r"^(b|j|jr|jal|beq|bne|blez|bgtz|bltz|bgez|beqz|bnez)")
def is_branch(disasm):
    return bool(_BRANCHY.match(disasm.split(None, 1)[0].lower()))
def is_nop(disasm):
    return disasm.strip() in ("nop", "sll zero,zero,0", "sll zero,zero,0x0")

def _target_branch_filled(tgt):
    """For each control-flow instruction in the target (in order), whether its delay slot
    holds a real instruction (True) or a nop (False). Used to keep delay_fill from filling
    a slot the target leaves empty (e.g. one we deliberately unfilled)."""
    res = []
    for i, (_, dis) in enumerate(tgt):
        if is_branch(dis):
            nxt = tgt[i + 1][1] if i + 1 < len(tgt) else ""
            res.append(bool(nxt.strip()) and not is_nop(nxt))   # no trailing word == nop slot
    return res

def delay_fill(words, tgt=None):
    """Fill a branch/jr delay slot (currently nop) with the immediately-preceding
    independent instruction, dropping the nop. Only moves an instruction that neither
    writes a register the branch reads nor is itself control flow -- the exact SN aspsx
    schedule. When `tgt` is given, fill the k-th branch's slot only if the target's k-th
    branch is itself filled (so an intentionally-empty slot is left as-is). Returns a new
    word list (possibly shorter)."""
    want = _target_branch_filled(tgt) if tgt is not None else None
    out = list(words)
    i = 0
    k = -1
    while i < len(out) - 1:
        dis = out[i][1]
        if is_branch(dis):
            k += 1
            if is_nop(out[i + 1][1]) and i >= 1 and (want is None or (k < len(want) and want[k])):
                prev = out[i - 1]
                if not is_branch(prev[1]) and not is_nop(prev[1]):
                    pdef = E.defs_uses(_mk_insn(prev[1]))[0]
                    bruse = E.defs_uses(_mk_insn(dis))[1]
                    if not (pdef & bruse):
                        out[i - 1], out[i] = out[i], prev      # swap br above prev
                        del out[i + 1]                          # drop the nop
                        i += 1
                        continue
        i += 1
    return out

# ---------------------------------------------------------------------------
# operand-recolor: local dead-temp recolor of a commutative accumulate.
#
# The near-miss compiler sometimes accumulates a base+index address into the INDEX
# register while the target compiler accumulates into the BASE register:
#     near-miss: addu $I,$I,$B ; <mem> off($I)   (result in the index temp $I)
#     target:    addu $B,$B,$I ; <mem> off($B)   (result in the base temp $B)
# The op is commutative so the VALUE and the memory effect are identical; $I and
# $B are both caller-saved temps dead at the function exit, so the only visible
# difference is which dead register carries the address. A global register map
# (sigma) cannot express this -- $I and $B keep their identity everywhere else and
# only swap roles for this one accumulate -- so it is a LOCAL recolor: rewrite the
# accumulate to the target's form and rename the old destination ($I) to $B in the
# following instructions until $I is redefined. Target-guided (fire only on the
# k-th commutative accumulate whose target form is `op $B,$B,$I` while the near-miss
# form is the swapped `op $I,$I,$B`), and the equiv gate proves EQUAL (identical
# memory, the divergent regs dead). Bail if $B is written before the old $I dies.
# ---------------------------------------------------------------------------
_COMMUTATIVE_ACC = {"addu", "add", "and", "or", "xor", "nor"}
_ACC_S = re.compile(r"^(\s*)(\w+)\s+(\$\d+)\s*,\s*(\$\d+)\s*,\s*(\$\d+)\s*$")

def _accs_numeric(pairs, to_num):
    """List of (op, dst, rs, rt) numeric-register commutative ACCUMULATES (dst==rs, rt a
    real non-zero register) in order. Restricting to the accumulate form keeps the two
    sides aligned: a `op rd,rs,zero` renders as `addu` in a splat target but as a `move`
    pseudo-op in the near-miss assembly, so counting all 3-operand commutatives would
    mismatch."""
    zero = "$%d" % E.ABI2NUM["zero"] if "zero" in E.ABI2NUM else "$0"
    out = []
    for _, dis in pairs:
        m = re.match(r"(\w+)\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*(\$?\w+)\s*$", dis)
        if not m or m.group(1).lower() not in _COMMUTATIVE_ACC:
            continue
        d, s, t = to_num(m.group(2)), to_num(m.group(3)), to_num(m.group(4))
        if d and s and t and d == s and t != zero:
            out.append((m.group(1).lower(), d, s, t))
    return out

def _abi_to_num(tok):
    r = E.norm_reg(tok)
    return ("$%d" % E.ABI2NUM[r]) if r in E.ABI2NUM else None

def _num_to_num(tok):
    return tok if re.fullmatch(r"\$\d+", tok or "") else None

def _s_insn_lines(lines):
    return [i for i, l in enumerate(lines) if _s_is_insn(l)]

def operand_recolor_s(stext, tgt):
    """Recolor a commutative accumulate the near-miss computes into the index temp but
    the target computes into the base temp (a local dead-temp swap). Returns new text."""
    tgt_accs = _accs_numeric(tgt, _abi_to_num)
    if not tgt_accs:
        return stext
    lines = stext.split("\n")
    insn_ix = _s_insn_lines(lines)
    zero = "$%d" % E.ABI2NUM["zero"] if "zero" in E.ABI2NUM else "$0"
    our_accs = []           # (line_index, op, dst, rs, rt) -- accumulate form only
    for i in insn_ix:
        m = _ACC_S.match(lines[i])
        if m and m.group(2).lower() in _COMMUTATIVE_ACC \
                and m.group(3) == m.group(4) and m.group(5) != zero:
            our_accs.append((i, m.group(2).lower(), m.group(3), m.group(4), m.group(5)))
    if len(our_accs) != len(tgt_accs):
        return stext        # body not aligned: do not risk a recolor
    for (li, op, od, os_, ot), (top, td, ts, tt) in zip(our_accs, tgt_accs):
        if op != top:
            continue
        # target accumulates S into B: `op $B,$B,$S` (td==ts, S==tt). The near-miss is the
        # swapped `op $S,$S,$B` (od==os_==tt, and ot==td). Recolor od(=$S) -> $B(td).
        if not (td == ts and od == os_ and od == tt and ot == td):
            continue
        newB, oldI = td, od
        indent = lines[li][:len(lines[li]) - len(lines[li].lstrip())]
        # verify the merge is safe: scan forward from after the accumulate; newB must
        # not be written before oldI is redefined (else merging clobbers a live value).
        safe = True
        redef_at = None
        for j in insn_ix:
            if j <= li:
                continue
            mm = re.match(r"\s*([a-z][\w.]*)\s+(\$\d+)\b", lines[j])
            if not mm:
                continue
            mn, dst = mm.group(1).lower(), mm.group(2)
            is_store = mn in _STORE_MNEMS or mn.startswith(("sb", "sh", "sw"))
            if not is_store and dst == newB:      # newB overwritten while merging: unsafe
                safe = False; break
            if not is_store and dst == oldI:       # oldI redefined: stop renaming here
                redef_at = j; break
        if not safe:
            continue
        lines[li] = "%s%s\t%s,%s,%s" % (indent, op, newB, newB, oldI)
        reg_re = re.compile(r"(?<![\w$])" + re.escape(oldI) + r"(?![\w])")
        for j in insn_ix:
            if j <= li:
                continue
            if redef_at is not None and j >= redef_at:
                break
            lines[j] = reg_re.sub(newB, lines[j])
        return "\n".join(lines)
    return stext

# ---------------------------------------------------------------------------
# commutative-operand-swap (operand-canon, target-guided, post-sigma).
#
# A commutative op writes the SAME destination on both sides but our cc1 and
# the target's cc1 list the two source operands in the opposite order:
#     ours:   op $D,$X,$Y
#     target: op $D,$Y,$X        (same $D, sources swapped)
# The op is commutative so the value and every effect are identical; only the
# two source register FIELDS of this one instruction are transposed, which the
# global register map (sigma) cannot express -- sigma has already unified the
# registers (it canonicalizes commutative operands when building the map, so the
# swap is invisible to it) yet the emitted word still differs by the rs/rt field
# order. This is a pure in-place field swap: no destination changes, no other
# instruction is touched, no liveness reasoning is needed. Runs POST-sigma so
# both sides are in the same register-number space; target-guided (rewrite only
# to the order the target uses) and the equiv gate proves EQUAL.
#
# Distinct from operand_recolor_s, which handles a DIFFERENT destination (the
# accumulate lands in the wrong dead temp and must be renamed downstream). Here
# the destination already agrees; the swap alone closes the byte gap.
# ---------------------------------------------------------------------------
def _commutatives_numeric(pairs, to_num):
    """List of (op, dst, rs, rt) commutative 3-register insns with both sources real
    non-zero registers, in order. Excludes the `op rd,rs,zero` move-pseudo form so the
    two sides stay positionally aligned (a move renders differently across cc1s)."""
    zero = "$%d" % E.ABI2NUM["zero"] if "zero" in E.ABI2NUM else "$0"
    out = []
    for _, dis in pairs:
        m = re.match(r"(\w+)\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*(\$?\w+)\s*$", dis)
        if not m or m.group(1).lower() not in _COMMUTATIVE_ACC:
            continue
        d, s, t = to_num(m.group(2)), to_num(m.group(3)), to_num(m.group(4))
        if d and s and t and s != zero and t != zero:
            out.append((m.group(1).lower(), d, s, t))
    return out

def commutative_swap_s(stext, tgt):
    """Transpose the two source operands of a commutative insn whose destination already
    matches the target but whose rs/rt order does not. stext is sigma-applied (numeric
    register space, same as target after _abi_to_num). Returns new text (unchanged if the
    transform does not fire)."""
    tgt_c = _commutatives_numeric(tgt, _abi_to_num)
    if not tgt_c:
        return stext
    lines = stext.split("\n")
    insn_ix = _s_insn_lines(lines)
    zero = "$%d" % E.ABI2NUM["zero"] if "zero" in E.ABI2NUM else "$0"
    our = []                                 # (line_index, op, dst, rs, rt)
    for i in insn_ix:
        m = _ACC_S.match(lines[i])
        if m and m.group(2).lower() in _COMMUTATIVE_ACC \
                and m.group(4) != zero and m.group(5) != zero:
            our.append((i, m.group(2).lower(), m.group(3), m.group(4), m.group(5)))
    if len(our) != len(tgt_c):
        return stext                         # body not aligned: do not risk a swap
    changed = False
    for (li, op, od, os_, ot), (top, td, ts, tt) in zip(our, tgt_c):
        # destination must already agree; sources must be the same pair in swapped order.
        if op != top or od != td:
            continue
        if os_ == ts and ot == tt:
            continue                         # already in target order
        if os_ == tt and ot == ts:
            indent = lines[li][:len(lines[li]) - len(lines[li].lstrip())]
            lines[li] = "%s%s\t%s,%s,%s" % (indent, op, od, ts, tt)
            changed = True
    return "\n".join(lines) if changed else stext

# ---------------------------------------------------------------------------
# epilogue-unfill (FILL/UNFILL class, target-guided).
#
# Some PSX GCC 2.x builds deallocate the stack frame BEFORE the return:
#     lw   $ra,off($sp)
#     addiu $sp,$sp,+N      # dealloc precedes jr
#     jr   $ra
#     nop                   # jr delay is empty
# Another build fills the jr delay with the sp-restore instead:
#     lw   $ra,off($sp)
#     nop
#     jr   $ra
#     addiu $sp,$sp,+N      # dealloc in the jr delay slot
# Both restore $sp before the caller resumes (a jr delay insn always executes
# before control leaves), so the sp-dealloc commutes with the return + the nop:
# transposing the pre-jr nop and the in-delay `addiu $sp` is semantics-preserving.
# Target-guided: fire only when the target ends `addiu $sp,$sp,+N; jr $ra; (nop)`
# and the near-miss output ends `nop; jr $ra; addiu $sp,$sp,+N`. The equiv gate
# proves EQUAL. No verifier canon pass is needed: these are single-exit
# straight-line functions (a jal is a call event, not a block split), so the
# per-block havoc proof compares the same final live-out either way.
# ---------------------------------------------------------------------------
def _sp_delta(dis):
    """If `dis` is `addiu $sp,$sp,IMM`, return IMM (signed); else None."""
    m = re.match(r"addiu\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*(-?(?:0x)?[0-9a-fA-F]+)\s*$", dis)
    if not m or E.norm_reg(m.group(1)) != "sp" or E.norm_reg(m.group(2)) != "sp":
        return None
    try:
        return int(m.group(3), 0)
    except ValueError:
        return None

def _is_ret(dis):
    p = dis.split(None, 1)
    return p and p[0].lower() == "jr" and len(p) > 1 and E.norm_reg(p[1].strip()) == "ra"

def _target_wants_sp_before_jr(tgt):
    """True when the (pad-stripped) target ends `addiu $sp,$sp,+N ; jr $ra` -- dealloc
    before the return with an empty (stripped-nop) jr delay."""
    if len(tgt) < 2:
        return False
    if not _is_ret(tgt[-1][1]):
        return False
    d = _sp_delta(tgt[-2][1])
    return d is not None and d > 0

def epilogue_unfill(words, tgt):
    """Move a positive `addiu $sp` out of the jr delay slot to just before the jr,
    restoring a nop in the delay -- target-guided. Returns a new word list."""
    if not _target_wants_sp_before_jr(tgt):
        return words
    out = list(words)
    for j in range(len(out) - 1):
        if not _is_ret(out[j][1]):
            continue
        d = _sp_delta(out[j + 1][1])                     # addiu $sp in the jr delay slot
        if d is None or d <= 0:
            continue
        if j >= 1 and is_nop(out[j - 1][1]):             # transpose the pre-jr nop <-> delay
            out[j - 1], out[j + 1] = out[j + 1], out[j - 1]
        else:                                            # no pre-jr filler: hoist + fresh nop
            insn = out.pop(j + 1)
            out.insert(j, insn)
            out.insert(j + 2, ("00000000", "nop"))
        break
    return out

def _mk_insn(disasm):
    mnem, _, _ = insn_parts(disasm)
    p = disasm.split(None, 1)
    ops = split_ops(p[1]) if len(p) > 1 else []
    return E.Insn(mnem, ops, disasm)

# ---------------------------------------------------------------------------
# building an equiv.Func straight from a word list's disassembly
# ---------------------------------------------------------------------------
def func_from_words(name, words):
    insns = []
    for _, dis in words:
        insns.append(_mk_insn(dis))
    return E.Func(name, insns, [])

# objdump prints a branch/jump destination as a section address plus an auto-label
# (`beq $a1,$v0,30 <LM8>`, `j 34 <LM9>`), never the source's `.L` labels. To hand
# equiv.py a CFG with correct edges, resolve each such address to the instruction
# index it lands on and give both the branch operand and the target instruction a
# synthetic `.Lt<idx>` label. jr/jalr carry a register, not an address: no target.
_OBJ_TGT = re.compile(r",?\s*([0-9a-fA-F]+)\s+<[^>]*>\s*$")
def _branch_target_addr(dis):
    p = dis.split(None, 1)
    if not p:
        return None
    mn = p[0].lower()
    if mn in ("jr", "jalr") or not is_branch(dis):
        return None
    if mn.startswith("jal"):        # a real call (jal sym): not an intra-function edge
        if "<" in dis and re.search(r"<[A-Za-z_.$]", dis) and "+" not in dis:
            # jal to a named function, not a local offset -> leave as call, no edge
            return None
    m = _OBJ_TGT.search(dis)
    return int(m.group(1), 16) if m else None

def func_from_reloc(name, triples):
    """Build an equiv.Func from [(addr, be, disasm)] with intra-function branch edges
    resolved to synthetic `.Lt<idx>` labels (see _branch_target_addr)."""
    addr2idx = {a: i for i, (a, _, _) in enumerate(triples)}
    tgt_idx = set()
    for _, _, dis in triples:
        off = _branch_target_addr(dis)
        if off is not None and off in addr2idx:
            tgt_idx.add(addr2idx[off])
    lab_of = {i: ".Lt%d" % i for i in tgt_idx}
    insns = []
    for i, (_, _, dis) in enumerate(triples):
        off = _branch_target_addr(dis)
        ins = _mk_insn(dis)
        if off is not None and off in addr2idx:
            ins.ops = ins.ops[:-1] + [lab_of[addr2idx[off]]]   # objdump addr -> label
        ins.label = lab_of.get(i)
        insns.append(ins)
    return E.Func(name, insns, [lab_of[i] for i in sorted(lab_of)])

# ---------------------------------------------------------------------------
# compare helper
# ---------------------------------------------------------------------------
def cmp_words(tgt, got):
    n = max(len(tgt), len(got))
    ok = len(tgt) == len(got)
    rows = []
    for i in range(n):
        t = tgt[i] if i < len(tgt) else ("--------", "")
        g = got[i] if i < len(got) else ("--------", "")
        good = asmlib.words_eq(t, g)
        if not good:
            ok = False
        rows.append((i, t, g, good))
    return ok, rows

def show(rows, limit=100):
    for i, t, g, good in rows[:limit]:
        mark = "" if good else "   <<<"
        print("%2d  T %s %-24s | G %s %-24s%s" % (i, t[0], t[1], g[0], g[1], mark))

# ---------------------------------------------------------------------------
def solve_one(cfile, func, retty="void"):
    """Run the full solve pipeline for one (cfile, func). Returns a result dict:
      status   : 'verified' | 'bytes-only' | 'not-solved' | 'already' | 'error'
      sigma    : register correspondence our->target
      conflicts: sigma conflicts (empty for a pure correspondence)
      bytes_ok : produced words equal the target
      equiv_ok : our-original == target under sigma + live-set (the honesty gate)
      rows     : per-word compare rows (for display)
      error    : message when status == 'error'
    """
    r = {"func": func, "cfile": cfile, "retty": retty, "sigma": {}, "conflicts": [],
         "bytes_ok": False, "equiv_ok": False, "rows": [], "error": None,
         "status": "not-solved"}
    live = VOID_LIVE if retty == "void" else INT_LIVE

    s0 = os.path.join(SCRATCH, func + ".s")
    ok, err = asmlib.compile_to_s(cfile, s0)
    if not ok:
        r["status"] = "error"; r["error"] = err; return r
    our, err = asmlib.assemble_words(s0, func)
    if err:
        r["status"] = "error"; r["error"] = err; return r
    tgt = _strip_trailing_pad(asmlib.target_words(func))
    our = _strip_trailing_pad(our)

    base_ok, _ = cmp_words(tgt, our)
    if base_ok:
        r["status"] = "already"; r["bytes_ok"] = True; return r

    src = s0

    # 0a) un-hi-cse: rematerialize the %hi base of each global the target remats,
    # so our count of address-materialization insns aligns with the target before
    # sigma. Runs first because it changes instruction counts (adds a lui per use,
    # drops the shared hoist).
    remat = target_remat_syms(tgt)
    if remat:
        stext_r = open(src, encoding="utf-8", errors="replace").read()
        uncsed = un_hi_cse_s(stext_r, remat)
        if uncsed != stext_r:
            s_rm = os.path.join(SCRATCH, func + ".remat.s")
            open(s_rm, "w", encoding="utf-8").write(uncsed)
            our_rm, err_rm = asmlib.assemble_words(s_rm, func)
            if not err_rm:
                our = _strip_trailing_pad(our_rm); src = s_rm
                r["hi_cse"] = sorted(remat)

    # 0b) addr-form: fold our split-form globals to la-form for the symbols the target
    # materializes with an explicit la. Runs before sigma so the register
    # correspondence is derived from the (now count-aligned) la-form assembly.
    la_syms = target_laform_syms(tgt)
    if la_syms:
        stext0 = open(src, encoding="utf-8", errors="replace").read()
        folded = laform_fold_s(stext0, la_syms)
        if folded != stext0:
            s_la = os.path.join(SCRATCH, func + ".la.s")
            open(s_la, "w", encoding="utf-8").write(folded)
            our2, err2 = asmlib.assemble_words(s_la, func)
            if not err2:
                our = _strip_trailing_pad(our2); src = s_la
                r["addr_form"] = sorted(la_syms)

    # 0c) exit-merge + cond-delay-unfill: when the target funnels every return through
    # ONE shared `jr $ra` while the near-miss compiler emits a `jr` per return path, fold our returns into
    # that funnel form (merge_exits_s) and unfill any conditional-branch delay the target
    # leaves as nop (unfill_cond_delay_s). Target-guided: only when target is single-exit
    # and ours is not. Runs on the .s before sigma so the reassembled words align to target.
    n_ret_tgt = _count_returns(tgt)
    n_ret_our = _count_returns(our)
    canon = n_ret_tgt == 1 and n_ret_our > 1
    r["exit_merge"] = canon
    if canon:
        stext_e = open(src, encoding="utf-8", errors="replace").read()
        stext_e = unfill_cond_delay_s(stext_e, tgt)
        merged, nret = merge_exits_s(stext_e)
        if merged != open(src, encoding="utf-8", errors="replace").read():
            s_ex = os.path.join(SCRATCH, func + ".exit.s")
            open(s_ex, "w", encoding="utf-8").write(merged)
            our_ex, err_ex = asmlib.assemble_words(s_ex, func)
            if not err_ex:
                our = _strip_trailing_pad(our_ex); src = s_ex

    # 0d) operand-recolor: local dead-temp recolor of a commutative accumulate the near-miss
    # computes into the index temp while the target computes into the base temp. Runs before
    # sigma because it repairs the exact register-role swap that would otherwise be a sigma
    # conflict.
    stext_c = open(src, encoding="utf-8", errors="replace").read()
    recolored = operand_recolor_s(stext_c, tgt)
    if recolored != stext_c:
        s_rc = os.path.join(SCRATCH, func + ".recolor.s")
        open(s_rc, "w", encoding="utf-8").write(recolored)
        our_rc, err_rc = asmlib.assemble_words(s_rc, func)
        if not err_rc:
            our = _strip_trailing_pad(our_rc); src = s_rc
            r["operand_recolor"] = True

    # 1) reg-realloc
    sigma, conflicts = derive_sigma(our, tgt)
    r["sigma"] = sigma; r["conflicts"] = conflicts

    stext = open(src, encoding="utf-8", errors="replace").read()
    sigma_text = apply_sigma_to_s(stext, sigma)
    # 1b) commutative-operand-swap: transpose the rs/rt fields of a commutative insn whose
    # destination already matches the target but whose source order does not (a pure
    # in-place operand-canon swap, post-sigma so both sides share the register space).
    swapped_text = commutative_swap_s(sigma_text, tgt)
    r["commutative_swap"] = swapped_text != sigma_text
    s1 = os.path.join(SCRATCH, func + ".sigma.s")
    open(s1, "w", encoding="utf-8").write(swapped_text)
    renamed, err = asmlib.assemble_words(s1, func)
    if err:
        r["status"] = "error"; r["error"] = err; return r

    # 2) delay-slot fill POST pass (needs the nop still present); target-guided so it never
    # re-fills a slot we intentionally unfilled for the exit-merge.
    filled = delay_fill(renamed, tgt)
    # 3) epilogue-unfill: move the sp-dealloc out of the jr delay to before the jr when the
    # target deallocates the frame before returning (target-guided, semantics-preserving).
    epi = epilogue_unfill(filled, tgt)
    r["epi_unfill"] = epi is not filled and epi != filled
    final = _strip_trailing_pad(epi)
    match_ok, rows = cmp_words(tgt, final)
    r["bytes_ok"] = match_ok; r["rows"] = rows

    # equivalence gate. Build our side from reloc'd objdump WITH resolved branch edges
    # (func_from_reloc) so a branchy function's CFG is well-formed. canon_exits folds the
    # single-exit tail; canon_delays normalizes branch-delay placement so a filled-vs-nop
    # delay divergence in the body does not spuriously break block correspondence. Both are
    # semantics-preserving and applied identically to each side (see equiv.py); enabled
    # target-guided (only the exit-merge case) so same-shape functions keep the exact CFG.
    trip, err = asmlib.assemble_reloc_addr(s0, func)
    if err:
        r["status"] = "error"; r["error"] = err; return r
    while trip and trip[-1][2].strip() == "nop":     # triple = (addr, be, disasm)
        trip = trip[:-1]
    fa = func_from_reloc(func, trip)
    fb = E.parse_s(os.path.join(asmlib.TARGET_DIR, func + ".s"), func)
    res = E.check_equiv(fa, fb, live=live, regmap=sigma,
                        canon_exits=canon, canon_delays=canon)
    r["equiv_ok"] = res.equal
    r["equiv_reason"] = res.reason

    if match_ok and res.equal:
        r["status"] = "verified"
    elif match_ok:
        r["status"] = "bytes-only"          # bytes match but gate did not prove it: suspect
    else:
        r["status"] = "not-solved"
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cfile")
    ap.add_argument("func")
    ap.add_argument("--retty", choices=["void", "int"], default="void")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    r = solve_one(args.cfile, args.func, args.retty)
    if r["status"] == "error":
        print(r["error"]); sys.exit(2)
    if r["status"] == "already":
        print("ALREADY MATCHES"); sys.exit(0)

    if r["conflicts"]:
        print("sigma conflicts (not a pure register correspondence):")
        for c in r["conflicts"][:10]:
            print("  ", c)
    print("sigma:", {("$%s" % k): ("$%s" % v) for k, v in sorted(r["sigma"].items())})
    print("\n== after reg-realloc + delay-fill ==")
    show(r["rows"])
    print("\nBYTES:", "MATCH" if r["bytes_ok"] else "NO MATCH")
    print("EQUIV(our_original vs target, sigma, live=%s): %s"
          % (args.retty, ("EQUAL" if r["equiv_ok"] else "NOT-EQUAL")))
    if not r["equiv_ok"] and r.get("equiv_reason"):
        print("  reason:", r["equiv_reason"])
    print("\nRESULT:", "VERIFIED MATCH" if r["status"] == "verified"
          else ("BYTES-ONLY (unproven)" if r["status"] == "bytes-only" else "not solved"))
    sys.exit(0 if r["status"] == "verified" else 1)

def _strip_trailing_pad(words):
    while words and words[-1][1].strip() == "nop":
        words = words[:-1]
    return words

def _count_returns(words):
    """Number of `jr $ra` (return) instructions in a word list."""
    n = 0
    for _, dis in words:
        p = dis.split(None, 1)
        if p[0].lower() == "jr" and len(p) > 1 and E.norm_reg(p[1].strip()) == "ra":
            n += 1
    return n

if __name__ == "__main__":
    main()
