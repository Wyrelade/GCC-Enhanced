#!/usr/bin/env python3
"""Deterministic, target-guided .s normalizer for the C build.

Some functions compile to assembly that is semantically identical to the retail
target but differs in a few register-allocation or commutative-operand choices
that modern GNU as does not reproduce the same way the original PSY-Q toolchain
did. For those functions this pass rewrites the cc1 gas ``.s`` (between cc1 and
the assembler) so the emitted words match the retail bytes.

It is NOT a search: every pass is a deterministic rewrite, guided by the
in-repo retail disassembly (``asm/**/nonmatchings/**/<func>.s``). A manifest
(``asm_normalizer_manifest.json`` next to this file) lists, per function, the
ordered pass names to replay. Functions not in the manifest are left byte for
byte verbatim; an empty/absent manifest makes the whole pass a no-op.

The build relies on the final linked SHA-1 as the ground-truth gate: if a
recorded rewrite ever produced the wrong bytes the checksum would fail loudly.

No function names or repository paths are baked into this module. The manifest
supplies the function names; the caller supplies the toolchain paths and the
retail-asm root via a small context object.
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "asm_normalizer_manifest.json")

# --------------------------------------------------------------------------
# MIPS register table (self-contained; O32 ABI order)
# --------------------------------------------------------------------------
NUM2ABI = ["zero", "at", "v0", "v1", "a0", "a1", "a2", "a3",
           "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7",
           "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
           "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra"]
ABI2NUM = {n: i for i, n in enumerate(NUM2ABI)}
ABI2NUM["s8"] = 30           # fp alias
ABI2NUM["r0"] = 0


def norm_reg(tok):
    """'$a2' / '$4' / 'a2' / '$s8' -> canonical ABI name; a bare number is an
    IMMEDIATE, not a register (objdump prints small shift amounts as bare
    decimals like `sll v0,v0,8`), so only a $-prefixed number names a register."""
    s = tok.strip()
    had_dollar = s.startswith("$")
    t = s.lstrip("$")
    if t in ABI2NUM:
        return NUM2ABI[ABI2NUM[t]]
    if had_dollar and re.fullmatch(r"\d+", t) and int(t) < 32:
        return NUM2ABI[int(t)]
    if re.fullmatch(r"r\d+", t) and int(t[1:]) < 32:
        return NUM2ABI[int(t[1:])]
    return None


# --------------------------------------------------------------------------
# operand parsing on objdump / retail disassembly
# --------------------------------------------------------------------------
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
        if mm and norm_reg(mm.group(2)) is not None:
            skel.append(mm.group(1).strip() + "(#)")
            regs.append(norm_reg(mm.group(2)))
        else:
            r = norm_reg(op)
            if r is not None:
                regs.append(r)
                skel.append("#")
            else:
                skel.append(op)
    return mnem, regs, skel


# --------------------------------------------------------------------------
# reg-realloc: derive the register correspondence the target's naming implies
# and apply it to the cc1 gas .s
# --------------------------------------------------------------------------
def derive_sigma(our, tgt):
    """From positions whose mnemonic + non-register skeleton already agree, unify
    the register operands to build sigma. Returns (sigma dict, conflicts list)."""
    sigma, conflicts = {}, []
    for (gw, gd), (tw, td) in zip(our, tgt):
        gm, gr, gs = insn_parts(gd)
        tm, tr, ts = insn_parts(td)
        if gm != tm or len(gr) != len(tr):
            continue
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


def apply_sigma_to_s(stext, sigma):
    """Rename $N registers in cc1 gas assembly simultaneously per sigma (ABI->ABI)."""
    num = {}
    for a, b in sigma.items():
        if a in ABI2NUM and b in ABI2NUM:
            num[ABI2NUM[a]] = ABI2NUM[b]

    def repl(m):
        n = int(m.group(1))
        return "$%d" % num.get(n, n)

    return re.sub(r"\$(\d+)\b", repl, stext)


# --------------------------------------------------------------------------
# commutative-operand-swap: transpose the two source operands of a commutative
# insn whose destination already matches the target but whose rs/rt order does
# not (a field order the global register map cannot express)
# --------------------------------------------------------------------------
_COMMUTATIVE_ACC = {"addu", "add", "and", "or", "xor", "nor"}
_ACC_S = re.compile(r"^(\s*)(\w+)\s+(\$\d+)\s*,\s*(\$\d+)\s*,\s*(\$\d+)\s*$")


def _s_is_insn(line):
    s = line.strip()
    if not s or s.startswith((".", "#")) or s.endswith(":"):
        return False
    return True


def _s_insn_lines(lines):
    return [i for i, l in enumerate(lines) if _s_is_insn(l)]


def _abi_to_num(tok):
    r = norm_reg(tok)
    return ("$%d" % ABI2NUM[r]) if r in ABI2NUM else None


def _commutatives_numeric(pairs, to_num):
    """List of (op, dst, rs, rt) commutative 3-register insns with both sources
    real non-zero registers, in order. Excludes the `op rd,rs,zero` move-pseudo
    form so the two sides stay positionally aligned."""
    zero = "$%d" % ABI2NUM["zero"]
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
    """Transpose the two source operands of a commutative insn whose destination
    already matches the target but whose rs/rt order does not. stext is
    sigma-applied (numeric register space). Returns new text (unchanged if the
    transform does not fire)."""
    tgt_c = _commutatives_numeric(tgt, _abi_to_num)
    if not tgt_c:
        return stext
    lines = stext.split("\n")
    insn_ix = _s_insn_lines(lines)
    zero = "$%d" % ABI2NUM["zero"]
    our = []
    for i in insn_ix:
        m = _ACC_S.match(lines[i])
        if m and m.group(2).lower() in _COMMUTATIVE_ACC \
                and m.group(4) != zero and m.group(5) != zero:
            our.append((i, m.group(2).lower(), m.group(3), m.group(4), m.group(5)))
    if len(our) != len(tgt_c):
        return stext
    changed = False
    for (li, op, od, os_, ot), (top, td, ts, tt) in zip(our, tgt_c):
        if op != top or od != td:
            continue
        if os_ == ts and ot == tt:
            continue
        if os_ == tt and ot == ts:
            indent = lines[li][:len(lines[li]) - len(lines[li].lstrip())]
            lines[li] = "%s%s\t%s,%s,%s" % (indent, op, od, ts, tt)
            changed = True
    return "\n".join(lines) if changed else stext


# --------------------------------------------------------------------------
# la-form fold: rewrite the split form (`lui $B,%hi(SYM)` + `%lo(SYM)($B)`) into
# the la form (`lui;addiu $B,$B,%lo(SYM)` + `0($B)`). Both compute the same
# address; the original toolchain prefers la-form for some single-symbol
# globals. Target-guided: only fold symbols the target materializes with
# `addiu R,R,%lo(SYM)`.
# --------------------------------------------------------------------------
def target_laform_syms(tgt):
    """Symbols the target builds via an explicit la (lui;addiu %lo) then 0(base)."""
    syms = set()
    pat = re.compile(r"addiu\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*%lo\(([\w.$]+)\)")
    for _, dis in tgt:
        m = pat.search(dis)
        if m and norm_reg(m.group(1)) == norm_reg(m.group(2)):
            syms.add(m.group(3))
    return syms


def laform_fold_s(stext, syms):
    """In the cc1 gas .s, rewrite `lui $B,%hi(SYM)` + `%lo(SYM)($B)` split-form
    into la-form: insert `addiu $B,$B,%lo(SYM)` after the lui and change each
    `%lo(SYM)($B)` operand to `0($B)`. Only fires for SYM in `syms` and only when
    every use of $B in its live range is a `%lo(SYM)($B)` memory operand of that
    same SYM, so materializing the address and zeroing the offsets preserves
    every access. Rewrites one lui/base at a time."""
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
            j = i + 1
            legal = True
            hits = []
            while j < len(lines):
                body = lines[j].split("#", 1)[0]
                mdef = re.match(r"\s*[a-z][\w.]*\s+" + re.escape(base) + r"\s*,", body)
                if mdef:
                    break
                if reg_re.search(body):
                    if lo_here.search(body):
                        hits.append(j)
                    else:
                        legal = False
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


def laform_fold_pass(stext, tgt):
    return laform_fold_s(stext, target_laform_syms(tgt))


# --------------------------------------------------------------------------
# un-hi-cse (rematerialize): our cc1 hoists a global's `lui %hi` once and shares
# the base across several `%lo` loads; retail's cc1 rematerializes the `lui %hi`
# at EACH use (`lui R,%hi(S); lw R,%lo(S)(R)`, base == dest). This is a COUNT-
# changing pass (adds a lui per use, drops the shared hoist), so it must run
# BEFORE sigma derivation so the register correspondence is built from the
# count-aligned assembly. Target-guided (only symbols the target rematerializes).
# --------------------------------------------------------------------------
_LUI_HI = re.compile(r"^(\s*)lui\s+(\$\d+)\s*,\s*%hi\(([\w.$]+)\)")
_LW_LO = re.compile(r"^(\s*)lw\s+(\$\d+)\s*,\s*%lo\(([\w.$]+)\)\((\$\d+)\)")


def _is_insn_line(line):
    s = line.strip()
    if not s or s.startswith("#") or s.startswith(".") or s.endswith(":"):
        return False
    return bool(re.match(r"[a-z][\w.]*", s))


def _line_mnem(line):
    return line.strip().split(None, 1)[0].lower()


def target_remat_syms(tgt):
    """Symbols the target materializes inline as `lui R,%hi(S); lw R,%lo(S)(R)`
    (base == dest) -- the globals whose %hi retail rematerializes per access."""
    syms = set()
    lui_re = re.compile(r"lui\s+(\$?\w+)\s*,\s*%hi\(([\w.$]+)\)")
    lw_re = re.compile(r"lw\s+(\$?\w+)\s*,\s*%lo\(([\w.$]+)\)\((\$?\w+)\)")
    for i in range(len(tgt) - 1):
        a = lui_re.search(tgt[i][1])
        b = lw_re.search(tgt[i + 1][1])
        if not a or not b:
            continue
        rd, ld, lb = norm_reg(a.group(1)), norm_reg(b.group(1)), norm_reg(b.group(3))
        if a.group(2) == b.group(2) and rd == ld == lb:
            syms.add(a.group(2))
    return syms


def un_hi_cse_s(stext, remat_syms):
    """Un-CSE the %hi base of each `remat_syms` symbol: rematerialize `lui;lw` at
    each use and drop the shared hoist. Returns new text (unchanged if inert)."""
    if not remat_syms:
        return stext
    lines = stext.split("\n")
    # only un-CSE a symbol our cc1 actually HOISTED (`lw $D,%lo(S)($B)`, dest != base);
    # one already in base==dest remat form must not be duplicated.
    hoisted = set()
    for l in lines:
        m = _LW_LO.match(l)
        if m and m.group(3) in remat_syms and m.group(2) != m.group(4):
            hoisted.add(m.group(3))
    remat_syms = hoisted
    if not remat_syms:
        return stext
    region_of, reg, first_insn_line, saw = {}, 0, {}, False
    for idx, ln in enumerate(lines):
        if not _is_insn_line(ln):
            continue
        region_of[idx] = reg
        first_insn_line.setdefault(reg, idx)
        m = _LW_LO.match(ln)
        if m and m.group(3) in remat_syms:
            saw = True
        if _line_mnem(ln) in _STORE_MN:
            reg += 1
    if not saw:
        return stext
    inserts, delete = {}, set()
    for idx, ln in enumerate(lines):
        if idx not in region_of:
            continue
        mlui = _LUI_HI.match(ln)
        if mlui and mlui.group(3) in remat_syms:
            delete.add(idx)
            continue
        mlw = _LW_LO.match(ln)
        if mlw and mlw.group(3) in remat_syms:
            indent, dst, sym = mlw.group(1) or "\t", mlw.group(2), mlw.group(3)
            inserts.setdefault(region_of[idx], []).append(
                ["%slui\t%s,%%hi(%s)" % (indent, dst, sym),
                 "%slw\t%s,%%lo(%s)(%s)" % (indent, dst, sym, dst)])
            delete.add(idx)
    out, emitted = [], set()
    for idx, ln in enumerate(lines):
        r = region_of.get(idx)
        if r is not None and r not in emitted and idx == first_insn_line[r]:
            for pair in inserts.get(r, []):
                out.extend(pair)
            emitted.add(r)
        if idx in delete:
            continue
        out.append(ln)
    return "\n".join(out)


def un_hi_cse_pass(stext, tgt):
    return un_hi_cse_s(stext, target_remat_syms(tgt))


# --------------------------------------------------------------------------
# exit-merge: cc1 emits each return as its own `j $31` (with the return value in
# the delay slot); retail funnels every return through ONE shared `jr $ra` laid
# last, other paths reaching it by `j <EXIT>`. Rewrite the cc1 source so its
# layout matches. Also unfill any conditional-branch delay the target leaves as a
# nop. Count-changing (adds the exit block), so it runs BEFORE sigma. Fires only
# when target is single-exit and ours is not.
# --------------------------------------------------------------------------
_EXIT_LABEL = "$Lgcce_exit"
_RET_S = re.compile(r"^\s*(?:j\s+\$31|jr\s+\$(?:31|ra))\b")
_COND_S = re.compile(r"^(beq|bne|blez|bgtz|bltz|bgez|beqz|bnez)\b")


def _s_is_label(l):
    s = l.strip()
    return s.endswith(":") and not s.startswith(".")


def _s_mnem(l):
    return l.strip().split(None, 1)[0].lower()


def _s_is_return(l):
    return bool(_RET_S.match(l))


def _count_returns_words(words):
    n = 0
    for _, dis in words:
        p = dis.split(None, 1)
        if p[0].lower() == "jr" and len(p) > 1 and norm_reg(p[1].strip()) == "ra":
            n += 1
    return n


def merge_exits_s(stext):
    """Fold cc1's multiple `j $31` returns into one shared exit. Returns new text."""
    lines = stext.split("\n")
    rets = [i for i, l in enumerate(lines) if _s_is_return(l)]
    if len(rets) < 2:
        return stext
    last_i = rets[-1]
    out = []
    for i, l in enumerate(lines):
        if i == last_i:
            continue                       # drop the last return (keep its delay insn)
        if _s_is_return(l):
            out.append(re.sub(r"(?:j|jr)\s+\$(?:31|ra)\b",
                              "j\t%s" % _EXIT_LABEL, l, count=1))
        else:
            out.append(l)
    ins = len(out)
    for k in range(len(out) - 1, -1, -1):
        if out[k].strip().startswith(".end"):
            ins = k
            break
    exitblk = ["%s:" % _EXIT_LABEL, "\t.set\tnoreorder", "\t.set\tnomacro",
               "\tj\t$31", "\tnop", "\t.set\tmacro", "\t.set\treorder"]
    return "\n".join(out[:ins] + exitblk + out[ins:])


def _target_cond_delay_nop(tgt):
    res = []
    for i, (_, dis) in enumerate(tgt):
        if _COND_S.match(dis.split(None, 1)[0].lower()):
            nxt = tgt[i + 1][1] if i + 1 < len(tgt) else ""
            res.append(is_nop(nxt))
    return res


def unfill_cond_delay_s(stext, tgt):
    """Where the target's k-th conditional branch has a nop delay but our cc1 filled
    it, move our fill out to just after the branch and restore the nop."""
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
                    d = None
                    break
                d += 1
            filled = d is not None and not _src_is_nop(lines[d])
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
                    tail.append(lines[m])
                    m += 1
                result.extend(tail)
                result.append(displaced)
                idx = m
                continue
        result.append(l)
        idx += 1
    return "\n".join(result)


def exit_merge_pass(stext, tgt):
    """Combined pre-sigma pass: when the target is single-exit and ours is not,
    unfill any target-nop conditional delay, then funnel our returns to one exit."""
    if _count_returns_words(tgt) != 1:
        return stext
    our_rets = sum(1 for l in stext.split("\n") if _s_is_return(l))
    if our_rets <= 1:
        return stext
    stext = unfill_cond_delay_s(stext, tgt)
    return merge_exits_s(stext)


# --------------------------------------------------------------------------
# operand-recolor: a commutative accumulate our cc1 computes into the INDEX temp
# while the target computes into the BASE temp (`op $I,$I,$B` vs `op $B,$B,$I`).
# The op is commutative so the value and the memory effect are identical and both
# temps are caller-saved (dead at exit); the only difference is which dead temp
# carries the address. sigma cannot express it (the temps keep their identity
# everywhere else), so it is a LOCAL recolor: rewrite the accumulate to the
# target's destination and rename our old destination downstream until it is
# redefined. Target-guided; bail if the base temp is written before the old
# index temp dies (unsafe merge).
# --------------------------------------------------------------------------
def _accs_numeric(pairs, to_num):
    """(op, dst, rs, rt) commutative ACCUMULATES (dst==rs, rt a real non-zero reg)
    in order. The accumulate form keeps both sides aligned (a `op rd,rs,zero`
    renders as `move` in cc1 but `addu` in the target)."""
    zero = "$%d" % ABI2NUM["zero"]
    out = []
    for _, dis in pairs:
        m = re.match(r"(\w+)\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*(\$?\w+)\s*$", dis)
        if not m or m.group(1).lower() not in _COMMUTATIVE_ACC:
            continue
        d, s, t = to_num(m.group(2)), to_num(m.group(3)), to_num(m.group(4))
        if d and s and t and d == s and t != zero:
            out.append((m.group(1).lower(), d, s, t))
    return out


def operand_recolor_s(stext, tgt):
    """Recolor a commutative accumulate our cc1 computes into the index temp but the
    target computes into the base temp (a local dead-temp swap). Returns new text
    (unchanged if it does not fire). stext is sigma-applied (numeric reg space)."""
    tgt_accs = _accs_numeric(tgt, _abi_to_num)
    if not tgt_accs:
        return stext
    lines = stext.split("\n")
    insn_ix = _s_insn_lines(lines)
    zero = "$%d" % ABI2NUM["zero"]
    our_accs = []
    for i in insn_ix:
        m = _ACC_S.match(lines[i])
        if m and m.group(2).lower() in _COMMUTATIVE_ACC \
                and m.group(3) == m.group(4) and m.group(5) != zero:
            our_accs.append((i, m.group(2).lower(), m.group(3), m.group(4), m.group(5)))
    if len(our_accs) != len(tgt_accs):
        return stext
    for (li, op, od, os_, ot), (top, td, ts, tt) in zip(our_accs, tgt_accs):
        if op != top:
            continue
        # target accumulates S into B (`op $B,$B,$S`, td==ts, S==tt); ours is the
        # swapped `op $S,$S,$B` (od==os_==tt, ot==td). Recolor od(=$S) -> $B(td).
        if not (td == ts and od == os_ and od == tt and ot == td):
            continue
        newB, oldI = td, od
        indent = lines[li][:len(lines[li]) - len(lines[li].lstrip())]
        safe, redef_at = True, None
        for j in insn_ix:
            if j <= li:
                continue
            mm = re.match(r"\s*([a-z][\w.]*)\s+(\$\d+)\b", lines[j])
            if not mm:
                continue
            mn, dst = mm.group(1).lower(), mm.group(2)
            is_store = mn in _STORE_MN or mn.startswith(("sb", "sh", "sw"))
            if not is_store and dst == newB:
                safe = False
                break
            if not is_store and dst == oldI:
                redef_at = j
                break
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


PASSES = {
    # reg_realloc is applied specially (it needs sigma from words); the ordered
    # list in the manifest still names it so the recipe is explicit and auditable.
    "reg_realloc": None,
    "un_hi_cse": un_hi_cse_pass,
    "exit_merge": exit_merge_pass,
    "commutative_swap": commutative_swap_s,
    "operand_recolor": operand_recolor_s,
    "laform": laform_fold_pass,
}

# Passes that CHANGE the instruction count and so must run BEFORE sigma is derived
# (the register correspondence has to be built from count-aligned assembly). Any
# such pass listed in a recipe is applied ahead of reg_realloc regardless of the
# manifest order; the rest keep their listed order after sigma.
PRE_SIGMA_PASSES = ("un_hi_cse", "exit_merge")

# --------------------------------------------------------------------------
# WORD-LEVEL passes. The original PSY-Q assembler (aspsx) scheduled branch and
# load delay slots in ways modern GNU as does not reproduce, so a few classes
# cannot be expressed by a register/operand text rewrite alone: the same source
# assembles to a differently-scheduled word stream. These passes decide the fix
# on the ASSEMBLED words (guided by the retail target) and re-emit the source so
# the assembler keeps the schedule we chose:
#   * padnop           -- append the trailing section-alignment nop(s) the retail
#                         object carried after the function but the C compile omits
#                         (pure text append; the count comes from the target file).
#   * delay_fill       -- move the instruction before a branch/jr into its (nop)
#                         delay slot, matching aspsx's fill (target-guided).
#   * epilogue_unfill  -- move the stack-dealloc out of the jr delay slot to just
#                         before the jr, restoring an empty slot (target-guided).
# delay_fill / epilogue_unfill reorder instructions, so their span is re-emitted
# under `.set noreorder` (one instruction per line, verbatim source) to stop the
# assembler re-scheduling it. padnop needs neither (a trailing nop cannot be
# rescheduled).
WORD_PASSES = ("delay_fill", "epilogue_unfill", "reorder_indep", "padnop",
               "prologue_save_hoist")

# minimal register def/use for the delay_fill dependency guard (gas `$reg` syntax)
_STORE_MN = {"sb", "sh", "sw", "swl", "swr", "swc1", "swc2", "sd", "sdc1", "sdc2"}
_COND_BR_MN = {"beq", "bne", "blez", "bgtz", "bltz", "bgez", "beqz", "bnez",
               "bgtzl", "blezl", "bltzl", "bgezl", "beql", "bnel"}
_UNCOND_MN = {"b", "j"}
_CALL_CLOBBER = {"v0", "v1", "a0", "a1", "a2", "a3", "t0", "t1", "t2", "t3",
                 "t4", "t5", "t6", "t7", "t8", "t9", "ra", "at", "hi", "lo"}


def _reg(tok):
    r = norm_reg(tok)
    return r if r and r != "zero" else None


def defs_uses(disasm):
    """(defs, uses) register sets for one instruction's disassembly. Conservative:
    only what delay_fill needs (does the moved insn write a register the branch
    reads). Base register of a `off($b)` memory operand counts as a use."""
    p = disasm.split(None, 1)
    if not p:
        return set(), set()
    op = p[0].lower()
    ops = split_ops(p[1]) if len(p) > 1 else []
    defs, uses = set(), set()

    def use(tok):
        r = _reg(tok)
        if r:
            uses.add(r)
        mm = re.fullmatch(r"(.*)\((\$?\w+)\)", tok.strip())
        if mm:
            r2 = _reg(mm.group(2))
            if r2:
                uses.add(r2)

    if op in ("jal", "bal", "jalr", "bltzal", "bgezal"):
        defs |= set(_CALL_CLOBBER)
        uses |= {"a0", "a1", "a2", "a3"}
        if op == "jalr" and ops:
            use(ops[-1])
        return defs, uses
    if op in ("mult", "multu", "div", "divu"):
        for t in ops:
            use(t)
        defs |= {"hi", "lo"}
        return defs, uses
    if op == "mflo" or op == "mfhi":
        d = _reg(ops[0]) if ops else None
        if d:
            defs.add(d)
        uses.add("lo" if op == "mflo" else "hi")
        return defs, uses
    if op == "mtlo" or op == "mthi":
        if ops:
            use(ops[0])
        defs.add("lo" if op == "mtlo" else "hi")
        return defs, uses
    if op in _STORE_MN:
        for t in ops:
            use(t)
        return defs, uses
    if op in _COND_BR_MN:
        for t in ops[:-1]:
            use(t)
        return defs, uses
    if op in _UNCOND_MN:
        return defs, uses
    if op == "jr":
        if ops:
            use(ops[0])
        return defs, uses
    # default ALU/shift/load/lui/li/la/move: first operand is the destination.
    if ops:
        d = _reg(ops[0])
        if d:
            defs.add(d)
        for t in ops[1:]:
            use(t)
    return defs, uses


_BRANCHY = re.compile(r"^(b|j|jr|jal|beq|bne|blez|bgtz|bltz|bgez|beqz|bnez)")


def is_branch(disasm):
    return bool(_BRANCHY.match(disasm.split(None, 1)[0].lower()))


def is_nop(disasm):
    return disasm.strip() in ("nop", "sll zero,zero,0", "sll zero,zero,0x0")


def _target_branch_filled(tgt):
    """For each control-flow instruction in the target (in order), whether its
    delay slot holds a real instruction (True) or a nop (False)."""
    res = []
    for i, (_, dis) in enumerate(tgt):
        if is_branch(dis):
            nxt = tgt[i + 1][1] if i + 1 < len(tgt) else ""
            res.append(bool(nxt.strip()) and not is_nop(nxt))
    return res


def delay_fill(words, tgt):
    """Fill a branch/jr delay slot (currently nop) with the immediately-preceding
    independent instruction, dropping the nop -- the aspsx schedule. Target-guided:
    fill the k-th branch's slot only if the target's k-th branch is filled. Returns
    a new (word,disasm) list (possibly shorter)."""
    want = _target_branch_filled(tgt)
    out = list(words)
    i, k = 0, -1
    while i < len(out) - 1:
        dis = out[i][1]
        if is_branch(dis):
            k += 1
            if is_nop(out[i + 1][1]) and i >= 1 and k < len(want) and want[k]:
                prev = out[i - 1]
                if not is_branch(prev[1]) and not is_nop(prev[1]):
                    pdef = defs_uses(prev[1])[0]
                    bruse = defs_uses(dis)[1]
                    if not (pdef & bruse):
                        out[i - 1], out[i] = out[i], prev
                        del out[i + 1]
                        i += 1
                        continue
        i += 1
    return out


def _sp_delta(dis):
    m = re.match(r"addiu\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*(-?(?:0x)?[0-9a-fA-F]+)\s*$", dis)
    if not m or norm_reg(m.group(1)) != "sp" or norm_reg(m.group(2)) != "sp":
        return None
    try:
        return int(m.group(3), 0)
    except ValueError:
        return None


def _is_ret(dis):
    p = dis.split(None, 1)
    return len(p) > 1 and p[0].lower() == "jr" and norm_reg(p[1].strip()) == "ra"


def _target_wants_sp_before_jr(tgt):
    if len(tgt) < 2 or not _is_ret(tgt[-1][1]):
        return False
    d = _sp_delta(tgt[-2][1])
    return d is not None and d > 0


def epilogue_unfill(words, tgt):
    """Move a positive `addiu $sp` out of the jr delay slot to just before the jr,
    restoring an empty slot -- target-guided. Returns a new word list."""
    if not _target_wants_sp_before_jr(tgt):
        return words
    out = list(words)
    for j in range(len(out) - 1):
        if not _is_ret(out[j][1]):
            continue
        d = _sp_delta(out[j + 1][1])
        if d is None or d <= 0:
            continue
        if j >= 1 and is_nop(out[j - 1][1]):
            out[j - 1], out[j + 1] = out[j + 1], out[j - 1]
        else:
            insn = out.pop(j + 1)
            out.insert(j, insn)
            out.insert(j + 2, ("00000000", "nop"))
        break
    return out


# --------------------------------------------------------------------------
# word-level re-emission: after a reordering word pass, emit the span verbatim
# under `.set noreorder` so the assembler keeps the schedule we chose. The body
# is taken from the ORIGINAL source lines reordered to the new schedule (so
# relocations and symbol operands survive), not from disassembly.
# --------------------------------------------------------------------------
def target_words_full(asm_root, fn):
    """Every word in the target .s file for fn, including the trailing pad after
    endlabel (section-alignment nops). Returns a list of big-endian hex, or None."""
    p = find_target_s(asm_root, fn)
    if not p:
        return None
    pat = re.compile(r'/\*\s*[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+([0-9A-Fa-f]{8})\s+\*/')
    out = []
    for line in open(p, encoding="utf-8", errors="replace"):
        m = pat.search(line)
        if m:
            le = m.group(1)
            out.append((le[6:8] + le[4:6] + le[2:4] + le[0:2]).lower())
    return out


# --- source-line helpers for the reordering passes -------------------------
# The reordering passes rewrite the cc1 gas SOURCE (not the disassembly) so
# relocations and symbol operands survive, then wrap the touched instructions in
# `.set noreorder` / `.set reorder` so the assembler keeps the schedule.
def _src_indent(line):
    return line[:len(line) - len(line.lstrip())]


def _src_reg(tok):
    return norm_reg(tok.strip())


def _src_is_ret(line):
    """Source return: `j $31` / `jr $31` / `jr $ra` (cc1 emits `j $31`)."""
    m = re.match(r"\s*(j|jr)\s+(\$\w+)\s*$", line)
    return bool(m and _src_reg(m.group(2)) == "ra")


def _src_sp_delta(line):
    """`addu/addiu $sp,$sp,IMM` -> signed IMM, else None (cc1 uses addu here)."""
    m = re.match(r"\s*addi?u\s+(\$\w+)\s*,\s*(\$\w+)\s*,\s*(-?(?:0x)?[0-9a-fA-F]+)\s*$",
                 line)
    if not m or _src_reg(m.group(1)) != "sp" or _src_reg(m.group(2)) != "sp":
        return None
    try:
        return int(m.group(3), 0)
    except ValueError:
        return None


def epilogue_unfill_src(span, tgt):
    """Move the stack-dealloc (`addu $sp,$sp,+N`) out of the return's delay slot to
    just before the return, restoring an empty (nop) slot, and wrap the three
    instructions in `.set noreorder`. Target-guided: only when the target ends
    `addiu $sp,$sp,+N ; jr $ra ; (nop)`. Returns (new_span, fired)."""
    if not _target_wants_sp_before_jr(tgt):
        return span, False
    lines = span.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    for k in range(len(ins) - 1):
        li, ll = ins[k]
        if not _src_is_ret(ll):
            continue
        ni, nl = ins[k + 1]
        if _src_sp_delta(nl) is None or _src_sp_delta(nl) <= 0:
            continue
        indent = _src_indent(ll)
        block = [indent + ".set\tnoreorder",
                 indent + nl.strip(),          # dealloc moved before the return
                 ll,                            # the return (jr/j $ra)
                 indent + "nop",                # empty delay slot
                 indent + ".set\treorder"]
        out = []
        for idx, l in enumerate(lines):
            if idx == li:
                out.extend(block)
            elif idx == ni:
                continue                        # dealloc removed from the slot
            else:
                out.append(l)
        return "\n".join(out), True
    return span, False


def _src_is_branch(line):
    m = re.match(r"\s*([a-z]+)\b", line)
    if not m:
        return False
    mn = m.group(1)
    if mn in ("j", "jr") and re.search(r"\$\w+", line):
        return True             # `j $31` / `jr $ra` (register form)
    return is_branch(mn + " ")


def _src_is_nop(line):
    return line.strip() in ("nop", "sll\t$0,$0,0", "sll $0,$0,0")


def delay_fill_src(span, tgt, our_words):
    """Fill a branch/return delay slot (currently a nop after assembly) with the
    immediately-preceding independent instruction -- the aspsx schedule GNU as does
    not reproduce. Target-guided: fill the k-th branch only if the target's k-th
    branch is filled AND our assembled k-th branch slot is a nop. The moved
    instruction and the branch are wrapped in `.set noreorder`. Uses `our_words`
    (assembled) to locate nop slots precisely; the k-th source branch matches the
    k-th assembled branch (branches never macro-expand). Returns (new_span, fired)."""
    want = _target_branch_filled(tgt)
    # our_words: which assembled branch slots are currently nops
    our_slot_nop = []
    for bi, (_, dis) in enumerate(our_words):
        if is_branch(dis):
            nxt = our_words[bi + 1][1] if bi + 1 < len(our_words) else ""
            our_slot_nop.append(is_nop(nxt) or not nxt.strip())
    lines = span.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    drop = set()
    move = {}                   # branch_line_idx -> (prev_line_idx, prev_text)
    kbr = -1
    for p in range(len(ins)):
        i, l = ins[p]
        if not _src_is_branch(l):
            continue
        kbr += 1
        if kbr >= len(want) or not want[kbr]:
            continue
        if kbr >= len(our_slot_nop) or not our_slot_nop[kbr]:
            continue            # target fills but our slot already holds a real insn
        if p < 1:
            continue
        pi, pl = ins[p - 1]
        if _src_is_branch(pl) or _src_is_nop(pl):
            continue
        # dependency: prev must not define a register the branch reads
        bmn = re.match(r"\s*([a-z]+)", l).group(1)
        if not (defs_uses(pl.strip())[0] & defs_uses(l.strip())[1]):
            move[i] = pl.strip()
            drop.add(pi)
            # if the source already materialized a nop right after the branch, drop it
            if p + 1 < len(ins) and _src_is_nop(ins[p + 1][1]):
                drop.add(ins[p + 1][0])
    if not move:
        return span, False
    out = []
    for idx, l in enumerate(lines):
        if idx in drop:
            continue
        if idx in move:
            indent = _src_indent(l)
            out.append(indent + ".set\tnoreorder")
            out.append(l)                       # the branch
            out.append(indent + move[idx])      # preceding insn -> delay slot
            out.append(indent + ".set\treorder")
        else:
            out.append(l)
    return "\n".join(out), True


_PURE_ALU = set((
    "lui li la addiu addu addi ori xori andi and or xor sll srl sra sllv "
    "srlv srav slt slti sltu sltiu subu sub move nor").split())


def _pure_alu(disasm):
    """A register-only instruction with no memory or control-flow effect, so two
    adjacent ones may be swapped when register-independent (no aliasing to reason
    about)."""
    p = disasm.split(None, 1)
    return bool(p) and p[0].lower() in _PURE_ALU


def _frame_store(disasm):
    """A store whose base register is the stack pointer, e.g. `sw $ra,0x10($sp)`.
    Such a store touches only the current frame; swapping it past a pure-ALU insn
    (which has no memory effect) can never alias, so the pair is safe to reorder
    when register-independent. Retail's cc1 emits the prologue register-save store
    right after the `addiu $sp` allocation, before evaluating call arguments; our
    cc1 interleaves the arg setup ahead of the save, so this lets reorder_indep
    hoist the save back to the target's position."""
    p = disasm.split(None, 1)
    if not p or p[0].lower() not in _STORE_MN or len(p) < 2:
        return False
    ops = split_ops(p[1])
    if not ops:
        return False
    mm = re.fullmatch(r".*\((\$?\w+)\)", ops[-1].strip())
    return bool(mm) and _reg(mm.group(1)) == "sp"


def _reorder_swappable(a, b):
    """True when adjacent instructions a,b may be transposed on aliasing grounds
    (register-independence is checked separately). Two pure-ALU insns qualify (no
    memory, no control flow); so does a frame store paired with a pure-ALU insn."""
    pa, pb = _pure_alu(a), _pure_alu(b)
    if pa and pb:
        return True
    return (pa and _frame_store(b)) or (pb and _frame_store(a))


def _word_eq(our_w, tgt_w, tgt_dis):
    """Words equal, tolerating a relocated low-16 immediate: when the target insn
    carries a `%hi`/`%lo`/`%gp_rel` relocation its low half is symbol-dependent
    (our un-relinked assembly leaves it 0), so compare opcode+register fields only."""
    if our_w == tgt_w:
        return True
    if "%hi" in tgt_dis or "%lo" in tgt_dis or "%gp_rel" in tgt_dis:
        return (int(our_w, 16) >> 16) == (int(tgt_w, 16) >> 16)
    return False


def _indep(a, b):
    """Two instructions have no register data dependency in either direction and do
    not both write the same register, so their relative order does not affect
    results."""
    ad, au = defs_uses(a)
    bd, bu = defs_uses(b)
    return not ((ad & bu) or (bd & au) or (ad & bd))


def reorder_indep_src(span, tgt, our_words):
    """Reorder independent instructions to match the target's schedule. Retail's cc1
    sometimes emits data-independent instructions in a different order to ours -- an
    immediate load and a `lui` address setup transposed, or (the common wrapper case)
    the prologue register-save `sw $ra,K($sp)` scheduled ahead of the call-argument
    setup where our cc1 interleaves it later. Relocating one instruction across a run
    of others it is independent of and cannot alias is semantics-preserving.

    Target-guided and left-to-right: at the first position whose assembled word does
    not match the target, find the later instruction that belongs there and move it
    up, provided every instruction it passes is pairwise reorderable (`_reorder_swappable`
    -- pure-ALU pairs, or a frame store paired with pure-ALU) and register-independent
    of it. One instruction relocated per step (a single adjacent swap is the length-1
    case). Stops at the first position it cannot resolve so a later pass (e.g.
    epilogue_unfill) owns the tail. Requires a 1:1 source-insn to word mapping (no
    macro expansion). The touched positions are wrapped in `.set noreorder` so the
    assembler keeps the schedule. Returns (new_span, fired).

    Our and target word counts may differ (e.g. our epilogue is still delay-filled
    while the target's is not); this pass only aligns the common prefix and stops at
    the first position it cannot resolve, leaving the tail to epilogue_unfill."""
    n = len(our_words)
    lines = span.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    if len(ins) != n:                       # macro expansion -> mapping unsafe
        return span, False
    ow = list(our_words)
    order = list(range(n))                  # order[p] = original source index at p
    swapped = set()
    pos = 0
    while pos < n - 1 and pos < len(tgt):
        if _word_eq(ow[pos][0], tgt[pos][0], tgt[pos][1]):
            pos += 1
            continue
        # find the instruction that belongs at `pos` further down the stream
        j = None
        for k in range(pos + 1, n):
            if _word_eq(ow[k][0], tgt[pos][0], tgt[pos][1]):
                j = k
                break
        if j is None:
            break                           # word not produced by a mere reorder
        mover = ow[j][1]
        # every instruction the mover hops over must be safe + independent
        if not all(_reorder_swappable(mover, ow[k][1]) and _indep(mover, ow[k][1])
                   for k in range(pos, j)):
            break
        ow.insert(pos, ow.pop(j))
        order.insert(pos, order.pop(j))
        swapped |= set(range(pos, j + 1))
        pos += 1
    if not swapped:
        return span, False
    final = [ins[order[p]][1].strip() for p in range(n)]
    line_pos = {ins[p][0]: p for p in range(n)}
    out = []
    for idx, l in enumerate(lines):
        if idx in line_pos:
            p = line_pos[idx]
            indent = _src_indent(ins[p][1])
            if p in swapped and (p - 1) not in swapped:
                out.append(indent + ".set\tnoreorder")
            out.append(indent + final[p])
            if p in swapped and (p + 1) not in swapped:
                out.append(indent + ".set\treorder")
        else:
            out.append(l)
    return "\n".join(out), True


def _src_ra_save_idx(ins):
    """Index in the source insn list of the prologue `sw $ra,K($sp)` register-save,
    or None."""
    for p, l in enumerate(ins):
        m = re.match(r"\s*sw\s+(\$\w+)\s*,\s*-?(?:0x)?[0-9a-fA-F]+\((\$\w+)\)\s*$", l)
        if m and _src_reg(m.group(1)) == "ra" and _src_reg(m.group(2)) == "sp":
            return p
    return None


def _tgt_ra_save_idx(tgt):
    """Index in the target word stream of the `sw $ra,...($sp)` register-save, or
    None."""
    for i, (_w, dis) in enumerate(tgt):
        p = dis.split(None, 1)
        if not p or p[0].lower() != "sw":
            continue
        d, u = defs_uses(dis)
        if "ra" in u and "sp" in u:
            return i
    return None


def prologue_save_hoist_src(span, tgt):
    """Hoist the prologue register-save `sw $ra,K($sp)` up to the position the target
    puts it (immediately after the stack allocation). Retail's cc1 emits the save right
    after the `addiu/subu $sp` allocation, before evaluating the call arguments; our cc1
    schedules the argument setup first and saves `$ra` later. Moving the save earlier is
    semantics-preserving: it stores to the freshly allocated frame and the instructions
    it passes are register-independent of it and never touch memory (a frame store can
    only alias another memory op, and there are none between). The moved region is
    wrapped in `.set noreorder`. Target-guided: fires only when the target actually
    places the save earlier than we do, and only across hop-safe independent insns.
    Word-level 1:1 mapping is NOT required (unlike reorder_indep), so it is robust to
    the load-delay nops maspsx inserts later in the body. Returns (new_span, fired)."""
    desired = _tgt_ra_save_idx(tgt)
    if desired is None:
        return span, False
    lines = span.split("\n")
    idxs = [i for i, l in enumerate(lines) if _s_is_insn(l)]
    ins = [lines[i] for i in idxs]
    s_ra = _src_ra_save_idx(ins)
    if s_ra is None or desired >= s_ra:
        return span, False
    mover = ins[s_ra].strip()
    for k in range(desired, s_ra):
        if not (_reorder_swappable(mover, ins[k].strip()) and _indep(mover, ins[k].strip())):
            return span, False
    # rebuild the insn order with the save relocated to `desired`
    new_ins = list(ins)
    new_ins.insert(desired, new_ins.pop(s_ra))
    reordered = set(range(desired, s_ra + 1))
    out = []
    ip = 0
    for i, l in enumerate(lines):
        if i in idxs:
            indent = _src_indent(l)
            if ip in reordered and (ip - 1) not in reordered:
                out.append(indent + ".set\tnoreorder")
            out.append(indent + new_ins[ip].strip())
            if ip in reordered and (ip + 1) not in reordered:
                out.append(indent + ".set\treorder")
            ip += 1
        else:
            out.append(l)
    return "\n".join(out), True


_REORDER_SRC = {
    "epilogue_unfill": epilogue_unfill_src,
    "delay_fill": delay_fill_src,
    "reorder_indep": reorder_indep_src,
    "prologue_save_hoist": prologue_save_hoist_src,
}


def emit_noreorder_span(span, tgt, our_words, reorder_passes):
    """Apply the reordering source passes (delay_fill / epilogue_unfill) to a span
    in order. Each rewrites the cc1 source and wraps the touched instructions in
    `.set noreorder` so the assembler keeps the schedule. `tgt` is the target's
    (pad-stripped) [(word, disasm)] list; `our_words` is our assembled span."""
    txt = span
    for name in reorder_passes:
        fn = _REORDER_SRC.get(name)
        if fn is None:
            raise NotImplementedError("reorder pass not wired: %s" % name)
        if name in ("delay_fill", "reorder_indep"):
            txt, _fired = fn(txt, tgt, our_words)
        else:
            txt, _fired = fn(txt, tgt)
    return txt


def apply_padnop(span, our_words, tgt_full):
    """Append the trailing pad nop(s) the retail object carried but the C compile
    omitted. Count = target total words - our assembled words; only fires when the
    extra target words are all nop (a pure section-alignment pad)."""
    need = len(tgt_full) - len(our_words)
    if need <= 0:
        return span, False
    if any(w != "00000000" for w in tgt_full[len(our_words):]):
        return span, False          # tail is not pure padding: refuse
    m = re.search(r'(?m)^([ \t]*)\.end[ \t]+\S+', span)
    indent = m.group(1) if m else "\t"
    pad = "".join("%snop\n" % indent for _ in range(need))
    if m:
        return span[:m.start()] + pad + span[m.start():], True
    return span + pad, True


# --------------------------------------------------------------------------
# words: assemble our .s span, read the retail target .s
# --------------------------------------------------------------------------
def _strip_trailing_pad(words):
    """Drop trailing all-zero (`nop`/`sll zero,zero,0`) padding words the linker
    section alignment adds, so our and target lengths line up."""
    out = list(words)
    while out and out[-1][0] == "00000000":
        out.pop()
    return out


def assemble_words(ctx, sfile, fn):
    """gas .s -> maspsx --run-assembler -> objdump; return [(be_hex, disasm)] for fn."""
    obj = sfile + ".asmnorm.o"
    cmd = ([ctx["python"], ctx["maspsx_py"]] + ctx["maspsx_flags"]
           + ["--gnu-as-path=%s" % ctx["as_bin"]] + ctx["maspsx_as_flags"]
           + ["-o", obj, sfile])
    rc, o, e = ctx["run"](cmd)
    if rc:
        return None, "MASPSX:\n" + o + e
    # -z: do NOT collapse runs of zero words into `...`, or a trailing pad of
    # `nop`s (e.g. the alignment tail padnop appends) would be miscounted.
    rc, o, e = ctx["run"]([ctx["objdump"], "-dz", obj])
    if rc:
        return None, "OBJDUMP:\n" + o + e
    words, in_fn = [], False
    for line in o.splitlines():
        if re.match(r'^[0-9a-f]+ <%s>:' % re.escape(fn), line):
            in_fn = True
            continue
        if in_fn:
            m2 = re.match(r'^[0-9a-f]+ <([A-Za-z_.$][\w.$]*)>:', line)
            if m2 and m2.group(1) != fn and not re.match(r'^(L|\.L|LM|\$L)', m2.group(1)):
                break
            m = re.match(r'^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(.*)', line)
            if m:
                words.append((m.group(1).lower(), m.group(2).strip()))
    try:
        os.remove(obj)
    except OSError:
        pass
    return words, None


def find_target_s(asm_root, fn):
    """Locate the retail nonmatchings .s for a function under asm_root. Returns a
    path or None. No unit/path is baked in; the layout is asm/**/nonmatchings/**/."""
    want = fn + ".s"
    for dirpath, _dirs, files in os.walk(asm_root):
        if "nonmatchings" not in dirpath.replace("\\", "/").split("/"):
            continue
        if want in files:
            return os.path.join(dirpath, want)
    return None


def target_words(asm_root, fn):
    """splat target .s -> [(be_hex, disasm)] (little-endian column -> BE)."""
    p = find_target_s(asm_root, fn)
    if not p:
        return None
    s = open(p, encoding="utf-8", errors="replace").read()
    words = []
    pat = re.compile(r'/\*\s*[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+([0-9A-Fa-f]{8})\s+\*/\s+(\S+)(.*)')
    started = False
    for line in s.splitlines():
        if re.match(r'^\s*glabel\s+%s\b' % re.escape(fn), line):
            started = True
            continue
        if started and re.match(r'^\s*endlabel\b', line):
            break
        if not started:
            continue
        m = pat.search(line)
        if m:
            le = m.group(1)
            be = (le[6:8] + le[4:6] + le[2:4] + le[0:2]).lower()
            words.append((be, (m.group(2) + m.group(3)).strip()))
    return words


# --------------------------------------------------------------------------
# per-function span splicing
# --------------------------------------------------------------------------
ENT_RE = re.compile(r'(?m)^[ \t]*\.ent[ \t]+(\S+)')


def _read_bytes_str(path):
    with open(path, "rb") as f:
        return f.read().decode("latin-1")


def _write_bytes_str(path, s):
    with open(path, "wb") as f:
        f.write(s.encode("latin-1"))


def split_spans(text):
    """[(name, start, end)] for each `.ent NAME ... .end NAME` code span."""
    spans = []
    for m in ENT_RE.finditer(text):
        name = m.group(1)
        endm = re.search(r'(?m)^[ \t]*\.end[ \t]+%s\b' % re.escape(name),
                         text[m.start():])
        if not endm:
            continue
        end = m.start() + endm.end()
        nl = text.find("\n", end)
        end = len(text) if nl < 0 else nl + 1
        spans.append((name, m.start(), end))
    return spans


def normalize_span_text(span, tgt, sigma, passes):
    """Replay the manifest's ordered passes on one function's span. reg_realloc
    applies sigma (derived from the words); the .s-text operators run after."""
    txt = span
    for name in passes:
        if name == "reg_realloc":
            txt = apply_sigma_to_s(txt, sigma)
        else:
            fn = PASSES.get(name)
            if fn is None:
                raise RuntimeError("unknown normalizer pass: %s" % name)
            txt = fn(txt, tgt)
    return txt


def load_manifest(path=MANIFEST):
    """{func: {"passes": [name, ...]}} -- absent/empty file = {}."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data or {}


def normalize_s(s_file, ctx, manifest=None):
    """Splice-normalize the manifest functions in a cc1 .s in place. Reads once,
    rewrites once. Returns the sorted list of function names it rewrote.

    ctx keys: python, maspsx_py, maspsx_flags, as_bin, maspsx_as_flags, objdump,
    run (callable -> (rc, out, err)), asm_root."""
    if manifest is None:
        manifest = load_manifest()
    text = _read_bytes_str(s_file)
    spans = split_spans(text)
    out = text
    rewrote = []
    # Pass 1 -- text passes (reg_realloc / commutative_swap / laform). Splice from
    # the tail so earlier (start,end) offsets stay valid; words are always assembled
    # from the ON-DISK cc1 .s (unmodified until the pass-1 write below).
    for name, start, end in sorted(spans, key=lambda x: -x[1]):
        if name not in manifest:
            continue
        passes = manifest[name].get("passes", [])
        text_passes = [p for p in passes if p not in WORD_PASSES]
        if not text_passes:
            if any(p in WORD_PASSES for p in passes):
                rewrote.append(name)
            continue
        tgt = target_words(ctx["asm_root"], name)
        if tgt is None:
            raise RuntimeError("no target .s found for %s under %s"
                               % (name, ctx["asm_root"]))
        tgt = _strip_trailing_pad(tgt)

        # Split the recipe around sigma: count-changing PRE_SIGMA passes (e.g.
        # un_hi_cse) run first so sigma is derived from count-aligned assembly.
        pre = [p for p in text_passes if p in PRE_SIGMA_PASSES]
        rest = [p for p in text_passes if p not in PRE_SIGMA_PASSES]
        span = out[start:end]
        for p in pre:
            span = PASSES[p](span, tgt)
        if pre:                             # re-assemble to align sigma to remat form
            out_pre = out[:start] + span + out[end:]
            _write_bytes_str(s_file, out_pre)
        our, err = assemble_words(ctx, s_file, name)
        if err:
            raise RuntimeError("assemble %s:\n%s" % (name, err))
        our = _strip_trailing_pad(our)
        sigma, _conf = derive_sigma(our, tgt)
        norm = normalize_span_text(span, tgt, sigma, rest)
        out = out[:start] + norm + out[end:]
        if pre:                             # keep disk in sync for the next span
            _write_bytes_str(s_file, out)
        if name not in rewrote:
            rewrote.append(name)

    # Pass 2 -- word passes. These need the ASSEMBLED words of the text-normalized
    # code, so write pass 1 to disk first and assemble each span in full-file
    # context. Reordering ops (delay_fill / epilogue_unfill) run before padnop,
    # because they change the word count and padnop must re-measure after them.
    word_funcs = [n for n in manifest
                  if any(p in WORD_PASSES for p in manifest[n].get("passes", []))]
    if word_funcs:
        # 2a -- reordering passes (source rewrite + `.set noreorder`)
        _write_bytes_str(s_file, out)
        for name, start, end in sorted(split_spans(out), key=lambda x: -x[1]):
            if name not in word_funcs:
                continue
            reorder = [p for p in manifest[name]["passes"]
                       if p in ("delay_fill", "epilogue_unfill", "reorder_indep",
                                "prologue_save_hoist")]
            if not reorder:
                continue
            our, err = assemble_words(ctx, s_file, name)
            if err:
                raise RuntimeError("assemble %s (reorder):\n%s" % (name, err))
            tgt = target_words(ctx["asm_root"], name)
            if tgt is None:
                raise RuntimeError("no target .s for %s under %s"
                                   % (name, ctx["asm_root"]))
            tgt = _strip_trailing_pad(tgt)
            new_span = emit_noreorder_span(out[start:end], tgt, our, reorder)
            out = out[:start] + new_span + out[end:]

        # 2b -- padnop (pure text append; re-measure against post-reorder words)
        _write_bytes_str(s_file, out)
        for name, start, end in sorted(split_spans(out), key=lambda x: -x[1]):
            if name not in word_funcs:
                continue
            if "padnop" not in manifest[name]["passes"]:
                continue
            our, err = assemble_words(ctx, s_file, name)
            if err:
                raise RuntimeError("assemble %s (padnop):\n%s" % (name, err))
            tgt_full = target_words_full(ctx["asm_root"], name)
            if tgt_full is None:
                raise RuntimeError("no target .s for %s under %s"
                                   % (name, ctx["asm_root"]))
            our_hex = [w for w, _ in our]
            new_span, _fired = apply_padnop(out[start:end], our_hex, tgt_full)
            out = out[:start] + new_span + out[end:]

    _write_bytes_str(s_file, out)
    return sorted(rewrote)
