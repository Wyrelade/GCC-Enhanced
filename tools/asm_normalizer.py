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
# store-side un_hi_cse: the get/set-through-assembler-temp leaf
# `T f(T a){ T old = G; G = a; return old; }`. Our cc1 CSEs ONE `lui $B,%hi(G)`
# and shares $B as the base of both the load (`lw $D,%lo(G)($B)`, dest != base)
# and the store (`sw $src,%lo(G)($B)`). The retail codegen instead (a)
# materializes the load base==dest (`lui $D,%hi(G); lw $D,%lo(G)($D)`) and (b)
# rematerializes a SEPARATE `lui $at,%hi(G)` for the store. Plain un_hi_cse
# handles only the load and would leave the store's base ($B) dangling, so this
# pass does both together. Count-changing (one net extra `lui`) -> PRE_SIGMA.
# --------------------------------------------------------------------------
_SW_LO = re.compile(r"^(\s*)(s[bhw])\s+(\$\d+)\s*,\s*%lo\(([\w.$]+)\)\((\$\d+)\)")


def target_store_remat_syms(tgt):
    """Symbols the target stores through a rematerialized `$at` base: it carries
    a base==dest `lui;lw` for S AND a `s? R,%lo(S)($at)` store of S."""
    loads = target_remat_syms(tgt)
    if not loads:
        return set()
    syms = set()
    sw_re = re.compile(r"s[bhw]\s+\$\w+\s*,\s*%lo\(([\w.$]+)\)\(\$at\)")
    for _w, dis in tgt:
        m = sw_re.search(dis)
        if m and m.group(1) in loads:
            syms.add(m.group(1))
    return syms


def un_hi_cse_store_s(stext, remat_syms):
    """Split a shared %hi base that feeds both a load and a store of the same
    symbol S in `remat_syms`: fold the load to base==dest and give the store its
    own `lui $at`. Fires only when our source has that exact shared shape. The
    fresh `lui $at` is hoisted above a leading return, since cc1 often schedules
    the store into the return's delay slot."""
    if not remat_syms:
        return stext
    lines = stext.split("\n")
    lui_idx = {}
    for idx, ln in enumerate(lines):
        m = _LUI_HI.match(ln)
        if m and m.group(3) in remat_syms:
            lui_idx.setdefault(m.group(3), (idx, m.group(2)))
    out = list(lines)
    changed = False
    for sym, (li, base) in lui_idx.items():
        lw_i = sw_i = None
        lw_dst = sw_indent = sw_src = None
        for idx, ln in enumerate(lines):
            m = _LW_LO.match(ln)
            if m and m.group(3) == sym and m.group(4) == base and m.group(2) != base:
                lw_i, lw_dst = idx, m.group(2)
            m = _SW_LO.match(ln)
            if m and m.group(4) == sym and m.group(5) == base:
                sw_i, sw_indent, sw_src = idx, m.group(1) or "\t", m.group(3)
        if lw_i is None or sw_i is None:
            continue
        indent = (_LW_LO.match(lines[lw_i]).group(1) or "\t")
        out[li] = None
        out[lw_i] = ("%slui\t%s,%%hi(%s)\n%slw\t%s,%%lo(%s)(%s)"
                     % (indent, lw_dst, sym, indent, lw_dst, sym, lw_dst))
        out[sw_i] = "%ssw\t%s,%%lo(%s)($1)" % (sw_indent, sw_src, sym)
        ins_at = sw_i
        j = sw_i - 1
        while j >= 0 and (not _is_insn_line(lines[j])):
            j -= 1
        if j >= 0 and _RET_S.match(lines[j]):
            ins_at = j
        out[ins_at] = "%slui\t$1,%%hi(%s)\n%s" % (sw_indent, sym, out[ins_at])
        changed = True
    if not changed:
        return stext
    return "\n".join(l for l in out if l is not None)


def un_hi_cse_store_pass(stext, tgt):
    return un_hi_cse_store_s(stext, target_store_remat_syms(tgt))


# --------------------------------------------------------------------------
# base-CSE-collapse: our cc1 accesses a global through the split form
# `lui $B,%hi(S); ... sw $x,%lo(S)($B)` and THEN materializes the full address
# `addiu $B,$B,%lo(S)` for a later pointer use (e.g. passing &S to a call), so the
# early accesses use the hi-only base and the later ones the full base. Retail
# materializes the full base ONCE, right after the lui, and every access uses a
# plain `k($B)` offset. laform is the wrong tool here (it would insert a SECOND
# `addiu %lo` because the full base already exists). This pass moves the existing
# `addiu $B,$B,%lo(S)` up to just after its lui and rewrites each intervening
# `%lo(S)($B)` / `%lo(S+k)($B)` operand to `0($B)` / `k($B)`. Semantics-preserving
# when, between the lui and the addiu, $B is never redefined and is only read as
# the base of those %lo(S) memory operands (so after the move every such access
# still computes %hi(S)+%lo(S)+k), and the region is straight-line (no label, no
# branch, no noreorder block). Count-preserving but order-changing, so it runs
# PRE_SIGMA (sigma is then derived from the collapsed, position-aligned words).
# Target-guided: only symbols the target itself materializes as an adjacent
# `lui R,%hi(S); addiu R,R,%lo(S)` pair.
# --------------------------------------------------------------------------
def target_collapse_syms(tgt):
    """Symbols the target materializes as an adjacent `lui R,%hi(S)` +
    `addiu R,R,%lo(S)` full base."""
    syms = set()
    lui_re = re.compile(r"lui\s+(\$?\w+)\s*,\s*%hi\(([\w.$]+)\)")
    add_re = re.compile(r"addiu\s+(\$?\w+)\s*,\s*(\$?\w+)\s*,\s*%lo\(([\w.$]+)\)")
    for i in range(len(tgt) - 1):
        a = lui_re.search(tgt[i][1])
        b = add_re.search(tgt[i + 1][1])
        if not a or not b:
            continue
        r = norm_reg(a.group(1))
        if (a.group(2) == b.group(3) and norm_reg(b.group(1)) == r
                and norm_reg(b.group(2)) == r):
            syms.add(a.group(2))
    return syms


def _label_referenced(lines, name):
    """Some instruction line references label `name` (a real branch target), as
    opposed to cc1's `LMn:` line-marker labels, which nothing branches to."""
    pat = re.compile(r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])")
    return any(_is_insn_line(l) and pat.search(l.split("#", 1)[0]) for l in lines)


def base_cse_collapse_s(stext, syms):
    """Fold split `%lo(S)($B)` accesses onto the full base `addiu $B,$B,%lo(S)` that
    follows them, moving that addiu up to just after the lui. See block comment."""
    if not syms:
        return stext
    lines = stext.split("\n")
    changed = False
    for sym in syms:
        i = 0
        while i < len(lines):
            m = _LUI_HI.match(lines[i])
            if not m or m.group(3) != sym:
                i += 1
                continue
            indent, base = m.group(1) or "\t", m.group(2)
            reg_re = re.compile(r"(?<![\w$])" + re.escape(base) + r"(?![\w])")
            lo_re = re.compile(r"%lo\(" + re.escape(sym) + r"(?:\+(\d+))?\)\("
                               + re.escape(base) + r"\)")
            add_re = re.compile(r"^\s*addiu\s+" + re.escape(base) + r"\s*,\s*"
                                + re.escape(base) + r"\s*,\s*%lo\(" + re.escape(sym)
                                + r"\)\s*(#.*)?$")
            hits, add_i, j = [], None, i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith(".set") and "noreorder" in s:
                    break
                if s.endswith(":") and not s.startswith(".") and \
                        _label_referenced(lines, s[:-1]):
                    break                       # branch target -> not straight-line
                if not _is_insn_line(lines[j]):
                    j += 1
                    continue
                if add_re.match(lines[j]):
                    add_i = j
                    break
                body = lines[j].split("#", 1)[0]
                mn = _line_mnem(lines[j])
                if _COND_S.match(mn) or mn in ("j", "jr", "jal", "jalr", "b", "bal"):
                    break
                if re.match(r"\s*[a-z][\w.]*\s+" + re.escape(base) + r"\s*,", body) \
                        and mn not in _STORE_MN:
                    break                       # $B redefined
                if reg_re.search(body):
                    if len(lo_re.findall(body)) != 1 or \
                            reg_re.search(lo_re.sub("", body)):
                        break                   # $B read other than as %lo(S) base
                    hits.append(j)
                j += 1
            if add_i is None or not hits:
                i += 1
                continue
            for h in hits:
                lines[h] = lo_re.sub(lambda mm: "%d(%s)" % (int(mm.group(1) or 0), base),
                                     lines[h])
            add_line = lines.pop(add_i)
            lines.insert(i + 1, add_line)
            changed = True
            i = add_i + 1
    return "\n".join(lines) if changed else stext


def base_cse_collapse_pass(stext, tgt):
    return base_cse_collapse_s(stext, target_collapse_syms(tgt))


# --------------------------------------------------------------------------
# shift-const-fold: our cc1's CSE sees a constant K already live in a register
# (`li $r,K`, e.g. a value about to be stored) and substitutes that register for
# the constant shift count, emitting `sll $d,$s,$r` (gas: sllv) where retail keeps
# the immediate form `sll $d,$s,K`. Rewrite the register-count shift back to the
# immediate form when the count register's straight-line reaching definition in
# the span is `li $r,K` (or `addiu/ori $r,$0,K`) with 0 <= K <= 31. Semantics-
# preserving: a variable shift uses the low 5 bits of the count register, which
# equal K. Target-guided: only for (op, K) pairs the target itself emits as an
# immediate shift. Count-preserving; runs PRE_SIGMA so sigma is not derived from a
# sllv/sll opcode-field mismatch.
# --------------------------------------------------------------------------
_SHIFT_REG_S = re.compile(r"^(\s*)(sll|srl|sra)(v?)\s+(\$\w+)\s*,\s*(\$\w+)\s*,\s*(\$\w+)\s*(#.*)?$")
_CONST_DEF_S = re.compile(
    r"^\s*(?:li\s+(\$\w+)\s*,\s*(-?(?:0x[0-9a-fA-F]+|\d+))"
    r"|(?:addiu|ori)\s+(\$\w+)\s*,\s*\$(?:0|zero)\s*,\s*(-?(?:0x[0-9a-fA-F]+|\d+)))\s*(#.*)?$")


def target_imm_shifts(tgt):
    """{(op, K)} immediate shifts the target emits, e.g. ("sll", 16)."""
    out = set()
    pat = re.compile(r"^(sll|srl|sra)\s+\S+\s*,\s*\S+\s*,\s*(0x[0-9a-fA-F]+|\d+)\s*$")
    for _w, dis in tgt:
        m = pat.match(dis.strip())
        if m:
            out.add((m.group(1), int(m.group(2), 0)))
    return out


def shift_const_fold_s(stext, allowed):
    if not allowed:
        return stext
    lines = stext.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    changed = False
    for p, (li, l) in enumerate(ins):
        m = _SHIFT_REG_S.match(l)
        if not m:
            continue
        cnt = _src_reg(m.group(6))
        if not cnt:
            continue
        k = None
        for q in range(p - 1, -1, -1):
            qi, ql = ins[q]
            if q == p - 1 and _src_is_branch(ql) and not _src_is_ret(ql):
                continue            # we sit in its delay slot: it has not transferred yet
            if _src_is_branch(ql) or any(_branch_target_label(lines, ins, x)
                                        for x in lines[qi:li]):
                break
            d, _u = defs_uses(ql.split("#", 1)[0].strip())
            if cnt in d:
                c = _CONST_DEF_S.match(ql)
                if c:
                    reg = c.group(1) or c.group(3)
                    if _src_reg(reg) == cnt:
                        k = int(c.group(2) or c.group(4), 0)
                break
        if k is None or not (0 <= k <= 31) or (m.group(2), k) not in allowed:
            continue
        lines[li] = "%s%s\t%s,%s,%d" % (m.group(1), m.group(2), m.group(4), m.group(5), k)
        changed = True
    return "\n".join(lines) if changed else stext


def shift_const_fold_pass(stext, tgt):
    return shift_const_fold_s(stext, target_imm_shifts(tgt))


# --------------------------------------------------------------------------
# zero-remat: our cc1 copies a register it knows holds zero (`move $d,$s` right
# after `move $s,$0`, e.g. two zero call arguments) where retail materializes the
# zero again (`addu $d,$zero,$zero`). Rewrite the copy to `move $d,$0` when the
# source register's straight-line reaching definition in the span is a zero
# materialization. Semantics-preserving ($s is 0 there). Target-guided: rewrites
# at most as many copies as the target has more zero materializations than ours,
# in source order. Count-preserving; runs PRE_SIGMA with shift_const_fold.
# --------------------------------------------------------------------------
_MOVE_S = re.compile(r"^(\s*)move\s+(\$\w+)\s*,\s*(\$\w+)\s*(#.*)?$")
_ZERO_DEF_S = re.compile(
    r"^\s*(?:move\s+(\$\w+)\s*,\s*\$(?:0|zero)"
    r"|li\s+(\$\w+)\s*,\s*0"
    r"|(?:addu|or)\s+(\$\w+)\s*,\s*\$(?:0|zero)\s*,\s*\$(?:0|zero)"
    r"|(?:addiu|ori)\s+(\$\w+)\s*,\s*\$(?:0|zero)\s*,\s*0)\s*(#.*)?$")
_ZERO_TGT = re.compile(r"^(?:addu|or)\s+\S+\s*,\s*\$?zero\s*,\s*\$?zero\s*$"
                       r"|^(?:addiu|ori)\s+\S+\s*,\s*\$?zero\s*,\s*0x0*\s*$"
                       r"|^(?:move\s+\S+\s*,\s*\$?zero|li\s+\S+\s*,\s*0)\s*$")


def zero_remat_s(stext, budget):
    if budget <= 0:
        return stext
    lines = stext.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    changed = False
    for p, (li, l) in enumerate(ins):
        if budget <= 0:
            break
        m = _MOVE_S.match(l)
        if not m:
            continue
        src = _src_reg(m.group(3))
        if not src or src == _src_reg("$0"):
            continue
        zero = False
        for q in range(p - 1, -1, -1):
            qi, ql = ins[q]
            if q == p - 1 and _src_is_branch(ql):
                continue            # we sit in its delay slot: it has not transferred yet
            if _src_is_branch(ql) or any(_branch_target_label(lines, ins, x)
                                        for x in lines[qi:li]):
                break
            d, _u = defs_uses(ql.split("#", 1)[0].strip())
            if src in d:
                z = _ZERO_DEF_S.match(ql)
                zero = bool(z) and _src_reg(next(g for g in z.groups()[:4] if g)) == src
                break
        if not zero:
            continue
        lines[li] = "%smove\t%s,$0" % (m.group(1), m.group(2))
        budget -= 1
        changed = True
    return "\n".join(lines) if changed else stext


def _tail_count(dis_list, src=False):
    """Words after the `lw $ra` restore that are not frame loads, stack adjusts,
    returns or nops (the work the epilogue still does after reloading $ra).
    src=True: the list is cc1 source insns, weighted by their assembled length."""
    n = 0
    for d in dis_list:
        t = d.split("#", 1)[0].strip() if src else d.strip()
        p = t.split(None, 1)
        if not p:
            continue
        mn = p[0].lower()
        if mn in ("jr", "j") or is_nop(t):
            break
        if _frame_load(t) or _src_sp_delta(t) is not None:
            continue
        n += _src_nwords(t) if src else 1
    return n


# --------------------------------------------------------------------------
# ra-restore-sink: our cc1 (notably in no-split units, where a global store or an
# `la` is one macro insn to the scheduler) reloads `$ra` early in the epilogue,
# ahead of trailing global stores or the return-value setup; retail reloads it
# after them. Sink our `lw $31,K($sp)` past as many trailing words as the target
# still executes after its own `lw $ra`. Semantics-preserving: every insn passed is
# checked not to read or write $31/$sp and not to store into the frame word K;
# control flow and labels stop the move. Count-preserving text pass.
# --------------------------------------------------------------------------
_CALLEE_RS = {"s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "fp", "ra"}


def _callee_restore(body):
    """Dest reg of a callee-saved restore `lw $sN/$fp/$ra,K($sp)`, else None."""
    if not _frame_load(body):
        return None
    d, _u = defs_uses(body)
    d = set(d) & _CALLEE_RS
    return next(iter(d)) if len(d) == 1 else None


def _unit_permute(lines, blk, order):
    """Rewrite the line range covered by units `blk` [(a, z, body), ...] so unit
    order[k] occupies slot k; non-unit lines between units stay in place."""
    texts = [lines[a:z + 1] for a, z, _b in blk]
    seg = []
    for k, (a, z, _b) in enumerate(blk):
        seg.extend(texts[order[k]])
        nxt = blk[k + 1][0] if k + 1 < len(blk) else z + 1
        seg.extend(lines[z + 1:nxt])
    lines[blk[0][0]:blk[-1][1] + 1] = seg


def _unit_words(body):
    return 2 if re.search(r"%lo\(", body) else _src_nwords(body)


def callee_restore_sink_s(stext, tgt):
    """Second phase of ra_restore_sink: the whole callee-saved restore group.
    Retail can keep all `lw $sN/$ra` restores after trailing work (global stores,
    return setup) where our cc1 interleaves them. Hoist our earliest trailing work
    units above the first restore until the work left after it equals the
    target's. Each hoisted unit must not read/write a register restored by a load
    it passes, touch $sp, or store into the frame; labels/branches stop the move.
    Units are single insns or `.set noat` expanded symbolic accesses (moved whole)."""
    tl = [d for _w, d in tgt]
    rets = [i for i, d in enumerate(tl) if re.match(r"\s*jr\s+\$?ra\s*$", d.strip())]
    if not rets:
        return stext
    j = rets[-1]
    first = None
    for i in range(j - 1, -1, -1):
        mn = tl[i].split(None, 1)[0].lower() if tl[i].strip() else ""
        if mn.startswith("b") or mn in ("j", "jal", "jr", "jalr"):
            break
        if _callee_restore(tl[i].strip()):
            first = i
    if first is None:
        return stext
    want = _tail_count(tl[first + 1:])
    lines = stext.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    blocks = _sm_units(lines, ins)
    orets = [i for i, l in ins if _src_is_ret(l)]
    if not orets or not blocks:
        return stext
    # the block that ends right before the return (the `.set noreorder` return
    # group starts the next block)
    before = [b for b in blocks if b[-1][1] < orets[-1]]
    if not before:
        return stext
    blk = before[-1]
    if any(i > blk[-1][1] and i < orets[-1] and not lines[i].strip().startswith(".set")
           for i, _l in ins):
        return stext
    bodies = [_sm_dep_body(b) for _a, _z, b in blk]
    rest = [k for k, b in enumerate(bodies) if _callee_restore(b)]
    if not rest:
        return stext
    k0 = rest[0]
    have = sum(_unit_words(b) for b in bodies[k0 + 1:]
               if not (_frame_load(b) or _src_sp_delta(b) is not None))
    need = have - want
    if need < 0:
        return stext
    hoist, passed = [], set()
    for k in range(k0, len(blk)):
        if need <= 0:
            break
        b = bodies[k]
        r = _callee_restore(b)
        if r:
            passed.add(r)
            continue
        if _src_sp_delta(b) is not None or _frame_load(b):
            return stext
        d, u = defs_uses(b)
        if (set(d) | set(u)) & (passed | {"sp"}):
            return stext
        hoist.append(k)
        need -= _unit_words(b)
    if need != 0:
        return stext
    if hoist:
        order = list(range(k0)) + hoist + [k for k in range(k0, len(blk)) if k not in hoist]
        _unit_permute(lines, blk, order)
    return _restore_group_order("\n".join(lines), tl, first, j)


def _restore_group_order(stext, tl, first, j):
    """When our epilogue ends in a contiguous run of callee-saved restores holding
    the same registers as the target's trailing run, emit them in the target's
    order (loads from distinct frame words into distinct regs commute)."""
    t_regs = [_callee_restore(tl[i].strip()) for i in range(first, j)]
    if not t_regs or None in t_regs:
        return stext
    lines = stext.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    orets = [p for p, (_i, l) in enumerate(ins) if _src_is_ret(l)]
    if not orets:
        return stext
    pr = orets[-1]
    grp = []
    for p in range(pr - 1, -1, -1):
        r = _callee_restore(ins[p][1].split("#", 1)[0].strip())
        if not r:
            break
        grp.insert(0, (p, r))
    if len(grp) != len(t_regs) or sorted(r for _p, r in grp) != sorted(t_regs):
        return stext
    if any(_branch_target_label(lines, ins, x)
           for x in lines[ins[grp[0][0]][0] + 1:ins[grp[-1][0]][0]]):
        return stext
    by = {r: lines[ins[p][0]] for p, r in grp}
    for (p, _r), tr in zip(grp, t_regs):
        lines[ins[p][0]] = by[tr]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sched-match: retail sometimes keeps a straight-line block in source order where
# our sched2 hoists some insns (e.g. global stores of s-regs moved up so the
# restores can issue early). Match every insn of a straight-line source block to a
# target insn by a canonical key (mnemonic, regs, immediate / symbol; symbolic
# macros keyed by their symbol so `lui $at` halves are ignored), then emit the
# block in target order when every transposed pair is independent: no register
# RAW/WAR/WAW, and memory ops only transpose when neither stores or they touch
# provably different places (distinct symbols, or $sp frame vs a symbol, or
# disjoint $sp words). Blocks end at labels that are branch targets, branches,
# calls and `.set noreorder` regions. Count-preserving text pass.
# --------------------------------------------------------------------------
_SYM_RE = re.compile(r"%(?:hi|lo)\(([^)]+)\)")


def _sm_int(t):
    """Integer operand; also splat's `(K >> 16)` / `(K & 0xFFFF)` hi/lo forms."""
    m = re.fullmatch(r"\((-?0x[0-9A-Fa-f]+|-?\d+)\s*(>>\s*16|&\s*0xFFFF)\)", t.strip())
    if m:
        v = int(m.group(1), 0)
        return (v >> 16) & 0xFFFF if ">>" in m.group(2) else v & 0xFFFF
    try:
        return int(t, 0)
    except ValueError:
        return None


def _sm_key(body, target):
    p = body.split(None, 1)
    if not p:
        return None
    mn = p[0].lower()
    ops = [o.strip() for o in split_ops(p[1])] if len(p) > 1 else []
    if target:
        syms = _SYM_RE.findall(body)
        if syms:
            if mn == "lui":
                return "skip"
            reg = norm_reg(ops[0]) if ops else None
            if mn in ("addiu", "ori"):
                return ("la", reg, syms[0])
            return (mn, reg, syms[0])
    else:
        mo = _mem_operand(body)
        if mn in _STORE_MN or mn in ("lw", "lh", "lhu", "lb", "lbu"):
            if len(ops) == 2 and not mo and "(" not in ops[1]:
                return (mn, norm_reg(ops[0]), ops[1])
        if mn == "la" and len(ops) == 2:
            return ("la", norm_reg(ops[0]), ops[1])
        if mn == "move" and len(ops) == 2:
            mn, ops = "addu", [ops[0], ops[1], "$0"]
        elif mn == "li" and len(ops) == 2:
            v = _sm_int(ops[1])
            if v is None or not -0x8000 <= v < 0x8000:
                return None
            mn, ops = "addiu", [ops[0], "$0", ops[1]]
        elif mn == "subu" and len(ops) == 3 and _sm_int(ops[2]) is not None:
            mn, ops = "addiu", [ops[0], ops[1], str(-_sm_int(ops[2]))]
        elif mn == "addu" and len(ops) == 3 and _sm_int(ops[2]) is not None:
            mn = "addiu"
    key = [mn]
    for o in ops:
        m = re.fullmatch(r"(-?(?:0x[0-9a-fA-F]+|\d+))\((\$?\w+)\)", o)
        if m:
            key += [int(m.group(1), 0), norm_reg(m.group(2))]
        elif _sm_int(o) is not None:
            key.append(_sm_int(o))
        elif o.startswith("$"):
            r = norm_reg(o)
            key.append("zero" if r in ("zero", "0") else r)
        else:
            return None
    return tuple(key)


def _sm_mem(body):
    """None (no memory), or (is_store, where) with where = ('sym', name) /
    ('sp', off, size) / ('any',)."""
    p = body.split(None, 1)
    mn = p[0].lower() if p else ""
    if not (mn in _STORE_MN or mn in ("lw", "lh", "lhu", "lb", "lbu", "lwl", "lwr")):
        return None
    st = mn in _STORE_MN
    mo = _mem_operand(body)
    if mo:
        return (st, ("sp", mo[1], mo[2]) if mo[0] == "sp" else ("any",))
    ops = split_ops(p[1]) if len(p) > 1 else []
    if len(ops) == 2 and "(" not in ops[1]:
        return (st, ("sym", re.split(r"[+-]", ops[1].strip())[0]))
    return (st, ("any",))


def _sm_indep(a, b, stable=frozenset()):
    """`stable`: registers no insn of the block redefines, so two accesses based
    on one of them at disjoint offsets cannot alias."""
    da, ua = defs_uses(a)
    db, ub = defs_uses(b)
    da, ua, db, ub = set(da), set(ua), set(db), set(ub)
    if da & (db | ub) or ua & db:
        return False
    ma, mb = _sm_mem(a), _sm_mem(b)
    if ma and mb and (ma[0] or mb[0]):
        wa, wb = ma[1], mb[1]
        oa, ob = _mem_operand(a), _mem_operand(b)
        if (wa[0] == "any" and wb[0] == "any" and oa and ob and oa[0] == ob[0]
                and oa[0] in stable):
            return oa[1] + oa[2] <= ob[1] or ob[1] + ob[2] <= oa[1]
        if "any" in (wa[0], wb[0]):
            return False
        if wa[0] == "sym" and wb[0] == "sym":
            return wa[1] != wb[1]
        if wa[0] == "sp" and wb[0] == "sp":
            return wa[1] + wa[2] <= wb[1] or wb[1] + wb[2] <= wa[1]
        return True                         # frame vs global symbol
    return True


def _sm_units(lines, ins):
    """Straight-line blocks of movable units. A unit is one insn line, or an
    explicitly expanded symbolic access `.set noat; lui $1,%hi(S); op ..%lo(S)($1);
    .set at` (keyed and dependency-checked by its %lo insn). Returns
    [[(first_line, last_line, key_body), ...], ...]."""
    blocks, cur, noreo, prev_br = [], [], False, False
    ins_at = {i for i, _l in ins}

    def flush():
        if len(cur) > 1:
            blocks.append(list(cur))
        del cur[:]
    i = 0
    while i < len(lines):
        l = lines[i]
        s = l.strip()
        if s.startswith(".set") and s.split()[-1] == "noat" and not noreo:
            j = i + 1
            grp = []
            while j < len(lines) and not (lines[j].strip().startswith(".set")
                                          and lines[j].strip().split()[-1] == "at"):
                if j in ins_at:
                    grp.append(j)
                j += 1
            if (j < len(lines) and len(grp) == 2 and not prev_br
                    and re.match(r"\s*lui\s+\$(?:1|at)\s*,\s*%hi\(", lines[grp[0]])):
                cur.append((i, j, lines[grp[1]].split("#", 1)[0].strip()))
                i = j + 1
                continue
            flush()
            i = j + 1
            prev_br = False
            continue
        if s.startswith(".set") and "noreorder" in s:
            noreo = True
        elif s.startswith(".set") and s.split()[-1] == "reorder":
            noreo = False
        if i in ins_at:
            br = bool(_src_is_branch(l) or _src_is_ret(l) or re.match(r"\s*jalr?\b", l))
            if noreo or br or prev_br:
                flush()
            else:
                cur.append((i, i, l.split("#", 1)[0].strip()))
            prev_br = br
        elif _branch_target_label(lines, ins, l):
            flush()
        i += 1
    flush()
    return blocks


def _sm_srckey(body):
    m = re.match(r"(\w+)\s+(\$\w+)\s*,\s*%lo\(([^)]+)\)\(\$(?:1|at)\)\s*$", body)
    if m:
        return (m.group(1).lower(), norm_reg(m.group(2)), m.group(3))
    return _sm_key(body, False)


def _sm_dep_body(body):
    """Body for dependency checks: an expanded `op $r,%lo(S)($1)` acts like the
    macro `op $r,S` (the $at temp is private to its unit)."""
    m = re.match(r"(\w+)\s+(\$\w+)\s*,\s*%lo\(([^)]+)\)\(\$(?:1|at)\)\s*$", body)
    return "%s\t%s,%s" % m.groups() if m else body


def sched_match_pass(stext, tgt):
    tkeys = [_sm_key(d.strip(), True) for _w, d in tgt]
    lines = stext.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    used = set()
    edits = []
    for blk in _sm_units(lines, ins):
        bodies = [_sm_dep_body(b) for _a, _z, b in blk]
        pos = []
        for (_a, _z, b) in blk:
            k = _sm_srckey(b)
            t = next((q for q, tk in enumerate(tkeys)
                      if k is not None and tk == k and q not in used), None)
            if t is None:
                pos = None
                break
            pos.append(t)
            used.add(t)
        if pos is None:
            continue
        order = sorted(range(len(blk)), key=lambda x: pos[x])
        if order == list(range(len(blk))):
            continue
        dset = set()
        for b in bodies:
            dset |= set(defs_uses(b)[0])
        stable = frozenset(r for b in bodies for r in defs_uses(b)[1]) - dset
        if not all(_sm_indep(bodies[x], bodies[y], stable)
                   for a_, x in enumerate(order) for y in order[a_ + 1:] if y < x):
            continue
        edits.append((blk, order))
    if not edits:
        return stext
    for blk, order in sorted(edits, key=lambda e: -e[0][0][0]):
        _unit_permute(lines, blk, order)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# fallthrough-fill: in units assembled by aspsx with cc1 delay filling off
# (`nosplit_nodb` / `nodb`), a conditional branch's delay slot is filled with the
# FIRST instruction of the fall-through path when that instruction is harmless on
# the taken path (a pure ALU/lui write of a register the taken path redefines
# before reading). cc1 leaves those branches unfilled (the assembler pads a nop).
# Target-guided: the k-th conditional branch of ours is paired with the k-th of
# the target; fire only when the target's slot holds our fall-through insn.
# Semantics: the moved insn now also runs on the taken path, so its destination
# must be dead there (straight-line scan from the target label: written before
# read; a return counts as a read of $v0 and of callee-saved registers; $v1 is
# treated as dead there: no 64-bit returns in this code).
# Count-preserving text pass (the assembler's nop is replaced by the moved word).
# --------------------------------------------------------------------------
_FF_ALU = {"addu", "addiu", "subu", "and", "andi", "or", "ori", "xor", "xori",
           "nor", "sll", "srl", "sra", "sllv", "srlv", "srav", "slt", "slti",
           "sltu", "sltiu", "lui", "li", "move"}
_FF_RET_LIVE = {"v0", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "fp",
                "sp", "ra", "gp"}


def _ff_word_form(body):
    """The single-word spelling of a 1-word source insn (`li $r,K` with K = hi<<16
    becomes `lui $r,hi`), or None when it is not a 1-word pure-ALU insn."""
    p = body.split(None, 1)
    if not p or p[0].lower() not in _FF_ALU:
        return None
    if p[0].lower() != "li" and _src_nwords(body) != 1:
        return None
    if p[0].lower() == "li":
        ops = split_ops(p[1])
        v = _sm_int(ops[1].strip()) if len(ops) == 2 else None
        if v is None:
            return None
        if -0x8000 <= v < 0x8000:
            return "addiu\t%s,$0,%d" % (ops[0].strip(), v)
        if v & 0xFFFF:
            return None
        return "lui\t%s,0x%x" % (ops[0].strip(), (v >> 16) & 0xFFFF)
    return body


def _ff_dead_on(lines, ins, label, reg, seen=None):
    """`reg` is dead at `label`: on every path from it (both sides of conditional
    branches, `j` followed, delay slots included) it is written before it is read.
    A return reads $v0 and the callee-saved registers; calls and indirect
    jumps give up (conservative)."""
    seen = set() if seen is None else seen
    if label in seen:
        return True
    seen.add(label)
    at = next((k for k, l in enumerate(lines) if l.strip() == label + ":"), None)
    if at is None:
        return False
    rest = [(i, l) for i, l in ins if i > at]
    nr, on = set(), False
    for x, xl in enumerate(lines):
        s = xl.strip()
        if s.startswith(".set") and "noreorder" in s:
            on = True
        elif s.startswith(".set") and s.split()[-1] == "reorder":
            on = False
        elif on:
            nr.add(x)
    k = 0
    while k < len(rest):
        i, l = rest[k]
        body = l.split("#", 1)[0].strip()
        br = _src_is_branch(l) or _src_is_ret(l)
        if re.match(r"\s*jalr?\b", l):
            return False
        d, u = defs_uses(body)
        if reg in u:
            return False
        if br:
            # the delay slot (noreorder only) runs before the transfer
            slot = i in nr and k + 1 < len(rest)
            if slot:
                sb = rest[k + 1][1].split("#", 1)[0].strip()
                sd, su = defs_uses(sb)
                if reg in su:
                    return False
                if reg in sd:
                    return True
            if _src_is_ret(l):
                return reg not in _FF_RET_LIVE
            lab = re.search(r"(\$L\w+)\s*$", body)
            if not lab:
                return False
            if not _ff_dead_on(lines, ins, lab.group(1), reg, seen):
                return False
            if body.split(None, 1)[0].lower() in ("j", "b"):
                return True
            k += 2 if slot else 1           # fall-through continues after the slot
            continue
        if reg in d:
            return True
        k += 1
    return False


def fallthrough_fill_pass(stext, tgt):
    tb = [k for k, (_w, d) in enumerate(tgt)
          if d.split(None, 1)[0].lower() in _COND_BR_MN] if tgt else []
    lines = stext.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    noreo = False
    ours = []                               # (line index of branch, in noreorder)
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(".set") and "noreorder" in s:
            noreo = True
        elif s.startswith(".set") and s.split()[-1] == "reorder":
            noreo = False
        m = re.match(r"\s*([a-z]+)\b", l)
        if _s_is_insn(l) and m and m.group(1) in _COND_BR_MN:
            ours.append((i, noreo))
    if len(ours) != len(tb):
        return stext
    edits = []
    for (bi, nr), tk in zip(ours, tb):
        if nr or tk + 1 >= len(tgt):
            continue
        tkey = _sm_key(tgt[tk + 1][1].strip(), True)
        if tkey in (None, "skip") or tkey == ("sll", "zero", "zero", 0):
            continue
        # the first fall-through insn with the target's slot key that can be
        # hoisted over every insn before it in the straight-line block
        pick, seen = None, []
        for fi, fl in ins:
            if fi <= bi:
                continue
            if any(_branch_target_label(lines, ins, x) for x in lines[bi + 1:fi]):
                break
            fb = fl.split("#", 1)[0].strip()
            if _src_is_branch(fl) or _src_is_ret(fl) or re.match(r"\s*jalr?\b", fl):
                break
            form = _ff_word_form(fb)
            if form is not None and _sm_key(form, False) == tkey:
                if all(_sm_indep(form, s) for s in seen):
                    pick = (fi, form)
                break
            seen.append(fb)
        if pick is None:
            continue
        fi, form = pick
        d, _u = defs_uses(form)
        lab = re.search(r"(\$L\w+)\s*$", lines[bi].split("#", 1)[0])
        if len(d) != 1 or not lab or not _ff_dead_on(lines, ins, lab.group(1), next(iter(d))):
            continue
        edits.append((bi, fi, form))
    if not edits:
        return stext
    for bi, fi, form in sorted(edits, reverse=True):
        ind = _src_indent(lines[bi])
        del lines[fi]
        lines[bi:bi + 1] = [ind + ".set\tnoreorder", ind + ".set\tnomacro",
                            lines[bi], ind + form, ind + ".set\tmacro",
                            ind + ".set\treorder"]
    return "\n".join(lines)


def ra_restore_sink_pass(stext, tgt):
    stext = callee_restore_sink_s(stext, tgt)
    tl = [d for _w, d in tgt]
    t_ra = [i for i, d in enumerate(tl) if re.match(r"\s*lw\s+\$?ra\s*,", d)]
    if not t_ra:
        return stext
    want = _tail_count(tl[t_ra[-1] + 1:])
    lines = stext.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    o_ra = [p for p, (_i, l) in enumerate(ins)
            if re.match(r"\s*lw\s+\$(?:31|ra)\s*,\s*(-?\d+)\(\$(?:sp|29)\)", l)]
    if not o_ra:
        return stext
    p0 = o_ra[-1]
    k = int(re.match(r"\s*lw\s+\S+\s*,\s*(-?\d+)", ins[p0][1]).group(1))
    have = _tail_count([l for _i, l in ins[p0 + 1:]], src=True)
    need = have - want
    if need <= 0:
        return stext
    last = None
    q = p0 + 1
    while need > 0 and q < len(ins):
        qi, ql = ins[q]
        body = ql.split("#", 1)[0].strip()
        if _src_is_branch(ql) or _src_is_ret(ql) or any(
                _branch_target_label(lines, ins, x) for x in lines[ins[q - 1][0] + 1:qi]):
            return stext
        d, u = defs_uses(body)
        if {"ra", "sp"} & (set(d) | set(u)) and not _frame_load(body):
            return stext
        mo = _mem_operand(body)
        if mo and mo[0] == "sp" and body.split()[0].lower() in _STORE_MN \
                and mo[1] < k + 4 and k < mo[1] + mo[2]:
            return stext
        if not (_frame_load(body) or _src_sp_delta(body) is not None):
            need -= _src_nwords(ql)
        last = q
        q += 1
    if need != 0 or last is None:
        return stext
    li = ins[p0][0]
    moved = lines[li]
    at = ins[last][0]
    out = lines[:li] + lines[li + 1:at + 1] + [moved] + lines[at + 1:]
    return "\n".join(out)


def zero_remat_pass(stext, tgt):
    want = sum(1 for _w, dis in tgt if _ZERO_TGT.match(dis.strip()))
    have = sum(1 for l in stext.split("\n")
               if _s_is_insn(l) and _ZERO_DEF_S.match(l))
    return zero_remat_s(stext, want - have)

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
    # one exit label per function: several exit-merged functions share a file
    ent = ENT_RE.search(stext)
    label = _EXIT_LABEL + ("_" + ent.group(1) if ent else "")
    last_i = rets[-1]
    out = []
    for i, l in enumerate(lines):
        if i == last_i:
            continue                       # drop the last return (keep its delay insn)
        if _s_is_return(l):
            out.append(re.sub(r"(?:j|jr)\s+\$(?:31|ra)\b",
                              "j\t%s" % label, l, count=1))
        else:
            out.append(l)
    ins = len(out)
    for k in range(len(out) - 1, -1, -1):
        if out[k].strip().startswith(".end"):
            ins = k
            break
    exitblk = ["%s:" % label, "\t.set\tnoreorder", "\t.set\tnomacro",
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
    noreorder = False
    while idx < len(lines):
        l = lines[idx]
        st = l.strip()
        if st.startswith(".set"):
            if "noreorder" in st:
                noreorder = True
            elif re.match(r"\.set\s+reorder\b", st):
                noreorder = False
        if _s_is_insn(l) and _COND_S.match(_s_mnem(l)):
            k += 1
            d = idx + 1
            while d < len(lines) and not _s_is_insn(lines[d]):
                if _s_is_label(lines[d]):
                    d = None
                    break
                d += 1
            # only a branch cc1 scheduled itself (inside .set noreorder) has its
            # next insn in the delay slot; in reorder mode the assembler pads it
            filled = noreorder and d is not None and not _src_is_nop(lines[d])
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
    "un_hi_cse_store": un_hi_cse_store_pass,
    "exit_merge": exit_merge_pass,
    "commutative_swap": commutative_swap_s,
    "operand_recolor": operand_recolor_s,
    "laform": laform_fold_pass,
    "base_cse_collapse": base_cse_collapse_pass,
    "shift_const_fold": shift_const_fold_pass,
    "zero_remat": zero_remat_pass,
    "ra_restore_sink": ra_restore_sink_pass,
    "sched_match": sched_match_pass,
    "fallthrough_fill": fallthrough_fill_pass,
}

# Passes that CHANGE the instruction count and so must run BEFORE sigma is derived
# (the register correspondence has to be built from count-aligned assembly). Any
# such pass listed in a recipe is applied ahead of reg_realloc regardless of the
# manifest order; the rest keep their listed order after sigma.
PRE_SIGMA_PASSES = ("un_hi_cse", "un_hi_cse_store", "exit_merge", "base_cse_collapse",
                    "shift_const_fold", "zero_remat")

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
               "prologue_save_hoist", "delay_slot_select")

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
    keep_nop = set()            # branch line idx whose filler leaves a load-delay nop
    pre_fill = {}               # branch line idx -> macro expansion words kept ahead
    tgt_br = [k for k, (_w, d) in enumerate(tgt) if is_branch(d)]
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
        bu = defs_uses(l.strip())[1]
        if bmn in ("jal", "bal"):
            # a call's slot runs before the transfer: an argument set up there still
            # reaches the callee, so $a0-$a3 are not a hazard for the jal itself.
            bu = bu - {"a0", "a1", "a2", "a3"}
        if not (defs_uses(pl.strip())[0] & bu):
            exp = _expand_sym(pl)
            if exp is None:
                exp = _expand_la(pl)
            if exp is not None:
                # aspsx expands the macro first and fills the slot with its LAST
                # word; the address setup stays ahead of the branch.
                pre_fill[i] = exp[:-1]
                move[i] = exp[-1]
                drop.add(pi)
                continue
            move[i] = pl.strip()
            drop.add(pi)
            # aspsx inserts a load-delay nop BEFORE it fills the slot: when the moved
            # insn consumed the load right before it, that nop stays behind in its
            # old place. Keep it explicitly when the target has it.
            if p >= 2 and kbr < len(tgt_br) and tgt_br[kbr] >= 1 \
                    and is_nop(tgt[tgt_br[kbr] - 1][1]):
                ld = ins[p - 2][1].split("#", 1)[0].strip()
                mn = ld.split(None, 1)[0].lower() if ld else ""
                if mn in ("lb", "lbu", "lh", "lhu", "lw", "lwl", "lwr") and \
                        (defs_uses(ld)[0] & defs_uses(pl.split("#", 1)[0].strip())[1]):
                    keep_nop.add(i)
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
            if idx in pre_fill:
                out.append(indent + ".set\tnoat")
                out.extend(indent + w for w in pre_fill[idx])
            if idx in keep_nop:
                out.append(indent + "nop")      # the load-delay nop aspsx left
            out.append(l)                       # the branch
            out.append(indent + move[idx])      # preceding insn -> delay slot
            if idx in pre_fill:
                out.append(indent + ".set\tat")
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
    if (pa and _frame_store(b)) or (pb and _frame_store(a)):
        return True
    # an ALU insn and a load: no memory write, so only registers matter
    if (pa and _is_load(b)) or (pb and _is_load(a)):
        return True
    # a stack-frame load and a store through the assembler temp ($at only ever
    # holds a global symbol's %hi address) touch disjoint memory
    return (_frame_load(a) and _at_store(b)) or (_frame_load(b) and _at_store(a))


def _is_load(disasm):
    p = disasm.split(None, 1)
    return bool(p) and p[0].lower() in ("lb", "lbu", "lh", "lhu", "lw")


def _mem_base(disasm):
    p = disasm.split(None, 1)
    if len(p) < 2:
        return None
    ops = split_ops(p[1])
    mm = re.fullmatch(r".*\((\$?\w+)\)", ops[-1].strip()) if ops else None
    return norm_reg(mm.group(1)) if mm else None


def _frame_load(disasm):
    return _is_load(disasm) and _mem_base(disasm) == "sp"


def _at_store(disasm):
    p = disasm.split(None, 1)
    return bool(p) and p[0].lower() in _STORE_MN and _mem_base(disasm) == "at"


def _word_eq(our_w, tgt_w, tgt_dis):
    """Words equal, tolerating a relocated low-16 immediate: when the target insn
    carries a `%hi`/`%lo`/`%gp_rel` relocation its low half is symbol-dependent
    (our un-relinked assembly leaves it 0), so compare opcode+register fields only."""
    if our_w == tgt_w:
        return True
    if "%hi" in tgt_dis or "%lo" in tgt_dis or "%gp_rel" in tgt_dis:
        return (int(our_w, 16) >> 16) == (int(tgt_w, 16) >> 16)
    if re.match(r"\s*(jal|j)\s+[A-Za-z_.]", tgt_dis):
        # symbolic jump target: its 26-bit field is a relocation, only the
        # opcode is fixed before linking
        return (int(our_w, 16) >> 26) == (int(tgt_w, 16) >> 26)
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
    if len(ins) != n:
        # the assembler padded words the source does not spell (a load-delay or
        # branch-slot nop) or expanded a macro: only the 1:1 prefix before the
        # first such word is safe to reorder.
        k = 0
        while k < min(n, len(ins)):
            if _src_nwords(ins[k][1]) != 1:
                break
            if is_nop(our_words[k][1]) and not _src_is_nop(ins[k][1]):
                break
            k += 1
        if k < 2:
            return span, False
        n = k
    ow = list(our_words[:n])
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


def _mem_operand(line):
    """(base_reg, offset, size) of a load/store `op $r,off($b)` source/disasm line,
    or None. Offset must be a plain integer (a %lo reloc operand returns None)."""
    body = line.split("#", 1)[0].strip()
    p = body.split(None, 1)
    if len(p) < 2:
        return None
    size = {"sb": 1, "sh": 2, "sw": 4, "lb": 1, "lbu": 1, "lh": 2, "lhu": 2,
            "lw": 4}.get(p[0].lower())
    if size is None:
        return None
    ops = split_ops(p[1])
    mm = (re.fullmatch(r"(-?(?:0x[0-9a-fA-F]+|\d+))\((\$?\w+)\)", ops[-1].strip())
          if ops else None)
    if not mm:
        return None
    return _src_reg(mm.group(2)), int(mm.group(1), 0), size


def _global_base(ins, idx, reg):
    """True when the most recent source definition of `reg` before insn `idx` is a
    `lui reg,%hi(S)` or `addiu reg,reg,%lo(S)` -- i.e. reg holds a global's address,
    which can never point into the current stack frame."""
    for k in range(idx - 1, -1, -1):
        d, _u = defs_uses(ins[k][1].split("#", 1)[0].strip())
        if reg in d:
            body = ins[k][1]
            return bool(re.match(r"\s*lui\s+\S+\s*,\s*%hi\(", body) or
                        re.match(r"\s*addiu\s+\S+\s*,\s*\S+\s*,\s*%lo\(", body))
    return False


def _no_alias(ins, ia, ib):
    """Two source memory insns (by index) provably touch disjoint bytes: same base
    register (not redefined between them) with disjoint [off, off+size) ranges, or
    one is $sp-framed and the other's base holds a global address."""
    ma, mb = _mem_operand(ins[ia][1]), _mem_operand(ins[ib][1])
    if ma is None or mb is None:
        return False
    (ba, oa, sa), (bb, ob, sb) = ma, mb
    if ba == bb:
        lo, hi = min(ia, ib), max(ia, ib)
        for k in range(lo + 1, hi):
            if ba in defs_uses(ins[k][1].split("#", 1)[0].strip())[0]:
                return False
        return oa + sa <= ob or ob + sb <= oa
    if ba == "sp" and _global_base(ins, ib, bb):
        return True
    if bb == "sp" and _global_base(ins, ia, ba):
        return True
    return False


def _branch_target_label(lines, ins, l):
    """`l` is a label some instruction in the span references (a real control-flow
    join), as opposed to cc1's `LMn:` line-marker labels, which nothing branches to."""
    if not _s_is_label(l):
        return False
    name = l.strip()[:-1]
    pat = re.compile(r"(?<![\w$.])" + re.escape(name) + r"(?![\w$])")
    return any(pat.search(il.split("#", 1)[0]) for _i, il in ins)


def _slot_sink_mover(lines, ins, sp, our_words, jw, tgt_slot):
    """For the jal at source index `sp` / word index `jw`, find the earlier source
    insn A the target sinks into the delay slot (its word equals `tgt_slot`) and
    return A's source index when moving it down past the intervening insns W and the
    current slot insn B is semantics-preserving; else None.

    The window A..jal must map 1:1 source<->word (only stores and non-macro pure-ALU
    insns, which maspsx never pads), A must be a store or pure-ALU insn, register-
    independent of every insn in W and of B, and -- when A is a store -- provably
    non-aliasing with every memory insn in W. B must be pure-ALU (it only crosses the
    jal, which it already preceded in execution as a delay-slot insn)."""
    def _plain(line):
        b = line.split("#", 1)[0].strip()
        mn = b.split(None, 1)[0].lower() if b else ""
        if mn in ("la",):
            return False                    # macro -> multi-word
        if mn == "li":
            v = split_ops(b.split(None, 1)[1])[-1]
            try:
                n = int(v, 0)
            except ValueError:
                return False
            return -0x8000 <= n <= 0xFFFF
        return mn in _PURE_ALU or mn in _STORE_MN

    def _clean(line):
        return line.split("#", 1)[0].strip()

    if sp + 1 >= len(ins):
        return None
    b = _clean(ins[sp + 1][1])
    if not _pure_alu(b):
        return None
    if sp < 1 or not _plain(ins[sp - 1][1]):
        return None
    for d in range(2, jw + 1):
        p, sa = jw - d, sp - d
        if sa < 0:
            return None
        if not _plain(ins[sa][1]):
            return None
        if not _word_eq(our_words[p][0], tgt_slot[0], tgt_slot[1]):
            continue
        if sa >= 1 and _src_is_branch(ins[sa - 1][1]):
            return None                     # A is another branch's delay slot
        if any(_branch_target_label(lines, ins, l)
               for l in lines[ins[sa][0]:ins[sp][0]]):
            return None                     # a join point inside the window
        a = _clean(ins[sa][1])
        a_store = a.split(None, 1)[0].lower() in _STORE_MN
        if not (a_store or _pure_alu(a)):
            return None
        for k in range(sa + 1, sp):
            w = _clean(ins[k][1])
            if not _indep(a, w):
                return None
            if a_store and w.split(None, 1)[0].lower() in _STORE_MN:
                if not _no_alias(ins, sa, k):
                    return None
        if not _indep(a, b):
            return None
        return sa
    return None


def delay_slot_select_src(span, tgt, our_words):
    """Swap a jal's delay-slot instruction with the instruction immediately before
    the jal when the target has exactly those two transposed. Retail's cc1 and ours
    both compute two independent argument-setup instructions before a call but pick
    a different one to sink into the branch delay slot; ours does the reverse. A
    delay-slot instruction executes before control transfers, so both orderings
    present identical argument registers to the callee -- swapping the two is
    semantics-preserving when they are register-independent. reorder_indep cannot
    express this because the move crosses the jal (a control transfer it will not
    hop). Target-guided and minimal: fires only on an exact A<->B transposition around
    a jal where both A and B are pure-ALU and mutually independent and the slot does
    not already match.

    Anchored on the jal, NOT on a global 1:1 source-word mapping: the assembler inserts
    load-delay nops elsewhere in the body, so the source-insn and word counts differ.
    But a jal's immediate word neighbours DO correspond to its immediate source
    neighbours -- the pre insn is pure-ALU (we check) so no nop is inserted between it
    and the jal, and the following word is the delay slot itself. So align the k-th jal
    across our words, the target words and the source insns, and read A/B from the jal's
    neighbours in each. The touched region is wrapped in `.set noreorder` so the
    assembler keeps the schedule. Returns (new_span, fired)."""
    def _is_call(dis):
        p = dis.split(None, 1)
        return bool(p) and p[0].lower() in ("jal", "bal", "jalr")
    lines = span.split("\n")
    ins = [(i, l) for i, l in enumerate(lines) if _s_is_insn(l)]
    our_jal = [k for k, (_w, d) in enumerate(our_words) if _is_call(d)]
    tgt_jal = [k for k, (_w, d) in enumerate(tgt) if _is_call(d)]
    src_jal = [p for p, (_i, l) in enumerate(ins)
               if re.match(r"\s*(jal|bal|jalr)\b", l)]
    if not our_jal or len(our_jal) != len(tgt_jal) or len(our_jal) != len(src_jal):
        return span, False
    swaps = []                              # (pre_pos, jal_pos, slot_pos) in ins
    for k in range(len(our_jal)):
        jw, tw, sp = our_jal[k], tgt_jal[k], src_jal[k]
        if jw < 1 or jw + 1 >= len(our_words):
            continue
        if tw < 1 or tw + 1 >= len(tgt):
            continue
        if sp < 1 or sp + 1 >= len(ins):
            continue
        aw, bw = our_words[jw - 1], our_words[jw + 1]
        if _word_eq(bw[0], tgt[tw + 1][0], tgt[tw + 1][1]):
            continue                        # slot already correct
        if not _word_eq(bw[0], tgt[tw - 1][0], tgt[tw - 1][1]):
            continue                        # our slot insn is not the target's pre
        if _word_eq(aw[0], tgt[tw + 1][0], tgt[tw + 1][1]):
            a = ins[sp - 1][1].strip()
            b = ins[sp + 1][1].strip()
            if not (_pure_alu(a) and _pure_alu(b) and _indep(a, b)):
                continue
            swaps.append((sp - 1, sp, sp + 1))
            continue
        # sink: the target's slot insn sits FURTHER back in ours (a store, e.g. a
        # struct field init, that retail's scheduler sank into the call's slot).
        m = _slot_sink_mover(lines, ins, sp, our_words, jw, tgt[tw + 1])
        if m is not None:
            swaps.append((m, sp, sp + 1))
    if not swaps:
        return span, False
    order = list(range(len(ins)))
    wrapped = set()
    for pre, j, slot in sorted(swaps, reverse=True):
        # A (at `pre`) goes to the slot; B (the old slot) goes just before the jal;
        # everything between keeps its order. The adjacent case is a transposition.
        blk = order[pre:slot + 1]
        order[pre:slot + 1] = blk[1:-2] + [blk[-1], blk[-2], blk[0]]
        wrapped |= set(range(pre, slot + 1))
    final = {p: ins[order[p]][1].strip() for p in range(len(ins))}
    line_pos = {ins[p][0]: p for p in range(len(ins))}
    out = []
    for idx, l in enumerate(lines):
        if idx in line_pos:
            p = line_pos[idx]
            indent = _src_indent(ins[p][1])
            if p in wrapped and (p - 1) not in wrapped:
                out.append(indent + ".set\tnoreorder")
            out.append(indent + final[p])
            if p in wrapped and (p + 1) not in wrapped:
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


def _sym_operand(line):
    """(op, has_index) when the insn's memory operand is a bare symbol (`S`,
    `S+k`, `S($r)`): a cc1 no-split macro the assembler expands. Else None."""
    b = line.split("#", 1)[0].strip()
    p = b.split(None, 1)
    if len(p) < 2:
        return None
    op = p[0].lower()
    if op not in _STORE_MN and op not in ("lb", "lbu", "lh", "lhu", "lw", "lwl", "lwr"):
        return None
    ops = split_ops(p[1])
    m = re.fullmatch(r"([A-Za-z_.][\w.$]*(?:\+\d+)?)(\(\$\w+\))?", ops[-1].strip())
    if not m:
        return None
    return op, bool(m.group(2))


def _expand_sym(line):
    """The aspsx expansion (list of source insns) of a symbolic-address load/store
    macro, or None when `line` is not one. Loads use the destination as the temp
    unless it is the index register (then $at); stores always use $at."""
    so = _sym_operand(line)
    if not so:
        return None
    b = line.split("#", 1)[0].strip()
    op, rest = b.split(None, 1)
    ops = split_ops(rest)
    r = ops[0].strip()
    m = re.fullmatch(r"([A-Za-z_.][\w.$]*(?:\+\d+)?)(?:\((\$\w+)\))?", ops[-1].strip())
    sym, idx = m.group(1), m.group(2)
    is_load = not op.startswith("s")
    t = r if (is_load and r != idx and op not in ("lwl", "lwr")) else "$1"
    out = ["lui\t%s,%%hi(%s)" % (t, sym)]
    if idx:
        out.append("addu\t%s,%s,%s" % (t, t, idx))
    out.append("%s\t%s,%%lo(%s)(%s)" % (op, r, sym, t))
    return out


def _expand_la(line):
    """The aspsx expansion of an address macro `la $r,SYM[+k]` (lui + addiu %lo),
    or None. Only used where a pass must split the macro (delay-slot filling)."""
    b = line.split("#", 1)[0].strip()
    p = b.split(None, 1)
    if len(p) < 2 or p[0] != "la":
        return None
    ops = split_ops(p[1])
    if len(ops) != 2:
        return None
    r, sym = ops[0].strip(), ops[1].strip()
    if not re.fullmatch(r"[A-Za-z_.][\w.$]*(?:\+\d+)?", sym):
        return None
    return ["lui\t%s,%%hi(%s)" % (r, sym), "addiu\t%s,%s,%%lo(%s)" % (r, r, sym)]


def _src_nwords(line):
    """Assembled word count of one cc1 source insn: symbolic-address loads/stores
    expand to lui(+addu)+access, `la` and out-of-range `li` to two words."""
    so = _sym_operand(line)
    if so:
        return 3 if so[1] else 2
    b = line.split("#", 1)[0].strip()
    p = b.split(None, 1)
    if p and p[0] == "la":
        return 2
    if p and p[0] == "li" and len(p) > 1:
        try:
            v = int(split_ops(p[1])[-1], 0)
        except ValueError:
            return 1
        return 1 if (-0x8000 <= v <= 0xFFFF) else 2
    return 1


def _frame_store_disjoint(a, b):
    """Both are $sp-framed stores to non-overlapping slots."""
    if not (_frame_store(a) and _frame_store(b)):
        return False
    ma, mb = _mem_operand(a), _mem_operand(b)
    if ma is None or mb is None:
        return False
    return ma[1] + ma[2] <= mb[1] or mb[1] + mb[2] <= ma[1]


def _global_load(line):
    """A load from a global symbol's storage (symbolic operand), which can never
    alias a store into the current stack frame."""
    so = _sym_operand(line)
    if so:
        return not so[0].startswith("s")
    b = line.split("#", 1)[0].strip()
    return _is_load(b) and "%lo(" in b


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
    if s_ra is None:
        return span, False
    # `desired` is a target WORD index; map it to a source insn index through each
    # source insn's assembled length (symbolic-address macros expand to 2-3 words).
    w, d_src = 0, None
    for k, l in enumerate(ins):
        if w >= desired:
            d_src = k
            break
        w += _src_nwords(l)
    if d_src is None or w != desired:
        return span, False
    desired = d_src
    if desired >= s_ra:
        return span, False
    mover = ins[s_ra].strip()
    for k in range(desired, s_ra):
        other = ins[k].strip()
        if not _indep(mover, other):
            return span, False
        if not (_reorder_swappable(mover, other) or _frame_store_disjoint(mover, other)
                or _global_load(other)):
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
    "delay_slot_select": delay_slot_select_src,
}


def emit_noreorder_span(span, tgt, our_words, reorder_passes, reassemble=None):
    """Apply the reordering source passes (delay_fill / epilogue_unfill) to a span
    in order. Each rewrites the cc1 source and wraps the touched instructions in
    `.set noreorder` so the assembler keeps the schedule. `tgt` is the target's
    (pad-stripped) [(word, disasm)] list; `our_words` is our assembled span.

    A word-consuming pass (delay_fill / reorder_indep / delay_slot_select) reads the
    ASSEMBLED words of the CURRENT source. When an earlier pass in the recipe already
    reordered the source (e.g. prologue_save_hoist ahead of delay_slot_select), the
    words from before that pass are stale and the positional guidance is wrong. If a
    `reassemble(txt) -> words` callback is supplied, refresh the words from the current
    span text before each word-consuming pass; otherwise fall back to the words passed
    in (correct when no earlier pass moved anything)."""
    txt = span
    for name in reorder_passes:
        fn = _REORDER_SRC.get(name)
        if fn is None:
            raise NotImplementedError("reorder pass not wired: %s" % name)
        if name in ("delay_fill", "reorder_indep", "delay_slot_select"):
            words = reassemble(txt) if reassemble is not None else our_words
            txt, _fired = fn(txt, tgt, words)
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


# --------------------------------------------------------------------------
# nosplit (META pass, not a rewrite). The retail executable links translation
# units built with DIFFERENT cc1 address-generation settings: some were compiled
# with split addresses (cc1 itself emits `lui $r,%hi(S)` / `%lo(S)($r)` pairs and
# CSEs/la-forms them), others with `-mno-split-addresses`, where cc1 emits the
# symbolic macro operand (`lw $r,S($idx)`, `sw $r,S`) and the assembler (aspsx /
# maspsx) expands it: `lui $dst,%hi(S)` + `addu` + `lw $dst,%lo(S)($dst)` for a
# load, a fresh `lui $at` for a store or when the destination is the index. The
# project may compile several retail units as one C file, so a function from a
# no-split unit is recovered by
# compiling the file a second time with -mno-split-addresses and splicing that
# function's `.ent ... .end` span in place of the default one. It is still plain
# cc1 output of the same C; the other passes in the recipe then run on it as usual.
# Local `$L`/`LM` labels are renamed in the spliced span so its numbering (which
# can drift between the two compiles) cannot collide with the rest of the file.
#
# Units also differ in who fills branch delay slots: some let cc1's delayed-branch
# pass fill them (`.set noreorder` blocks in the cc1 output), others were built
# with -fno-delayed-branch and left it to aspsx, which first inserts load-delay
# nops and then fills a slot with the preceding insn (delay_fill models that). So
# each META pass names one alternate compile ("flavor") of the same file.
# --------------------------------------------------------------------------
ALT_FLAVORS = {
    "nosplit": ["-mno-split-addresses"],
    "nosplit_nodb": ["-mno-split-addresses", "-fno-delayed-branch"],
    "nodb": ["-fno-delayed-branch"],
}
META_PASSES = tuple(ALT_FLAVORS)
# local-label prefix per alternate compile (nosplit keeps the historical "ns")
_ALT_TAG = {"nosplit": "ns", "nosplit_nodb": "nsnd", "nodb": "nd"}


def alt_flavors(manifest):
    """The alternate-compile flavors a manifest needs."""
    return sorted({p for v in manifest.values() for p in v.get("passes", [])
                   if p in ALT_FLAVORS})


def _lc_body(text, lc):
    """The data lines of constant `lc` (from its label to the next label or
    section directive), or None when it is not defined."""
    m = re.search(r"(?m)^%s:\s*\n((?:[ \t]+\.(?:ascii|byte|half|word|space|align)"
                  r"\b.*\n)*)" % re.escape(lc), text)
    return m.group(1) if m else None


def expand_sym_macros(span):
    """Expand every symbolic-address load/store macro in a span into the explicit
    words aspsx emits (see _expand_sym), so the source keeps a 1:1 insn<->word map
    for the word-level passes. Inside a `.set noreorder` block the expansion is
    left to the assembler (a macro there is already placed by cc1)."""
    out, noreorder = [], False
    for l in span.split("\n"):
        s = l.strip()
        if s.startswith(".set"):
            if "noreorder" in s:
                noreorder = True
            elif re.match(r"\.set\s+reorder\b", s):
                noreorder = False
        exp = None if (noreorder or not _s_is_insn(l)) else _expand_sym(l)
        if exp is None:
            out.append(l)
            continue
        ind = _src_indent(l)
        uses_at = any(re.search(r"\$1(?![0-9])", w) for w in exp)
        if uses_at:
            out.append(ind + ".set\tnoat")
        out.extend(ind + w for w in exp)
        if uses_at:
            out.append(ind + ".set\tat")
    return "\n".join(out)


def aspsx_label_nops(span):
    """aspsx puts a load-delay nop straight after the load, even when a label
    separates the load from the insn that consumes it; maspsx emits `label: nop`,
    which moves every branch into that label one word early. Spell the nop out
    right after the load (reorder mode only) so maspsx sees no hazard left."""
    lines = span.split("\n")
    out, noreorder = [], False
    for k, l in enumerate(lines):
        out.append(l)
        s = l.strip()
        if s.startswith(".set"):
            if "noreorder" in s:
                noreorder = True
            elif re.match(r"\.set\s+reorder\b", s):
                noreorder = False
        if noreorder or not _s_is_insn(l):
            continue
        b = l.split("#", 1)[0].strip()
        if not _is_load(b):
            continue
        d = defs_uses(b)[0]
        saw_label, j = False, k + 1
        while j < len(lines) and not _s_is_insn(lines[j]):
            if _s_is_label(lines[j]):
                saw_label = True
            j += 1
        if saw_label and j < len(lines):
            nb = lines[j].split("#", 1)[0].strip()
            # a direct `jal` reads no register when it issues (its delay slot
            # runs first); defs_uses models its argument registers as uses
            uses = set() if re.match(r"jal\s", nb) else defs_uses(nb)[1]
            if d & uses:
                out.append(_src_indent(l) + "nop")
    return "\n".join(out)


def splice_alt_spans(text, alt_text, names, tag="ns"):
    """Replace each function span in `names` with the same function's span from
    `alt_text`. Returns (new_text, [spliced names]). `tag` keeps the renamed local
    labels unique per alternate compile: the LM line-marker counters of two flavor
    compiles overlap, so two flavors must not share a prefix."""
    alt = {n: alt_text[s:e] for n, s, e in split_spans(alt_text)}
    done = []
    for name, start, end in sorted(split_spans(text), key=lambda x: -x[1]):
        if name not in names:
            continue
        if name not in alt:
            raise RuntimeError("alt compile: %s missing" % name)
        span = alt[name]
        for lc in set(re.findall(r"\$LC\d+", span)):
            if _lc_body(text, lc) is None or _lc_body(text, lc) != _lc_body(alt_text, lc):
                raise RuntimeError("alt compile: %s references %s, which differs between "
                                   "the two compiles" % (name, lc))
        span = re.sub(r"\$L(?!C)(\w+)", r"$L%s_\1" % tag, span)
        span = re.sub(r"(?<![\w$.])LM(\d+)\b", r"LM%s\1" % tag, span)
        span = aspsx_label_nops(expand_sym_macros(span))
        text = text[:start] + span + text[end:]
        done.append(name)
    return text, done


def normalize_s(s_file, ctx, manifest=None):
    """Splice-normalize the manifest functions in a cc1 .s in place. Reads once,
    rewrites once. Returns the sorted list of function names it rewrote.

    ctx keys: python, maspsx_py, maspsx_flags, as_bin, maspsx_as_flags, objdump,
    run (callable -> (rc, out, err)), asm_root."""
    if manifest is None:
        manifest = load_manifest()
    text = _read_bytes_str(s_file)
    rewrote = []
    # Pass 0 -- alternate-compile flavors (nosplit, ...): take the function's span
    # from that flavor's cc1 output before any other pass sees it.
    for flavor in alt_flavors(manifest):
        ns = [n for n in manifest if flavor in manifest[n].get("passes", [])]
        alt = (ctx.get("alt_s") or {}).get(flavor)
        if not alt or not os.path.exists(alt):
            raise RuntimeError("%s functions %s need ctx['alt_s'][%r]"
                               % (flavor, ns, flavor))
        text, done = splice_alt_spans(text, _read_bytes_str(alt), ns,
                                      _ALT_TAG.get(flavor, flavor))
        _write_bytes_str(s_file, text)
        rewrote.extend(done)
    spans = split_spans(text)
    out = text
    # Pass 1 -- text passes (reg_realloc / commutative_swap / laform). Splice from
    # the tail so earlier (start,end) offsets stay valid; words are always assembled
    # from the ON-DISK cc1 .s (unmodified until the pass-1 write below).
    for name, start, end in sorted(spans, key=lambda x: -x[1]):
        if name not in manifest:
            continue
        passes = [p for p in manifest[name].get("passes", []) if p not in META_PASSES]
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
                                "prologue_save_hoist", "delay_slot_select")]
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

            def _reasm(span_txt, _s=start, _e=end, _n=name):
                """Refresh the assembled words from the current (partially reordered)
                span so a later word-consuming pass sees the real schedule."""
                _write_bytes_str(s_file, out[:_s] + span_txt + out[_e:])
                w, werr = assemble_words(ctx, s_file, _n)
                if werr:
                    raise RuntimeError("assemble %s (reorder-refresh):\n%s" % (_n, werr))
                return _strip_trailing_pad(w)

            new_span = emit_noreorder_span(out[start:end], tgt, our, reorder,
                                           reassemble=_reasm)
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
    return sorted(set(rewrote))
