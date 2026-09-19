#!/usr/bin/env python3
"""GCC-Enhanced Phase C: target-guided equivalence solver.

Given our cc1 near-miss and the KNOWN target bytes, search a small set of
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

VOID_LIVE = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "gp", "sp", "fp", "ra"]

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
# delay-slot fill POST pass (rule-general)
# ---------------------------------------------------------------------------
_BRANCHY = re.compile(r"^(b|j|jr|jal|beq|bne|blez|bgtz|bltz|bgez|beqz|bnez)")
def is_branch(disasm):
    return bool(_BRANCHY.match(disasm.split(None, 1)[0].lower()))
def is_nop(disasm):
    return disasm.strip() in ("nop", "sll zero,zero,0", "sll zero,zero,0x0")

def delay_fill(words):
    """Fill a branch/jr delay slot (currently nop) with the immediately-preceding
    independent instruction, dropping the nop. Only moves an instruction that neither
    writes a register the branch reads nor is itself control flow -- the exact SN aspsx
    schedule. Returns a new word list (possibly shorter)."""
    out = list(words)
    i = 0
    while i < len(out) - 1:
        dis = out[i][1]
        if is_branch(dis) and is_nop(out[i + 1][1]) and i >= 1:
            prev = out[i - 1]
            if not is_branch(prev[1]) and not is_nop(prev[1]):
                # independence: branch must not read prev's destination register
                pm, pr, ps = insn_parts(prev[1])
                pdef = E.defs_uses(_mk_insn(prev[1]))[0]
                bruse = E.defs_uses(_mk_insn(dis))[1]
                if not (pdef & bruse):
                    # move prev into the delay slot: [..., prev, br, nop] -> [..., br, prev]
                    out[i - 1], out[i] = out[i], prev      # swap br above prev
                    del out[i + 1]                          # drop the nop
                    i += 1
                    continue
        i += 1
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
    live = VOID_LIVE if retty == "void" else None

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

    # 1) reg-realloc
    sigma, conflicts = derive_sigma(our, tgt)
    r["sigma"] = sigma; r["conflicts"] = conflicts

    stext = open(s0, encoding="utf-8", errors="replace").read()
    s1 = os.path.join(SCRATCH, func + ".sigma.s")
    open(s1, "w", encoding="utf-8").write(apply_sigma_to_s(stext, sigma))
    renamed, err = asmlib.assemble_words(s1, func)
    if err:
        r["status"] = "error"; r["error"] = err; return r

    # 2) delay-slot fill POST pass (needs the nop still present)
    final = _strip_trailing_pad(delay_fill(renamed))
    match_ok, rows = cmp_words(tgt, final)
    r["bytes_ok"] = match_ok; r["rows"] = rows

    # equivalence gate
    our_sym, err = asmlib.assemble_words_reloc(s0, func)
    if err:
        r["status"] = "error"; r["error"] = err; return r
    fa = func_from_words(func, _strip_trailing_pad(our_sym))
    fb = E.parse_s(os.path.join(asmlib.TARGET_DIR, func + ".s"), func)
    res = E.check_equiv(fa, fb, live=live, regmap=sigma)
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

if __name__ == "__main__":
    main()
