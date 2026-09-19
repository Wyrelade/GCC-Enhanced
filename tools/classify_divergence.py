#!/usr/bin/env python3
"""GCC-Enhanced Phase A - divergence corpus and taxonomy.

Parses a matching-decomp project's parked-function journal (a text file where each line begins
with a function symbol and describes why that function does not yet match) into a structured
divergence taxonomy. Classifies each function into the GCC-Enhanced wall families, flags
body-perfect near-misses, extracts pad counts and permuter scores, writes divergence.json, and
prints a ranked build order (the most common wall family is the highest-yield solver to build
first).

Usage:
    python classify_divergence.py <journal_file> [out.json]

The journal format is intentionally loose: one function per line, starting with the symbol name
(for example "func_80012345 ... note text ..."). Adapt the FAMILIES keywords to your project's
vocabulary.
"""
import os, re, sys, json, collections

FAMILIES = {
    "reg-naming":   [r"reg[- ]swap", r"reg[- ]nam", r"reg[- ]alloc", r"coloring", r"colou?r",
                     r"base[- ]reg", r"register[- ]number", r"register rename", r"v0/v1",
                     r"saved[- ]reg", r"reg shift"],
    "addr-form":    [r"la[- ]fold", r"la[- ]form", r"\bsplit\b", r"address[- ]material",
                     r"lui\+addiu"],
    "hi-cse":       [r"%hi[- ]CSE", r"address-%hi", r"\bCSE\b", r"rematerial", r"LICM"],
    "epilogue-fill":[r"UNFILL", r"epilogue", r"FILL/UNFILL", r"jr[- ]RETURN", r"jr-return",
                     r"dealloc"],
    "delay-slot":   [r"delay[- ]slot", r"delay slot", r"j-delay", r"load[- ]delay",
                     r"branch[- ]delay"],
    "single-exit":  [r"single[- ]exit", r"branch[- ]merge", r"cross[- ]jump", r"tail[- ]merge",
                     r"shared exit", r"j-merge"],
    "operand-canon":[r"canonicaliz", r"OR[- ]tree", r"reassoc", r"operand[- ]reorder",
                     r"coalesc"],
    "loop-rotation":[r"loop[- ]rotation", r"rotation", r"peel", r"sign[- ]test", r"induction"],
    "frame-size":   [r"frame[- ]size", r"frame size", r"frame 0x"],
    "multu-sign":   [r"multu", r"sign-extended"],
    "blocked-class":[r"\$at", r"gp_rel", r"GTE", r"BIOS", r"jtbl", r"thunk"],
}
FAM_RE = {k: re.compile("|".join(v), re.I) for k, v in FAMILIES.items()}
SYM = re.compile(r"^([A-Za-z_][\w$.]*)\b")
BODY_PERFECT = re.compile(r"byte[- ]perfect|body[- ]perfect|near[- ]perfect", re.I)
PERM = re.compile(r"permuter[^.]*?(\d{2,5})|near-miss (\d{2,5})", re.I)
PAD = re.compile(r"pad(?:nop)?\s*(?:is\s*)?(\d+)|(\d+)\s*pad", re.I)


def classify(journal):
    rows = {}
    with open(journal, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SYM.match(line)
            if not m or not re.search(r"\d", m.group(1)):
                continue
            fn = m.group(1)
            classes = sorted(k for k, rx in FAM_RE.items() if rx.search(line))
            pm = PERM.search(line)
            pd = PAD.search(line)
            rows[fn] = {
                "func": fn,
                "body_perfect": bool(BODY_PERFECT.search(line)),
                "classes": classes,
                "n_classes": len(classes),
                "permuter_score": next((int(g) for g in (pm.groups() if pm else []) if g), None),
                "pad_words": next((int(g) for g in (pd.groups() if pd else []) if g), None),
                "note": line.strip(),
            }
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    journal = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "divergence.json"
    rows = classify(journal)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    hist = collections.Counter(c for r in rows.values() for c in r["classes"])
    print(f"parsed {len(rows)} functions -> {out}\n")
    print("== wall-family frequency (drives solver build order) ==")
    for c, n in hist.most_common():
        print(f"  {c:14s} {n}")

    cand = [r for r in rows.values() if r["body_perfect"] and "blocked-class" not in r["classes"]]
    cand.sort(key=lambda r: (r["n_classes"], -(r["permuter_score"] or 0)))
    print("\n== first-batch solver targets (body-perfect, fewest classes) ==")
    for r in cand[:15]:
        print(f"  {r['func']}  classes={r['classes']}  perm={r['permuter_score']}")


if __name__ == "__main__":
    main()
