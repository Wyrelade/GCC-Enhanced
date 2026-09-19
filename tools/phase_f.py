#!/usr/bin/env python3
"""Phase F: splice-normalize a compiler .s in place, function by function.

This is the integration seam of GCC-Enhanced. The POST solver (solve.py) proves,
per function, a semantics-preserving rewrite of a near-miss cc1 output whose bytes
equal the known target. Phase F applies those proven rewrites INSIDE a real build:
after cc1 emits the whole translation unit's `.s`, and before the assembler runs,
each function named in a manifest is normalized with the same solve.py operators;
every other function is left byte-for-byte verbatim.

Two ways to wire it into a project build without forking the build script:

  (a) LOCAL / non-invasive -- import the project's build module and reassign its
      per-file compile step to a wrapper that calls `normalize_s(s_path, manifest)`
      between cc1 and the assembler, then run the build normally. Because Python
      resolves a module global at call time, reassigning `buildmod.compile_c`
      takes effect and the tracked build file is byte-untouched. Best when the
      normalizer must stay out of the public tree.

  (b) UPSTREAM -- once the normalizer is stable, call `normalize_s` from the build
      directly. Makes the matches shippable from a clean checkout.

Manifest (JSON): {"<function>": {"retty": "void"|"int"}}. Empty/absent = no-op.

Configuration comes from asmlib's environment (GCCE_ROOT, GCCE_CC1, GCCE_MASPSX,
GCCE_AS, GCCE_OBJDUMP, GCCE_TARGET_DIR); no project symbols or paths are baked in.
"""
import os
import re
import sys
import json

import asmlib
import solve

# A function's code span. `\S+` stops before a CRLF `\r` (it is whitespace), so
# no end-of-line anchor is needed and none should assume `\r` is present.
ENT_RE = re.compile(r'(?m)^[ \t]*\.ent[ \t]+(\S+)')


def _read_bytes_str(path):
    """Read a .s as raw bytes decoded 1:1 (latin-1). NEVER read a compiler .s in
    text mode: it is frequently CRLF, and universal-newline read + `\\n` write
    silently strips every `\\r`, so unchanged regions no longer round-trip
    byte-for-byte."""
    with open(path, "rb") as f:
        return f.read().decode("latin-1")


def _write_bytes_str(path, s):
    with open(path, "wb") as f:
        f.write(s.encode("latin-1"))


def split_spans(text):
    """[(name, start, end)] for each `.ent NAME .. .end NAME` code span, `end`
    just past the newline that ends the `.end NAME` line."""
    spans = []
    for m in ENT_RE.finditer(text):
        name = m.group(1)
        endm = re.search(r'(?m)^[ \t]*\.end[ \t]+%s\b' % re.escape(name),
                         text[m.start():])
        if not endm:
            continue
        end = m.start() + endm.end()
        nl = text.find("\n", end)
        spans.append((name, m.start(), len(text) if nl < 0 else nl + 1))
    return spans


def normalize_span_text(span, tgt, sigma):
    """The .s-text operators, applied to one function's span only: reg-realloc
    rename, then commutative-operand swap. Function-local, no cross-function
    effects. (Word-level passes -- delay fill, epilogue unfill -- are not yet
    emitted here; a function whose recipe needs them will fail the final byte
    check loudly rather than ship wrong bytes.)"""
    txt = solve.apply_sigma_to_s(span, sigma)
    txt = solve.commutative_swap_s(txt, tgt)
    return txt


def normalize_s(s_path, manifest):
    """Splice-normalize the manifest functions in a cc1 .s in place. Reads once,
    writes once (byte-identical when the manifest hits nothing). Returns the set
    of functions rewritten."""
    text = _read_bytes_str(s_path)
    out = text
    rewrote = set()
    # Splice from the tail so earlier (start, end) offsets stay valid; sigma is
    # always derived from the ON-DISK cc1 words before any write.
    for name, start, end in sorted(split_spans(text), key=lambda x: -x[1]):
        if name not in manifest:
            continue
        our, err = asmlib.assemble_words(s_path, name)
        if err:
            raise RuntimeError("assemble %s:\n%s" % (name, err))
        tgt = solve._strip_trailing_pad(asmlib.target_words(name))
        our = solve._strip_trailing_pad(our)
        sigma, _conf = solve.derive_sigma(our, tgt)
        out = out[:start] + normalize_span_text(out[start:end], tgt, sigma) + out[end:]
        rewrote.add(name)
    _write_bytes_str(s_path, out)
    return rewrote


def honesty_gate(manifest, cfile_of):
    """Refuse to proceed unless solve.solve_one reports `verified` for every
    manifest function. `cfile_of(name)` returns that function's C stub path."""
    ok = True
    for func in sorted(manifest):
        retty = manifest[func].get("retty", "void")
        r = solve.solve_one(cfile_of(func), func, retty)
        st = r.get("status")
        print("gate: %-20s solve=%s" % (func, st))
        if st != "verified":
            print("  reason:", r.get("equiv_reason") or r.get("error") or "(not verified)")
            ok = False
    return ok


def main():
    if len(sys.argv) < 3:
        print("usage: phase_f.py <file.s> <manifest.json>")
        return 2
    s_path, manifest_path = sys.argv[1], sys.argv[2]
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    rewrote = normalize_s(s_path, manifest)
    print("normalized: %s" % (", ".join(sorted(rewrote)) if rewrote else "(none)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
