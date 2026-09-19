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


PASSES = {
    # reg_realloc is applied specially (it needs sigma from words); the ordered
    # list in the manifest still names it so the recipe is explicit and auditable.
    "reg_realloc": None,
    "commutative_swap": commutative_swap_s,
}


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
    rc, o, e = ctx["run"]([ctx["objdump"], "-d", obj])
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
    # Splice from the tail so earlier (start,end) offsets stay valid; words are
    # always assembled from the ON-DISK cc1 .s (unmodified until the final write).
    for name, start, end in sorted(spans, key=lambda x: -x[1]):
        if name not in manifest:
            continue
        passes = manifest[name].get("passes", [])
        if not passes:
            continue
        our, err = assemble_words(ctx, s_file, name)
        if err:
            raise RuntimeError("assemble %s:\n%s" % (name, err))
        tgt = target_words(ctx["asm_root"], name)
        if tgt is None:
            raise RuntimeError("no target .s found for %s under %s"
                               % (name, ctx["asm_root"]))
        tgt = _strip_trailing_pad(tgt)
        our = _strip_trailing_pad(our)
        sigma, _conf = derive_sigma(our, tgt)
        norm = normalize_span_text(out[start:end], tgt, sigma, passes)
        out = out[:start] + norm + out[end:]
        rewrote.append(name)
    _write_bytes_str(s_file, out)
    return sorted(rewrote)
