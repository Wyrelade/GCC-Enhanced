#!/usr/bin/env python3
"""GCC-Enhanced toolchain harness: compile C -> cc1 gas .s, assemble a .s to instruction
words through the project's real maspsx+as toolchain, and read a splat-style target .s.

This module is project-agnostic. Point it at your toolchain and target tree with
environment variables (or import it and set the module globals before use):

    GCCE_ROOT         project root (default: current directory)
    GCCE_CPP          C preprocessor / driver exe (gcc.exe used with -E)
    GCCE_CC1          the cc1 executable (e.g. a PsyQ CC1PSX.EXE or built cc1)
    GCCE_MASPSX       path to maspsx.py
    GCCE_AS           mips assembler exe (binutils mips-*-as)
    GCCE_OBJDUMP      mips objdump exe
    GCCE_TARGET_DIR   directory of per-function target .s files (<func>.s)
    GCCE_INCLUDE      colon-separated include dirs for the preprocessor (optional)

CC1/CPP/maspsx flags default to the common PSX GCC 2.8.x recipe and can be overridden by
assigning to the module-level *_FLAGS lists after import.
"""
import os, re, sys, subprocess

def _env(name, default=""):
    return os.environ.get(name, default)

ROOT = os.path.abspath(_env("GCCE_ROOT", os.getcwd()))

# Common PSX GCC 2.8.x recipe. Override by reassigning after import if your project differs.
CC1_FLAGS = ["-O2", "-mips1", "-mcpu=3000", "-w", "-funsigned-char", "-fpeephole",
    "-ffunction-cse", "-fpcc-struct-return", "-fcommon", "-msoft-float", "-mgas",
    "-fgnu-linker", "-gcoff", "-G0", "-quiet"]
CPP_FLAGS = ["-E", "-P", "-undef", "-nostdinc"]
MASPSX_FLAGS = ["--aspsx-version=2.77", "--run-assembler", "--expand-div"]
MASPSX_AS_FLAGS = ["-EL", "-march=r3000", "-mtune=r3000", "-no-pad-sections", "-G0"]

CPP = _env("GCCE_CPP")
CC1 = _env("GCCE_CC1")
MASPSX = _env("GCCE_MASPSX")
AS = _env("GCCE_AS")
OBJDUMP = _env("GCCE_OBJDUMP")
TARGET_DIR = _env("GCCE_TARGET_DIR")

def _includes():
    inc = _env("GCCE_INCLUDE")
    out = []
    for d in inc.split(os.pathsep) if inc else []:
        if d:
            out += ["-I", d]
    return out

def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       stdin=subprocess.DEVNULL)
    return r.returncode, r.stdout, r.stderr

def compile_to_s(cfile, out_s):
    """C -> preprocessed -> cc1 gas assembly (BEFORE maspsx). Returns (ok, err)."""
    i = out_s + ".i"
    rc, o, e = run([CPP] + CPP_FLAGS + _includes() + ["-o", i, cfile])
    if rc:
        return False, "CPP:\n" + o + e
    rc, o, e = run([CC1] + CC1_FLAGS + ["-o", out_s, i])
    if rc:
        return False, "CC1:\n" + o + e
    return True, None

def _maspsx_cmd(sfile, obj):
    return ([sys.executable, MASPSX] + MASPSX_FLAGS + ["--gnu-as-path=%s" % AS] +
            MASPSX_AS_FLAGS + _includes() + ["-o", obj, sfile])

def _objdump_fn(o, fn):
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
    return words

def assemble_words(sfile, fn):
    """gas .s -> maspsx --run-assembler -> objdump; return [(be_hex, disasm)] for fn."""
    obj = sfile + ".o"
    rc, o, e = run(_maspsx_cmd(sfile, obj))
    if rc:
        return None, "MASPSX:\n" + o + e
    rc, o, e = run([OBJDUMP, "-d", obj])
    if rc:
        return None, "OBJDUMP:\n" + o + e
    return _objdump_fn(o, fn), None

def assemble_words_reloc(sfile, fn):
    """Like assemble_words but folds R_MIPS_HI16/LO16 relocations back into the operand as
    %hi(sym)/%lo(sym), so the disassembly carries the symbol the .o zeroes out. Needed to
    build a symbol-faithful function for the equivalence gate."""
    obj = sfile + ".o"
    rc, o, e = run(_maspsx_cmd(sfile, obj))
    if rc:
        return None, "MASPSX:\n" + o + e
    rc, o, e = run([OBJDUMP, "-dr", obj])
    if rc:
        return None, "OBJDUMP:\n" + o + e
    out, in_fn, last = [], False, None
    for line in o.splitlines():
        if re.match(r'^[0-9a-f]+ <%s>:' % re.escape(fn), line):
            in_fn = True
            continue
        if not in_fn:
            continue
        m2 = re.match(r'^[0-9a-f]+ <([A-Za-z_.$][\w.$]*)>:', line)
        if m2 and m2.group(1) != fn and not re.match(r'^(L|\.L|LM|\$L)', m2.group(1)):
            break
        rm = re.search(r'R_MIPS_(HI16|LO16)\s+(\S+)', line)
        if rm and last is not None:
            kind = "%hi" if rm.group(1) == "HI16" else "%lo"
            sym = rm.group(2)
            be, dis = out[last]
            relop = "%s(%s)" % (kind, sym)
            if "(" in dis.split(",")[-1]:            # last operand is off(base): a load/store
                dis = re.sub(r"(,\s*)-?(?:0x)?[0-9a-f]+(\()", r"\1%s\2" % relop, dis, count=1)
            else:                                    # last operand is a bare immediate
                dis = re.sub(r",\s*-?(?:0x)?[0-9a-f]+\s*$", "," + relop, dis)
            out[last] = (be, dis)
            continue
        m = re.match(r'^\s*[0-9a-f]+:\s+([0-9a-f]{8})\s+(.*)', line)
        if m:
            out.append((m.group(1).lower(), m.group(2).strip()))
            last = len(out) - 1
    return out, None

def target_words(fn):
    """splat-style target .s -> [(be_hex, disasm)] (little-endian column converted to BE).
    Expects lines of the form `/* off addr <le_bytes> */  mnem operands` between
    `glabel <fn>` and `endlabel`."""
    p = os.path.join(TARGET_DIR, fn + ".s")
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

def words_eq(t, g):
    """Compare a target word to a produced word, ignoring reloc'd immediates (the
    %hi/%lo bits are zeroed in the .o before linking)."""
    tw, tx = t
    gw, gx = g
    if tw == "--------" or gw == "--------":
        return False
    txt = tx.lower()
    if '%hi' in txt or '%lo' in txt or 'R_MIPS' in txt:
        return tw[:4] == gw[:4]
    if txt.startswith('jal') or txt.startswith('j '):
        return tw[0] == gw[0]
    return tw == gw
