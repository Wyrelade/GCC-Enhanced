#!/usr/bin/env python3
"""GCC-Enhanced Phase B: MIPS1 (R3000, no GTE) basic-block semantic-equivalence verifier.

The honesty gate for the target-guided solver. Given two assembly sequences with the
same entry/exit register live-set, PROVE they have the same observable effect
(ABI-visible registers, memory, control-transfer trace) via symbolic execution + z3.

Design notes
------------
* Every integer MIPS1 instruction the decomp emits is modelled precisely over z3
  BitVec(32). Anything not modelled (cop2/GTE, unusual ops) is handled as an
  UNINTERPRETED FUNCTION of its operand values: equal inputs -> equal outputs, so the
  checker can never *prove* equivalence across a genuine semantic change, but self==self
  and legal rewrites still verify. This keeps the verifier sound (conservative): it may
  say NOT-EQUAL for a truly-equal pair it cannot reason about, but never EQUAL for a
  non-equal pair.
* Address relocations %hi(sym)/%lo(sym) are modelled so that la-form
  (lui;addiu %lo;lw 0) and split-form (lui;lw %lo) reconstruct the SAME effective
  address -> those two address materializations verify as equivalent.
* Calls (jal/jalr/bal) clobber the caller-saved set + memory opaquely but
  DETERMINISTICALLY (keyed by call index): two sequences with the same call sequence get
  the same post-call symbols, so surrounding differences still compare.
* Branch/jump instructions append a control event (condition expr + target-label index).
  The delay-slot instruction executes in architectural order (branch cond evaluated on
  pre-delay-slot state). Two sequences are equivalent iff: equal observable registers,
  equal memory, and an equal control-event trace.

Usage
-----
    equiv.py <a.s> <b.s> [--func NAME] [--live r1,r2,...] [-v]
    equiv.py --selftest <dir-of-.s>        # self==self EQUAL, self-vs-corrupt NOT-EQUAL

Exit code 0 == EQUAL / all self-tests pass; 1 == NOT-EQUAL / failure.
"""
import os, re, sys, glob, argparse
import z3

# ---------------------------------------------------------------------------
# register naming
# ---------------------------------------------------------------------------
NUM2ABI = ["zero","at","v0","v1","a0","a1","a2","a3","t0","t1","t2","t3","t4","t5",
           "t6","t7","s0","s1","s2","s3","s4","s5","s6","s7","t8","t9","k0","k1",
           "gp","sp","fp","ra"]
ABI2NUM = {n: i for i, n in enumerate(NUM2ABI)}
ABI2NUM["s8"] = 30           # fp alias
ABI2NUM["r0"] = 0

def norm_reg(tok):
    """'$a2' / '$4' / 'a2' / '$s8' -> canonical ABI name; a bare number is an
    IMMEDIATE, not a register (objdump prints small shift amounts as bare decimals
    like `sll v0,v0,8`), so only a $-prefixed number names a register."""
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

# ABI-observable-at-return live set (default). Caller-saved temps are dead at exit.
DEFAULT_LIVE = ["v0","v1","s0","s1","s2","s3","s4","s5","s6","s7","gp","sp","fp","ra"]

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
class Insn:
    __slots__ = ("mnem", "ops", "raw", "label")
    def __init__(self, mnem, ops, raw):
        self.mnem = mnem
        self.ops = ops
        self.raw = raw
        self.label = None      # label that sits immediately before this insn

class Func:
    def __init__(self, name, insns, labels):
        self.name = name
        self.insns = insns          # list[Insn]
        self.labels = labels        # ordered unique label names (for positional map)

_META = "/*"
_LABEL_RE = re.compile(r"^([A-Za-z_.$][\w.$]*):\s*$")
_GLABEL_RE = re.compile(r"^(?:glabel|dlabel|\.globl)\s+(\S+)")
_ALABEL_RE = re.compile(r"^(?:alabel|jlabel)\s+(\S+)")   # mid-function data/alt label
_ENDLABEL_RE = re.compile(r"^endlabel\b")
_COMMENT_RE = re.compile(r"/\*.*?\*/")

def _strip_meta(line):
    """Remove every /* ... */ comment (leading byte-dump AND trailing annotations)."""
    return _COMMENT_RE.sub("", line).strip()

def parse_s(path, func=None):
    """Parse a .s file. Returns a Func. If `func` is None, uses the first glabel or the
    whole file as one anonymous block."""
    text = open(path, encoding="utf-8", errors="replace").read()
    lines = text.splitlines()
    name = func
    insns, labels = [], []
    pending_label = None
    capturing = (func is None)      # if no func requested, capture from the top
    started_by_glabel = False

    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        gm = _GLABEL_RE.match(s)
        if gm:
            gname = gm.group(1)
            if func is None:
                if name is None:
                    name = gname
                capturing = True
                started_by_glabel = True
            elif gname == func:
                name = gname
                capturing = True
                started_by_glabel = True
                insns = []          # restart capture at the requested function
                labels = []
            elif capturing and started_by_glabel:
                break               # next function begins
            continue
        if _ENDLABEL_RE.match(s):
            if capturing and started_by_glabel:
                break
            continue
        if not capturing:
            continue
        body = _strip_meta(raw)
        if not body:
            continue
        am = _ALABEL_RE.match(body)
        if am:
            lab = am.group(1)
            if lab not in labels:
                labels.append(lab)
            pending_label = lab
            continue
        lm = _LABEL_RE.match(body)
        if lm:
            lab = lm.group(1)
            if lab not in labels:
                labels.append(lab)
            pending_label = lab
            continue
        if body.startswith("."):
            continue                # assembler directive
        # instruction: mnemonic then operands
        parts = body.split(None, 1)
        mnem = parts[0].lower()
        opstr = parts[1] if len(parts) > 1 else ""
        # split operands on commas at top level (offsets like 0x0($a0) stay intact)
        ops = [o.strip() for o in _split_ops(opstr)] if opstr else []
        ins = Insn(mnem, ops, body)
        ins.label = pending_label
        pending_label = None
        insns.append(ins)

    if name is None:
        name = os.path.basename(path)
    return Func(name, insns, labels)

def _split_ops(opstr):
    out, depth, cur = [], 0, ""
    for ch in opstr:
        if ch == "(":
            depth += 1; cur += ch
        elif ch == ")":
            depth -= 1; cur += ch
        elif ch == "," and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out

# ---------------------------------------------------------------------------
# symbolic machine
# ---------------------------------------------------------------------------
BV = lambda n: z3.BitVecSort(32)

class Machine:
    def __init__(self, tag):
        self.tag = tag
        self.regs = {}
        for r in NUM2ABI:
            self.regs[r] = z3.BitVecVal(0, 32) if r == "zero" \
                else z3.BitVec("%s_in" % r, 32)
        self.hi = z3.BitVec("hi_in", 32)
        self.lo = z3.BitVec("lo_in", 32)
        self.mem = z3.Array("mem_in", BV(32), z3.BitVecSort(8))
        # coprocessor register file (COP0 system, COP2/GTE), addr = cop<<8 | ctrl<<7 | reg
        self.cop = z3.Array("cop_in", BV(32), BV(32))
        self.events = []            # list of (kind, cond_expr_or_None, target_index)
        self.call_ix = 0
        self.uf_ix = 0
        self._symcache = {}         # symbol name -> z3 const (shared across A and B)

    # -- symbol constants for %hi/%lo. Shared by name so both machines agree. --
    def sym(self, name):
        # use a GLOBAL cache keyed on name so machine A and B use the SAME const
        return _global_sym(name)

    def rd(self, r):
        n = norm_reg(r)
        if n is None:
            raise ValueError("bad reg %r" % r)
        return self.regs["zero"] if n == "zero" else self.regs[n]

    def wr(self, r, val):
        n = norm_reg(r)
        if n is None:
            raise ValueError("bad reg %r" % r)
        if n != "zero":
            self.regs[n] = z3.simplify(val)

    def fresh(self, base):
        self.uf_ix += 1
        return z3.BitVec("uf_%s_%d" % (base, self.uf_ix), 32)

# --- global symbol / uninterpreted-op tables shared across both machines ---
_GSYM = {}
_UF = {}
def _global_sym(name):
    if name not in _GSYM:
        _GSYM[name] = z3.BitVec("sym_%s" % re.sub(r"\W", "_", name), 32)
    return _GSYM[name]
def _uf(key, arity):
    if key not in _UF:
        dom = [z3.BitVecSort(32)] * arity
        _UF[key] = z3.Function("op_%s_%d" % (re.sub(r"\W", "_", key), arity),
                               *dom, z3.BitVecSort(32))
    return _UF[key]

_MEMSORT = z3.ArraySort(z3.BitVecSort(32), z3.BitVecSort(8))
def _uf_callreg(key):
    """Callee result register: (mem, a0,a1,a2,a3) -> BV32."""
    k = "cr_" + key
    if k not in _UF:
        _UF[k] = z3.Function("callreg_%s" % re.sub(r"\W", "_", key),
                             _MEMSORT, *([z3.BitVecSort(32)] * 4), z3.BitVecSort(32))
    return _UF[k]
def _uf_callmem(key):
    """Callee memory effect: (mem, a0,a1,a2,a3) -> mem'. Same fn symbol for A and B,
    so equal pre-call state gives equal post-call memory; a differing pre-call store is
    still observable through the result."""
    k = "cm_" + key
    if k not in _UF:
        _UF[k] = z3.Function("callmem_%s" % re.sub(r"\W", "_", key),
                             _MEMSORT, *([z3.BitVecSort(32)] * 4), _MEMSORT)
    return _UF[k]
def reset_globals():
    _GSYM.clear(); _UF.clear()

# ---------------------------------------------------------------------------
# operand evaluation
# ---------------------------------------------------------------------------
def _reloc(tok):
    """(%hi|%lo)(sym) -> ('hi'|'lo', sym) else None."""
    m = re.fullmatch(r"%(hi|lo)\((.+)\)", tok.strip())
    return (m.group(1), m.group(2)) if m else None

def imm_val(tok, m):
    """Evaluate an immediate / reloc operand to a BV32."""
    tok = tok.strip()
    rl = _reloc(tok)
    if rl:
        kind, name = rl
        s = m.sym(name)
        if kind == "hi":
            # standard: %hi = (sym + 0x8000) >> 16, later shifted <<16 by lui
            return z3.LShR(s + z3.BitVecVal(0x8000, 32), 16)
        else:
            # %lo = sign-extended low 16 bits
            lo16 = z3.Extract(15, 0, s)
            return z3.SignExt(16, lo16)
    # numeric
    neg = tok.startswith("-")
    t = tok[1:] if neg else tok
    try:
        v = int(t, 16) if t.lower().startswith("0x") else int(t, 0)
    except ValueError:
        # splat writes lui/ori/andi immediates as constant C-expressions, e.g.
        # `(0xFFFFFF >> 16)` / `(0xFF000000 & 0xFFFF)`. Evaluate them to a concrete
        # value so they agree with objdump's already-folded immediate on the other
        # side; anything containing a real symbol stays an uninterpreted const.
        ce = _const_expr(tok)
        if ce is not None:
            return z3.BitVecVal(ce & 0xffffffff, 32)
        # bare symbol name used as an immediate -> treat as its symbol const
        return m.sym(tok)
    if neg:
        v = -v
    return z3.BitVecVal(v & 0xffffffff, 32)

# A constant integer expression as emitted by splat for split immediates:
# only integer literals and the C bit/arith operators, no identifiers.
_CONST_EXPR_CHARS = re.compile(r"[\s0-9a-fA-FxX()<>&|^+\-*~]+")
def _const_expr(tok):
    s = tok.strip()
    if not s or not _CONST_EXPR_CHARS.fullmatch(s):
        return None
    # every maximal alnum run must be a valid integer literal (guards a stray symbol)
    for run in re.findall(r"[0-9a-fA-FxX]+", s):
        try:
            int(run, 0)
        except ValueError:
            return None
    try:
        return int(eval(s, {"__builtins__": {}}, {}))
    except Exception:
        return None

def mem_addr(tok, m):
    """'0x4($a2)' or '($a2)' or 'sym' -> BV32 effective address, or None."""
    tok = tok.strip()
    mm = re.fullmatch(r"(.*)\((\$?\w+)\)", tok)
    if mm:
        off = mm.group(1).strip()
        base = m.rd(mm.group(2))
        if off == "" or off == "0":
            return base
        return z3.simplify(base + imm_val(off, m))
    # absolute (rare): a bare symbol
    rl = _reloc(tok)
    if rl:
        return imm_val(tok, m)
    return None

# byte-addressed little-endian memory helpers
def load(m, addr, nbytes, signed):
    bs = [z3.Select(m.mem, z3.simplify(addr + z3.BitVecVal(i, 32))) for i in range(nbytes)]
    val = bs[0]
    for i in range(1, nbytes):          # little-endian: byte0 = LSB
        val = z3.Concat(bs[i], val)
    if nbytes < 4:
        val = z3.SignExt(32 - 8*nbytes, val) if signed else z3.ZeroExt(32 - 8*nbytes, val)
    return z3.simplify(val)

def store(m, addr, val, nbytes):
    for i in range(nbytes):
        byte = z3.Extract(8*i + 7, 8*i, val)
        m.mem = z3.Store(m.mem, z3.simplify(addr + z3.BitVecVal(i, 32)), byte)

# ---------------------------------------------------------------------------
# instruction execution
# ---------------------------------------------------------------------------
_ALU3 = {
    "add": lambda a, b: a + b, "addu": lambda a, b: a + b,
    "sub": lambda a, b: a - b, "subu": lambda a, b: a - b,
    "and": lambda a, b: a & b, "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b, "nor": lambda a, b: ~(a | b),
    "slt": lambda a, b: z3.If(a < b, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)),
    "sltu": lambda a, b: z3.If(z3.ULT(a, b), z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)),
    "sllv": lambda a, b: a << (b & 31), "srlv": lambda a, b: z3.LShR(a, b & 31),
    "srav": lambda a, b: a >> (b & 31),
}
_ALUI = {
    "addi": lambda a, i: a + i, "addiu": lambda a, i: a + i,
    "andi": lambda a, i: a & i, "ori": lambda a, i: a | i, "xori": lambda a, i: a ^ i,
    "slti": lambda a, i: z3.If(a < i, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)),
    "sltiu": lambda a, i: z3.If(z3.ULT(a, i), z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)),
}
_SHIFT = {"sll": lambda a, s: a << s, "srl": lambda a, s: z3.LShR(a, s),
          "sra": lambda a, s: a >> s}
_LOAD = {"lw": (4, True), "lh": (2, True), "lhu": (2, False),
         "lb": (1, True), "lbu": (1, False)}
_STORE = {"sw": 4, "sh": 2, "sb": 1}
_BR2 = {"beq": lambda a, b: a == b, "bne": lambda a, b: a != b}
_BR1 = {
    "blez": lambda a: a <= 0, "bgtz": lambda a: a > 0,
    "bltz": lambda a: a < 0, "bgez": lambda a: a >= 0,
    "beqz": lambda a: a == 0, "bnez": lambda a: a != 0,
    "bltzal": lambda a: a < 0, "bgezal": lambda a: a >= 0,
}
# caller-clobbered set on a call
_CALLER_SAVED = ["at","v0","v1","a0","a1","a2","a3","t0","t1","t2","t3","t4","t5",
                 "t6","t7","t8","t9","ra"]

def label_index(func, target):
    t = target.strip()
    if t in func.labels:
        return func.labels.index(t)
    return ("addr", t)          # fall back to raw string when not a known label

def step(m, ins, func):
    op = ins.mnem
    o = ins.ops
    # normalise a few aliases
    if op == "move" and len(o) == 2:
        m.wr(o[0], m.rd(o[1])); return
    if op in ("nop", "b_nop"):
        return
    if op == "li" and len(o) == 2:
        m.wr(o[0], imm_val(o[1], m)); return
    if op == "la" and len(o) == 2:
        rl = _reloc(o[1])
        m.wr(o[0], m.sym(o[1]) if rl is None else imm_val(o[1], m)); return
    if op == "lui" and len(o) == 2:
        m.wr(o[0], z3.simplify(imm_val(o[1], m) << 16)); return
    if op == "negu" and len(o) == 2:
        m.wr(o[0], z3.simplify(-m.rd(o[1]))); return
    if op == "not" and len(o) == 2:
        m.wr(o[0], z3.simplify(~m.rd(o[1]))); return

    if op in _ALU3 and len(o) == 3:
        m.wr(o[0], z3.simplify(_ALU3[op](m.rd(o[1]), m.rd(o[2])))); return
    if op in _ALUI and len(o) == 3:
        m.wr(o[0], z3.simplify(_ALUI[op](m.rd(o[1]), imm_val(o[2], m)))); return
    if op in _SHIFT and len(o) == 3:
        sa = imm_val(o[2], m)
        m.wr(o[0], z3.simplify(_SHIFT[op](m.rd(o[1]), sa))); return

    if op in _LOAD:
        nb, signed = _LOAD[op]
        a = mem_addr(o[1], m)
        m.wr(o[0], load(m, a, nb, signed)); return
    if op in _STORE:
        nb = _STORE[op]
        a = mem_addr(o[1], m)
        store(m, a, m.rd(o[0]), nb); return

    if op in ("mult", "multu") and len(o) == 2:
        ext = z3.SignExt if op == "mult" else z3.ZeroExt
        prod = ext(32, m.rd(o[0])) * ext(32, m.rd(o[1]))
        m.lo = z3.simplify(z3.Extract(31, 0, prod))
        m.hi = z3.simplify(z3.Extract(63, 32, prod)); return
    if op in ("div", "divu") and len(o) == 2:
        a, b = m.rd(o[0]), m.rd(o[1])
        if op == "div":
            q, r = a / b, z3.SRem(a, b)
        else:
            q, r = z3.UDiv(a, b), z3.URem(a, b)
        # guard div-by-zero: undefined on MIPS -> uninterpreted but deterministic
        z = z3.BitVecVal(0, 32)
        m.lo = z3.simplify(z3.If(b == z, _uf("div0lo", 2)(a, b), q))
        m.hi = z3.simplify(z3.If(b == z, _uf("div0hi", 2)(a, b), r)); return
    if op == "mfhi" and len(o) == 1:
        m.wr(o[0], m.hi); return
    if op == "mflo" and len(o) == 1:
        m.wr(o[0], m.lo); return
    if op == "mthi" and len(o) == 1:
        m.hi = m.rd(o[0]); return
    if op == "mtlo" and len(o) == 1:
        m.lo = m.rd(o[0]); return

    # ---- control transfer ----
    if op in _BR2 and len(o) == 3:
        cond = _BR2[op](m.rd(o[0]), m.rd(o[1]))
        m.events.append(("branch", z3.simplify(cond), label_index(func, o[2]))); return
    if op in _BR1 and len(o) == 2:
        cond = _BR1[op](m.rd(o[0]))
        m.events.append(("branch", z3.simplify(cond), label_index(func, o[1]))); return
    if op in ("b", "j") and len(o) == 1:
        m.events.append(("jump", None, label_index(func, o[0]))); return
    if op == "jr" and len(o) == 1:
        m.events.append(("return" if norm_reg(o[0]) == "ra" else "jr",
                         None, ("reg", norm_reg(o[0])))); return
    if op in ("jal", "bal") and len(o) >= 1:
        m.call_ix += 1
        tgt = o[-1]
        m.events.append(("call", None, tgt))
        _clobber_call(m); return
    if op == "jalr":
        m.call_ix += 1
        m.events.append(("callr", None, ("reg", norm_reg(o[-1]))))
        _clobber_call(m); return
    if op in ("syscall", "break"):
        m.call_ix += 1
        m.events.append(("trap", None, op))
        _clobber_call(m); return

    # ---- coprocessor moves (COP0 system regs, COP2/GTE) ----
    cm = re.fullmatch(r"([cm])([tf])c([0-9])", op)
    if cm and len(o) == 2:
        ctrl = 1 if cm.group(1) == "c" else 0        # c=control reg, m=data reg
        direction = cm.group(2)                       # t=to-cop, f=from-cop
        copn = int(cm.group(3))
        num = _cop_num(o[1])
        idx = z3.BitVecVal((copn << 8) | (ctrl << 7) | (num & 0x7f), 32)
        if direction == "t":                          # mtcN/ctcN: GPR -> cop reg
            m.cop = z3.Store(m.cop, idx, m.rd(o[0]))
        else:                                         # mfcN/cfcN: cop reg -> GPR
            m.wr(o[0], z3.Select(m.cop, idx))
        return

    # ---- unknown / unmodelled: uninterpreted function of operand regs ----
    _opaque(m, ins)

def _cop_num(tok):
    t = tok.strip().lstrip("$")
    mm = re.match(r"(\d+)", t)
    return int(mm.group(1)) if mm else 0

def _clobber_call(m):
    """Model a call as an opaque-but-deterministic transformer of (memory, arg regs).
    Caller's memory is NOT blindly wiped: post-call memory is an uninterpreted function
    of pre-call memory + args, so a value stored before the call and read after it is
    still distinguishable between two sequences (sound), while identical call sequences
    on identical state agree."""
    ci = m.call_ix
    pre_mem = m.mem
    args = [m.regs[r] for r in ("a0", "a1", "a2", "a3")]
    for r in _CALLER_SAVED:
        if r == "ra":
            continue                    # ra := return address (not a data result)
        m.regs[r] = _uf_callreg("%d_%s" % (ci, r))(pre_mem, *args)
    m.regs["ra"] = z3.BitVec("post_call%d_ra" % ci, 32)
    m.hi = _uf_callreg("%d_hi" % ci)(pre_mem, *args)
    m.lo = _uf_callreg("%d_lo" % ci)(pre_mem, *args)
    m.mem = _uf_callmem("%d" % ci)(pre_mem, *args)

def _opaque(m, ins):
    """Unmodelled instruction: if it writes a GPR (op rd, rs, ...), set rd to a
    deterministic uninterpreted function of the source operand values so identical
    sequences agree and corruptions differ."""
    o = ins.ops
    src_vals = []
    for tok in o[1:]:
        n = norm_reg(tok)
        if n is not None:
            src_vals.append(m.rd(tok))
        else:
            a = mem_addr(tok, m)
            src_vals.append(a if a is not None else imm_val(tok, m))
    if o and norm_reg(o[0]) is not None:
        f = _uf(ins.mnem, max(1, len(src_vals)))
        args = src_vals if src_vals else [m.rd(o[0])]
        m.wr(o[0], f(*args))

# ---------------------------------------------------------------------------
# control-flow graph + liveness (sound, path-sensitive; no loop unrolling)
# ---------------------------------------------------------------------------
_COND_BR = set(_BR2) | set(_BR1)
_UNCOND = {"j", "b"}
ALL_GPRS = [r for r in NUM2ABI if r != "zero"]

def _is_term(mnem):
    return mnem in _COND_BR or mnem in _UNCOND or mnem == "jr"

class Block:
    __slots__ = ("insns", "label", "term", "delay", "kind", "targets")
    def __init__(self, insns, label):
        self.insns = insns
        self.label = label
        self.term = None      # terminator insn (branch/j/jr) or None (fallthrough)
        self.delay = None     # delay-slot insn or None
        self.kind = "fall"    # 'cond' | 'uncond' | 'return' | 'jr' | 'fall'
        self.targets = []     # target label strings (branch/jump)
        self._classify()
    def _classify(self):
        for ins in self.insns:
            if _is_term(ins.mnem):
                self.term = ins
        if self.term is None:
            self.kind = "fall"; return
        ti = self.insns.index(self.term)
        if ti + 1 < len(self.insns):
            self.delay = self.insns[ti + 1]
        m = self.term.mnem
        if m in _COND_BR:
            self.kind = "cond"; self.targets = [self.term.ops[-1]]
        elif m in _UNCOND:
            self.kind = "uncond"; self.targets = [self.term.ops[-1]]
        elif m == "jr":
            self.kind = "return" if norm_reg(self.term.ops[0]) == "ra" else "jr"

def build_blocks(func):
    insns = func.insns
    n = len(insns)
    blocks = []
    i = 0
    cur, cur_label = [], (insns[0].label if insns else None)
    while i < n:
        ins = insns[i]
        if ins.label and cur:
            blocks.append(Block(cur, cur_label))
            cur, cur_label = [], ins.label
        cur.append(ins)
        if _is_term(ins.mnem):
            if i + 1 < n:                 # pull in the delay-slot instruction
                cur.append(insns[i + 1]); i += 1
            blocks.append(Block(cur, cur_label))
            nxt = i + 1
            cur, cur_label = [], (insns[nxt].label if nxt < n else None)
        i += 1
    if cur:
        blocks.append(Block(cur, cur_label))
    return blocks

def _succ(blocks):
    lab2idx = {}
    for idx, b in enumerate(blocks):
        if b.label and b.label not in lab2idx:
            lab2idx[b.label] = idx
    succ = []
    for idx, b in enumerate(blocks):
        s = []
        if b.kind == "cond":
            t = b.targets[0]
            if t in lab2idx: s.append(lab2idx[t])
            if idx + 1 < len(blocks): s.append(idx + 1)   # fallthrough
        elif b.kind == "uncond":
            t = b.targets[0]
            if t in lab2idx: s.append(lab2idx[t])
        elif b.kind in ("return", "jr"):
            s = []                                        # exit
        else:  # fall
            if idx + 1 < len(blocks): s.append(idx + 1)
        succ.append(s)
    return succ, lab2idx

# --- per-instruction def/use of GPRs (+ hi/lo pseudo) ---
_WRITES0 = (set(_ALU3) | set(_ALUI) | set(_SHIFT) | set(_LOAD) |
            {"lui", "li", "la", "move", "negu", "not", "mfhi", "mflo"})
_CALL_DEFS = set(_CALLER_SAVED) | {"hi", "lo"}

def defs_uses(ins):
    op, o = ins.mnem, ins.ops
    defs, uses = set(), set()
    def add_use(tok):
        r = norm_reg(tok)
        if r and r != "zero": uses.add(r)
        mm = re.fullmatch(r"(.*)\((\$?\w+)\)", tok.strip())
        if mm:
            r2 = norm_reg(mm.group(2))
            if r2 and r2 != "zero": uses.add(r2)
    if op in ("jal", "bal", "jalr", "bltzal", "bgezal"):
        defs |= _CALL_DEFS
        uses |= {"a0", "a1", "a2", "a3"}
        if op == "jalr" and o: add_use(o[-1])
        return defs, uses
    if op in ("mult", "multu", "div", "divu"):
        for t in o: add_use(t)
        defs |= {"hi", "lo"}; return defs, uses
    if op in ("mthi",):
        add_use(o[0]); defs.add("hi"); return defs, uses
    if op in ("mtlo",):
        add_use(o[0]); defs.add("lo"); return defs, uses
    if op in ("mfhi",):
        d = norm_reg(o[0]);
        if d and d != "zero": defs.add(d)
        uses.add("hi"); return defs, uses
    if op in ("mflo",):
        d = norm_reg(o[0])
        if d and d != "zero": defs.add(d)
        uses.add("lo"); return defs, uses
    if op in _STORE or op in ("swc2", "swl", "swr"):
        for t in o: add_use(t)      # source reg + base reg both used
        return defs, uses
    if op in _COND_BR:
        for t in o[:-1]: add_use(t)
        return defs, uses
    if op in _UNCOND:
        return defs, uses
    if op == "jr":
        add_use(o[0]); return defs, uses
    cm = re.fullmatch(r"[cm]tc[0-9]", op)      # move-to-coprocessor
    if cm:
        if o: add_use(o[0])
        return defs, uses
    cf = re.fullmatch(r"[cm]fc[0-9]", op)      # move-from-coprocessor
    if cf:
        d = norm_reg(o[0])
        if d and d != "zero": defs.add(d)
        return defs, uses
    # op rd, rs, rt...  -> rd is def, rest uses (ALU/shift/load/lui/li/la/move/negu/not)
    if op in _WRITES0 and o:
        d = norm_reg(o[0])
        if d and d != "zero":
            defs.add(d)
        for t in o[1:]:
            add_use(t)
        return defs, uses
    # unknown/opaque op: matches _opaque (writes o[0] from o[1:]); treat o[0] as def+use
    if o and norm_reg(o[0]) is not None:
        d = norm_reg(o[0])
        if d != "zero":
            defs.add(d)
        for t in o[1:]:
            add_use(t)
        return defs, uses
    for t in o:
        add_use(t)
    return defs, uses

def liveness(blocks, succ, exit_live=None):
    """exit_live: reg-name set live at function returns (jr $ra). Defaults to the ABI
    observable set. Declared by the caller because it depends on the C return type
    (a void function leaves v0/v1 dead)."""
    if exit_live is None:
        exit_live = DEFAULT_LIVE
    n = len(blocks)
    live_in = [set() for _ in range(n)]
    live_out = [set() for _ in range(n)]
    exitset = {}
    for idx, b in enumerate(blocks):
        if not succ[idx]:
            exitset[idx] = set(exit_live) if b.kind == "return" else set(ALL_GPRS)
    changed = True
    while changed:
        changed = False
        for idx in range(n - 1, -1, -1):
            lo = set(exitset.get(idx, set()))
            for s in succ[idx]:
                lo |= live_in[s]
            li = set(lo)
            for ins in reversed(blocks[idx].insns):
                d, u = defs_uses(ins)
                li -= d
                li |= u
            if li != live_in[idx] or lo != live_out[idx]:
                live_in[idx] = li; live_out[idx] = lo; changed = True
    return live_out

# ---------------------------------------------------------------------------
# equivalence check
# ---------------------------------------------------------------------------
class Result:
    def __init__(self, equal, reason=""):
        self.equal = equal
        self.reason = reason
    def __bool__(self):
        return self.equal

def _prove_eq(ea, eb):
    """Return True iff ea == eb for ALL inputs (unsat of ea != eb)."""
    s = z3.Solver()
    s.set("timeout", 20000)
    s.add(ea != eb)
    r = s.check()
    return r == z3.unsat, r

class _CBlock:
    """A canonicalized block: exposes only .insns and .kind, the attributes the
    equivalence proof and liveness read."""
    __slots__ = ("insns", "kind")
    def __init__(self, insns, kind):
        self.insns = insns
        self.kind = kind

def _is_nop_insn(ins):
    if ins.mnem == "nop":
        return True
    if ins.mnem == "sll" and ins.ops and norm_reg(ins.ops[0]) == "zero":
        return True
    return False

def canonicalize_exits(blocks, succ):
    """Fold a multi-exit tail into ONE shared `jr $ra` exit so two functions that differ
    only in exit structure (a `jr` per return path vs a single shared `jr` the other paths
    branch to) compare as isomorphic. Applied identically to BOTH sides.

    Semantics-preserving per side: a `jr $ra` delay-slot instruction ALWAYS executes before
    control returns, so lowering it into the block body ahead of an unconditional transfer to
    a canonical `[jr $ra; nop]` exit keeps the architectural effect; and merging pure
    `[jr $ra; nop]` blocks preserves return-to-caller semantics. The per-block havoc-entry
    proof still runs on every block, so a genuine difference in a lowered body or delay
    instruction is still caught -- canonicalization only removes a spurious STRUCTURAL
    mismatch, it never hides a semantic one. Returns (new_blocks, new_succ).

    LIMITATION: the per-block proof runs each block from a havoc entry and compares block-local
    live-out, so an exit-merge verifies only when the return value is computed on CORRESPONDING
    blocks on both sides (the common pattern: the value is preset in the shared decision block
    and only the `jr` structure differs). A form that recomputes the value on a different block
    than the other side will fail the proof rather than pass falsely."""
    ret_idxs = [i for i, b in enumerate(blocks) if b.kind == "return"]
    if not ret_idxs:
        return blocks, succ
    pure = set()
    lowered = {}
    for i in ret_idxs:
        insns = blocks[i].insns
        ti = None
        for k, ins in enumerate(insns):
            if ins.mnem == "jr" and ins.ops and norm_reg(ins.ops[0]) == "ra":
                ti = k
        if ti is None:
            continue
        body = insns[:ti]
        delay = insns[ti + 1] if ti + 1 < len(insns) else None
        if not body and (delay is None or _is_nop_insn(delay)):
            pure.add(i)
        else:
            low = list(body)
            if delay is not None and not _is_nop_insn(delay):
                low.append(delay)
            lowered[i] = low
    jr = Insn("jr", ["$ra"], "jr $ra")
    nop = Insn("nop", [], "nop")
    idx_map = {}
    new_blocks = []
    for i, b in enumerate(blocks):
        if i in pure:
            continue
        idx_map[i] = len(new_blocks)
        if i in lowered:
            new_blocks.append(_CBlock(lowered[i], "uncond"))
        else:
            new_blocks.append(b)
    exit_idx = len(new_blocks)
    new_blocks.append(_CBlock([jr, nop], "return"))
    new_succ = []
    for i, b in enumerate(blocks):
        if i in pure:
            continue
        if i in lowered:
            new_succ.append([exit_idx])
            continue
        ns = []
        for t in succ[i]:
            if t in pure:
                ns.append(exit_idx)
            elif t in idx_map:
                ns.append(idx_map[t])
        new_succ.append(ns)
    new_succ.append([])
    for k in range(len(new_blocks)):
        if new_succ[k] == [exit_idx] and new_blocks[k].kind in ("fall", "uncond"):
            b = new_blocks[k]
            if not isinstance(b, _CBlock):
                b = _CBlock(list(b.insns), b.kind)
                new_blocks[k] = b
            b.kind = "uncond"
    return new_blocks, new_succ

def check_equiv(fa, fb, live=None, regmap=None, verbose=False, canon_exits=False):
    """Prove fa and fb compute the same observable effect. `regmap` (our-reg -> other-reg)
    lets the caller declare a register renaming; default identity. Sound method:
    require isomorphic CFGs, then prove each block pair equivalent from a havoc entry
    state over that block's live-out set (+ memory + coprocessor + branch condition +
    successor structure). Loops need no unrolling; renamed dead temps are excluded.

    `canon_exits`: when True, fold each side's multi-exit tail into one shared `jr $ra`
    exit before the isomorphism check (see canonicalize_exits), so functions equivalent
    modulo single-exit/branch-merge structure verify. Default False."""
    reset_globals()
    if regmap is None:
        regmap = {}
    ba = build_blocks(fa)
    bb = build_blocks(fb)
    sa, _ = _succ(ba)
    sb, _ = _succ(bb)
    if canon_exits:
        ba, sa = canonicalize_exits(ba, sa)
        bb, sb = canonicalize_exits(bb, sb)
    if len(ba) != len(bb):
        return Result(False, "CFG block count %d != %d" % (len(ba), len(bb)))
    for i in range(len(ba)):
        if ba[i].kind != bb[i].kind:
            return Result(False, "block %d kind %s != %s" % (i, ba[i].kind, bb[i].kind))
        if sa[i] != sb[i]:
            return Result(False, "block %d successors %s != %s" % (i, sa[i], sb[i]))
    lo_a = liveness(ba, sa, exit_live=live)

    for i in range(len(ba)):
        ma = Machine("A")
        for ins in ba[i].insns:
            step(ma, ins, fa)
        mb = Machine("B")
        for ins in bb[i].insns:
            step(mb, ins, fb)

        # branch condition (conditional blocks) + call-event trace
        ea = [e for e in ma.events if e[0] == "branch"]
        eb = [e for e in mb.events if e[0] == "branch"]
        if len(ea) != len(eb):
            return Result(False, "block %d branch-event count differs" % i)
        for (ka, ca, _), (kb, cb, _) in zip(ea, eb):
            ok, r = _prove_eq(z3.If(ca, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)),
                              z3.If(cb, z3.BitVecVal(1, 32), z3.BitVecVal(0, 32)))
            if not ok:
                return Result(False, "block %d branch condition differs (%s)" % (i, r))
        # call targets (order + symbol) must match
        cta = [e[2] for e in ma.events if e[0] in ("call", "callr", "trap")]
        ctb = [e[2] for e in mb.events if e[0] in ("call", "callr", "trap")]
        if cta != ctb:
            return Result(False, "block %d call trace %r != %r" % (i, cta, ctb))

        # live-out registers (mapped through regmap)
        for r in lo_a[i]:
            if r in ("hi", "lo"):
                va = ma.hi if r == "hi" else ma.lo
                vb = mb.hi if r == "hi" else mb.lo
            else:
                va = ma.regs[r]
                vb = mb.regs[regmap.get(r, r)]
            ok, res = _prove_eq(va, vb)
            if not ok:
                return Result(False, "block %d live-out $%s differs (%s)" % (i, r, res))
        # memory + coprocessor (always observable)
        x = z3.BitVec("addr_probe", 32)
        ok, res = _prove_eq(z3.Select(ma.mem, x), z3.Select(mb.mem, x))
        if not ok:
            return Result(False, "block %d memory differs (%s)" % (i, res))
        c = z3.BitVec("cop_probe", 32)
        ok, res = _prove_eq(z3.Select(ma.cop, c), z3.Select(mb.cop, c))
        if not ok:
            return Result(False, "block %d coprocessor state differs (%s)" % (i, res))

    return Result(True, "equivalent (%d blocks, live-out+mem+cop+control per block)" % len(ba))

# ---------------------------------------------------------------------------
# self-test: self==self EQUAL, self-vs-corrupt NOT-EQUAL
# ---------------------------------------------------------------------------
_SWAP = {"addu": "subu", "subu": "addu", "and": "or", "or": "and",
         "sw": "sb", "sh": "sb", "add": "sub", "xor": "and", "nor": "and",
         "beq": "bne", "bne": "beq", "sll": "srl", "srl": "sll", "sra": "sll",
         "slt": "sltu", "mult": "multu", "lw": "lb", "lhu": "lbu"}

_STORE_OPS = {"sw", "sh", "sb", "swc2", "swl", "swr"}
_COP_TO = re.compile(r"[cm]tc[0-9]")

def _clone(func):
    nf = Func(func.name, [Insn(i.mnem, list(i.ops), i.raw) for i in func.insns],
              list(func.labels))
    for k, i in enumerate(nf.insns):
        i.label = func.insns[k].label
    return nf

def _corrupt_variants(func):
    """Yield (new_func, desc) for each single-instruction mutation. A mutation on a dead
    value is genuinely equivalent, so the reject-test passes if AT LEAST ONE variant is
    detected NOT-EQUAL (proving the verifier can catch a real change). Multiple mutation
    kinds are tried so functions whose only swap-eligible ops are `move`s (addu $x,$y,$0)
    or whose only effect is a store/coprocessor write are still exercised."""
    # (a) opcode swaps
    for idx, ins in enumerate(func.insns):
        if ins.mnem in _SWAP:
            nf = _clone(func)
            nf.insns[idx].mnem = _SWAP[ins.mnem]
            yield nf, "%s->%s @%d" % (ins.mnem, _SWAP[ins.mnem], idx)
    # (b) bump a store / cop-write offset or source register (always observable)
    alt = {"v0": "v1", "v1": "v0", "a0": "a1", "a1": "a0", "a2": "a3", "a3": "a2",
           "t0": "t1", "t1": "t0", "s0": "s1", "s1": "s0", "zero": "at"}
    for idx, ins in enumerate(func.insns):
        if ins.mnem in _STORE_OPS or _COP_TO.fullmatch(ins.mnem):
            src = norm_reg(ins.ops[0]) if ins.ops else None
            if src and src in alt:
                nf = _clone(func)
                nf.insns[idx].ops[0] = "$" + alt[src]
                yield nf, "%s src %s->%s @%d" % (ins.mnem, src, alt[src], idx)
        if ins.mnem in _STORE_OPS and len(ins.ops) == 2:
            mm = re.fullmatch(r"(.*)\((\$?\w+)\)", ins.ops[1].strip())
            if mm:
                off = mm.group(1).strip() or "0"
                try:
                    v = int(off, 16) if off.lower().startswith("0x") else int(off, 0)
                except ValueError:
                    continue
                nf = _clone(func)
                nf.insns[idx].ops[1] = "0x%X(%s)" % (v + 0x40, mm.group(2))
                yield nf, "%s off +0x40 @%d" % (ins.mnem, idx)
    # (c) substitute a source register of a value-producing op (catches return-only
    #     functions whose only ops are register moves: addu $vN,$rs,$zero)
    for idx, ins in enumerate(func.insns):
        if ins.mnem in _WRITES0 and len(ins.ops) >= 2:
            for k in range(1, len(ins.ops)):
                src = norm_reg(ins.ops[k])
                if src and src in alt and alt[src] != norm_reg(ins.ops[0]):
                    nf = _clone(func)
                    nf.insns[idx].ops[k] = "$" + alt[src]
                    yield nf, "%s src%d %s->%s @%d" % (ins.mnem, k, src, alt[src], idx)

def selftest(directory, limit=None, verbose=False):
    files = sorted(glob.glob(os.path.join(directory, "*.s")))
    if limit:
        files = files[:limit]
    n_ok = n_self_fail = n_corrupt_fail = n_skip = 0
    fails = []
    for i, path in enumerate(files):
        try:
            fa = parse_s(path)
            fb = parse_s(path)
        except Exception as e:
            n_skip += 1
            continue
        if not fa.insns:
            n_skip += 1
            continue
        try:
            r_self = check_equiv(fa, fb)
        except Exception as e:
            n_self_fail += 1
            fails.append((os.path.basename(path), "self-exc: %s" % e))
            continue
        if not r_self.equal:
            n_self_fail += 1
            fails.append((os.path.basename(path), "SELF NOT EQUAL: " + r_self.reason))
            continue
        detected = False
        had_variant = False
        try:
            for cf, desc in _corrupt_variants(fa):
                had_variant = True
                if not check_equiv(fa, cf).equal:
                    detected = True
                    break
        except Exception as e:
            n_corrupt_fail += 1
            fails.append((os.path.basename(path), "corrupt-exc: %s" % e))
            continue
        if had_variant and not detected:
            n_corrupt_fail += 1
            fails.append((os.path.basename(path), "NO corruption detected (all variants equal)"))
            continue
        n_ok += 1
        if verbose and (i % 100 == 0):
            print("  ... %d/%d" % (i, len(files)))
    print("selftest: %d files | ok=%d self-fail=%d corrupt-fail=%d skip=%d"
          % (len(files), n_ok, n_self_fail, n_corrupt_fail, n_skip))
    for name, why in fails[:40]:
        print("  FAIL %s: %s" % (name, why))
    return n_self_fail == 0 and n_corrupt_fail == 0

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", nargs="?")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--func", default=None)
    ap.add_argument("--live", default=None, help="comma reg list overriding ABI default")
    ap.add_argument("--selftest", metavar="DIR")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest(args.selftest, args.limit, args.verbose)
        sys.exit(0 if ok else 1)

    if not args.a or not args.b:
        ap.error("need <a.s> <b.s>  (or --selftest DIR)")
    live = args.live.split(",") if args.live else None
    fa = parse_s(args.a, args.func)
    fb = parse_s(args.b, args.func)
    r = check_equiv(fa, fb, live=live, verbose=args.verbose)
    print(("EQUAL: " if r.equal else "NOT-EQUAL: ") + r.reason)
    sys.exit(0 if r.equal else 1)

if __name__ == "__main__":
    main()
