# GCC-Enhanced design

A target-guided matching backend for PSX-era GCC 2.x matching decompilation. This document is
the phased build plan. The motivating case study is Digimon World 2 (PSX), but the method is
general.

## Problem

Retail PSX titles of the GCC 2.x era were built with a specific, often unarchived, patched
compiler (frequently an SN Systems PsyQ cc1). A modern GCC 2.8.1 reproduces most function bodies
byte-perfect but diverges on a small recurring set of codegen decisions the C source cannot
steer:

- register naming and coloring swaps (whole-function base register choice)
- address materialization: la-form vs split form for single-access indexed globals, plus `%hi`
  CSE and rematerialization for repeated-global functions
- FILL vs UNFILL epilogues
- delay-slot fill, especially loop-entry jumps and volatile stores
- single-exit and branch-merge
- operand canonicalization, OR-tree reassociation, loop rotation, LICM

Hunting the exact retail cc1 typically fails: the whole obtainable GCC and egcs lineage can be
tested with none reproducing the private build. So the answer is to build tooling instead.

## Core insight

In a matching decomp the target bytes for every function are known. The tool does not predict
codegen blind; it finds a semantics-preserving rewriting of the near-miss output whose bytes
equal the known target. Constrained search plus verification, not compiler reproduction. Every
wall is a semantics-preserving transform with a known destination.

## Two tracks

- POST: an assembly-level solver that rewrites near-miss `.s` toward the target `.s` under an
  equivalence verifier. Fast to build, gives early matches. Legitimate only as a deterministic,
  verified, rule-general pass in the canonical toolchain, never a per-function byte hack.
- RTL: a patched GCC 2.8.1 built from source for the mipsel PSX target, with RTL passes
  instrumented to expose retail variants as compile-time knobs. Principled, clean reproducible
  build, heavier lift. POST's learned rules feed the RTL knobs.

End state: `C -> patched cc1 (+ per-function knobs) -> verified POST normalizer -> target bytes`,
deterministic, wired into the project build, still producing a byte-identical executable.

## Phase A. Divergence corpus and taxonomy

Measure before building. For every function, diff the local compiler output against the target
at instruction granularity and classify each divergence into the wall families above. A function
can carry several. Emit a machine-readable corpus and rank by match yield per transform
(body-perfect and fewest classes first). This decides the Phase C build order.

Deliverable: a classifier and a `divergence.json`. Acceptance: it reproduces the project's
hand-authored parked-function classifications.

## Phase B. Semantic-equivalence model and verifier

Build the verifier before the solver so no rewrite is trusted without a proof.

- A MIPS1 (r3000, no coprocessor 2) semantics model over a basic block: register file, memory,
  hi/lo, branch-delay and load-delay semantics, for the instruction subset the decomp uses.
- An equivalence checker: given two instruction sequences with the same entry and exit live-set,
  prove the same observable effect. Implement via a symbolic executor or an SMT encoding per
  block, stitched by control-flow graph.
- Transform legality predicates: register rename (ABI and live ranges), reorder (data,
  delay-slot and load-delay dependencies), address-form swap (same effective address), fill or
  unfill (same architectural effect including the slot instruction), exit merge (same return
  state).

Acceptance: passes on all already-matched functions and rejects a deliberately corrupted variant.

Implemented in `tools/equiv.py`. Two soundness lessons drove the final shape. First, executing a
function straight through in text order is unsound: a later write on a not-taken path overwrites a
register and hides a real difference on another path. The fix is a proper control-flow graph,
required isomorphic between the two versions, with each block pair proven equivalent from a havoc
entry state over that block's live-out set; loops need no unrolling and renamed dead temps drop
out. Second, a call must not blindly wipe caller memory (that also hides differences): post-call
state is an uninterpreted function of pre-call memory and argument registers, identical on both
sides, so a value stored before a call and read after it stays observable. Validated on a
911-function target corpus: 911 pass self-equivalence and 911 reject a corrupted variant.

## Phase C. Target-guided equivalence solver

The primary early-win engine. Operates on assembly, using the known target to prune.

Given the near-miss `.s`, the target `.s`, and the Phase A classes, apply class-specific rewrite
operators toward the target, each checked by Phase B:

- register reallocation: constraint-solve the assignment that matches the target naming
- address form: rewrite la, split and `%hi`-remat forms to match the target
- delay slot: fill or unfill each slot with the legal candidate matching the target
- exit merge: fold or split `jr` tails to match the target control-flow tail
- operand canonicalization, OR-tree, rotation: apply the documented normal forms

Because the target is known, this is guided search (A* over transform sequences with a
byte-distance heuristic), not blind permutation, so it reaches the deterministic classes a
C-source permuter cannot. Output: a recorded transform script per function, the matched `.s`,
and the verifier proof.

Build order from a typical corpus: register naming is usually the largest family, so build the
register-reallocation solver first; then delay-slot; then address-form and `%hi` CSE together
(same materialization machinery); then single-exit and epilogue-fill (control-flow tail and
peephole).

Implemented in `tools/solve.py` (with `tools/asmlib.py` as the toolchain harness). The
register-reallocation and delay-slot-fill operators are proven end to end on a 0x20-byte
struct-copy function: the solver derives the register correspondence the target implies, handling
dead-register coalescing (a lui temporary and the base pointer both fold to one target register
because the temporary is dead after the address add), applies it to the cc1 assembly, rebuilds
through the real assembler, fills the return delay slot with the preceding independent store, and
reaches byte-identical target output. The verifier then proves the original compile equivalent to
the target under that register map and the declared void exit live-set, so the match is honest
regardless of the transform path. One practical note: because the object file zeroes relocated
immediates, the equivalence gate reads a symbol-faithful disassembly (relocations folded back into
`%hi`/`%lo`) so both sides carry the same global address.

The address-form family is extended by two more operators. The split-to-la fold rewrites a
split-form global (`lui`, then `%lo(SYM)(base)` in each memory operand) into la-form (an explicit
`addiu base,base,%lo(SYM)` then `0(base)` operands) for exactly the symbols the target
materializes with an explicit la, and only when every use of the base in its live range is a
`%lo(SYM)` operand of that same symbol, so the address is preserved. The un-hi-cse operator is its
mirror: when the near-miss compiler CSEs the high part of a global (hoists one `lui B,%hi(SYM)` and
reuses `B` for each `lw D,%lo(SYM)(B)`) but the target rematerializes `lui R,%hi(SYM); lw
R,%lo(SYM)(R)` (base equal to dest) before every access, it deletes the hoisted `lui`s and re-emits
the pair at the front of each access region (a run of instructions ending in a store), using the
load's own dest register as the base. A useful observation fell out of this: the schedule
divergence people attribute to the delay-slot family (which instruction fills the `lw`-to-store
load-delay) is often a consequence of the shared hoist rather than an independent wall. The
hoisted `lui` for the next symbol was what filled the slot; once the hoists are gone and the fresh
pair sits at the region front, the right-hand-side value computation falls into the slot exactly as
the target schedules, and the assembler inserts a load-delay nop only where no independent
instruction is available. So no separate scheduler pass was needed. Legality is proven, not
assumed: the gate checks the original compile against the target, so an intervening aliasing store
that invalidated a reload fails the proof instead of faking a match. Proven end to end on a
seven-store function that writes through four global pointer slots.

The single-exit (branch-merge) family is handled by a canonicalization step in the verifier
rather than a byte rewrite alone. When a target funnels every return through one shared `jr $ra`
that the other paths branch to, while the near-miss compiler emits a `jr` per return path (common
in leaf functions with no frame), the two control-flow graphs are not isomorphic and the gate
refuses them before any proof runs. Behind a flag, the verifier folds each side's exit tail into
one canonical `[jr $ra; nop]` exit: pure `jr` blocks are merged and their predecessors redirected;
an impure `[body; jr; delay]` block is lowered to `[body then delay]` with an unconditional edge to
the shared exit, which is sound because a jump-register delay-slot instruction always executes
before control returns; and a block whose only successor is the shared exit is relabelled as an
unconditional transfer so a path that jumps to the exit and a path that falls into it compare as
the same edge. This is applied identically to both sides and the per-block proof still runs on
every block, so a real difference in a lowered body is still caught. One limitation follows from
the per-block havoc-entry method: an exit merge verifies only when the return value is computed on
corresponding blocks on both sides, which is the usual pattern (the value is preset in the shared
decision block and only the return structure differs); a form that recomputes the value on a
different block than the target does will fail the proof rather than pass falsely. The flag
defaults off and the solver enables it only when it detects the situation from the target (the
target has one return, the near-miss has several), so no same-shape function is affected and the
whole-corpus self-test is unchanged.

## Phase D. cc1 reproduction at the RTL level

Optional but principled. Build GCC 2.8.1 from source for the target, confirm baseline parity
with the current matches, then instrument the diverging passes (local and global allocation
order, scheduling and delay-branch, final and epilogue, address printing, jump and CSE) and add
knobs selecting the retail variant. Measure the match rate from patched cc1 alone. If high, it
becomes the front end and POST cleans the residue; if low, keep the current front end plus POST.

## Phase E. Per-function knob search

Expose the Phase D knobs plus existing flags as a per-function search space, auto-search the
combo that yields the target, and cache winners in a build manifest. This is the flag sweep that
works, because the knobs reach the codegen-internal decisions that plain flags never touched.

## Phase F. Integration and reproducible build

Wire the backend into the project build. The build stays deterministic with no per-function asm
hacks, only recorded knobs and verified general transforms. Provide an LLM-agent CLI that
bootstraps a function and shows the remaining divergence and the closing transform or knob.

## Phase G. Validation, rollout, regression

Re-run the full function set, measure uplift, and require that every previously matched function
still matches. Fold each generalized transform or knob into the project's codegen notes.

## Risks

- Purity: a match must be a rebuild, not an asm edit. The verifier keeps POST honest. A match
  that needs a function-specific byte hack does not count.
- Ambiguity: where multiple equivalent forms exist, the solver picks the byte-matching one,
  which is correct because the target defines the choice.
- Effort: Phase D is the heavy lift. Phases A through C deliver matches without it, so sequence A
  to B to C for momentum, then D and E for a compiler-native build.
- Some walls may be intrinsically unreachable if they need information the C cannot encode. Phase
  A quantifies how many. The success metric is match uplift, not necessarily one hundred percent.
