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
