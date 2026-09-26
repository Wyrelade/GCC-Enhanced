# GCC-Enhanced

A target-guided matching backend for PSX-era GCC 2.x matching decompilation projects.

## What problem it solves

In a matching decompilation you rewrite a game in C so that the original toolchain recompiles it
into byte-identical assembly. The hard blocker on old PSX titles is the compiler: retail was
built with a specific, often unarchived, GCC 2.x build (frequently an SN Systems PsyQ cc1).
A modern GCC 2.8.1 reproduces most function bodies exactly but diverges on a small, recurring set
of codegen decisions that the C source cannot steer:

- register naming and coloring swaps (a base register lands in v0 instead of v1 across a whole
  function)
- address materialization: la-form (`lui;addiu;addu;lw 0`) versus the split form
  (`lui;addu;lw %lo`) for single-access indexed globals, plus `%hi` CSE and rematerialization
- FILL vs UNFILL epilogues (`lw ra; addiu sp; jr; nop` vs `lw ra; nop; jr; addiu sp`)
- delay-slot fill, especially loop-entry jumps and volatile stores
- single-exit and branch-merge (retail routes returns through one shared `jr`)
- operand canonicalization, OR-tree reassociation, loop rotation, LICM

Hunting the exact retail cc1 binary usually fails: the whole public GCC and egcs lineage can be
tested and none reproduce a private patched build.

## The core insight

In a matching decomp you already have the target bytes for every function. So the tool does not
have to predict retail codegen blind like a compiler. It has to find a semantics-preserving
rewriting of the near-miss output whose bytes equal the known target. That is a constrained
search plus verification problem, which is far more tractable than reproducing a compiler.

Every wall above is a semantics-preserving transform: rename registers, reschedule within
dependencies, choose a different address form, fill or unfill a delay slot, merge exits. The
destination is known; the tool finds the transform path and proves it is equivalent.

## Architecture

Two complementary tracks.

- POST (assembly-level solver). Takes the near-miss `.s` and the target `.s`, then searches
  semantics-preserving asm rewrites until the bytes match, gated by an equivalence verifier.
  Fast to build and iterate. Legitimate only as a deterministic, verified, rule-general compiler
  pass, never a per-function byte hack. The verifier is what keeps it honest.
- RTL (patched cc1). Rebuilds GCC 2.8.1 from source for the mipsel PSX target and patches the
  RTL passes (register allocation order, scheduling, delay-branch, final and epilogue, address
  printing) to expose retail variants as compile-time knobs. Principled, yields a clean
  reproducible build. Heavier. The POST track's learned rules feed the RTL knobs.

End state: `C -> patched cc1 (+ per-function knobs) -> verified POST normalizer -> target bytes`,
deterministic and reproducible, wired into the project build.

## Phases

See [docs/DESIGN.md](docs/DESIGN.md) for the full phased plan.

- A. Divergence corpus and taxonomy. Classify every function's divergence into the wall
  families; rank by match yield per transform. Decides build order.
- B. Semantic-equivalence model and verifier. A MIPS1 block-level equivalence checker. Built
  before the solver so no rewrite is ever trusted without a proof.
- C. Target-guided equivalence solver. The register, address, delay-slot, exit and operand
  rewriters, guided by the known target (A* with byte-distance heuristic), each checked by B.
- D. cc1 reproduction at the RTL level (optional but principled).
- E. Per-function knob search (a flag sweep that reaches codegen-internal decisions).
- F. Integration into the project build and an LLM-agent CLI.
- G. Validation, rollout, regression.

## Why LLM-friendly

The CLI mirrors a per-function isolation harness: point it at a function, it shows the remaining
divergence and the transform or knob that closes it. Agents drive it the same way a human would,
one function at a time, with the equivalence verifier as the correctness gate.

## Status

Phases A, B and the core of C are working.

- A. Divergence taxonomy: classifier ranks the wall families (register naming is the largest,
  so the register-reallocation solver is built first).
- B. Equivalence verifier: done. `tools/equiv.py` is a z3-backed MIPS1 block-level checker.
  Validated on a 911-function disassembled target corpus: every function verifies equivalent
  to itself, and every deliberately corrupted variant is rejected as not-equivalent
  (911/911, zero self-failures, zero missed corruptions).
- C. Target-guided solver: `tools/solve.py` proven end to end. It derives the register
  correspondence the target implies (including dead-register coalescing), rebuilds through the
  real toolchain, applies a rule-general delay-slot-fill pass, reaches byte-identical target
  output, and the verifier independently proves the original compile equivalent to the target
  under that register map. Demonstrated on both a reg-realloc-plus-delay-fill struct copy and,
  in isolation, the pure register-naming class (a straight-line bit-merge whose only divergence
  was a constant-to-register coloring swap, closed by the register correspondence alone).
  `tools/batch.py` runs the solver over a list of functions, classifies each as verified,
  bytes-only (bytes match but the verifier did not prove it, a suspect that never counts as a
  match), or not-solved, and caches verified wins in a manifest so a solved function is never
  re-searched. Address-form, un-hi-cse, exit-merge and epilogue-unfill rewriters are proven;
  the epilogue-unfill pass moves a stack deallocation out of the jump-register delay slot to
  before the return when the target frees the frame first, a semantics-preserving reorder the
  verifier confirms EQUAL.

The tool was started while decompiling Digimon World 2 (PSX, SLUS-01193), the motivating case
study, but the method is general to any PSX-era GCC 2.x matching decomp.

Case study status (2026-09-26): 785 of 907 functions (86.55%) of the Digimon World 2 main
executable are matched and rebuild byte-identical, up from 771 (85.01%) at the start of this
round. New this round: the assembler wrapper expands a large constant-offset load through its
own destination register like the original assembler, sra_keep (an arithmetic shift of a
zero-extended value), ior_add (an or of disjoint bit fields kept as an add), call_slot_sink (a
callee-saved constant moved from a call slot to the next jump slot), giv_rebase (an induction
base derived from the register just set from the same source), the pre-sigma scheduler pairing
symbolic loads whose temporaries differ, callee-saved renaming that includes $fp and tries
rotations of three registers, constant folding past conditional branches, and a callee argument
read analysis that follows the callee's control flow instead of single blocks.

## Tools

- `tools/equiv.py` - Phase B semantic-equivalence verifier. Symbolic execution over a MIPS1
  (R3000, no GTE) model with z3: register file, byte-addressed memory, hi/lo, coprocessor
  state, branch and load-delay ordering. Splits each function into basic blocks, requires the
  two control-flow graphs isomorphic, then proves each block pair equivalent from a havoc entry
  state over that block's live-out set. Handles register renaming (via a supplied correspondence
  map), dead-temp differences, and loops without unrolling. Two optional, semantics-preserving
  CFG canonicalizations let functions that differ only in tail or scheduling structure still
  verify: `canonicalize_exits` folds a multi-exit tail into one shared `jr $ra`, and
  `canonicalize_delays` normalizes branch delay-slot placement (sink each branch's delay
  instruction into its sole-predecessor successors, then drop copies proven dead). Both are
  applied identically to both sides and validated to keep the full corpus self-test green.
  Unmodelled instructions become uninterpreted functions of their operands, so it can never
  prove equivalence across a genuine change. Run `equiv.py a.s b.s` for a pair or
  `equiv.py --selftest DIR` for the corpus test.
- `tools/solve.py` - Phase C target-guided solver. Derives the register correspondence from the
  known target, applies it to the cc1 assembly, rebuilds through maspsx and the assembler, runs
  the address-form, un-hi-cse, exit-merge, delay-unfill, delay-slot-fill, epilogue-unfill,
  operand-recolor and commutative-operand-swap passes (each fired target-guided), compares to the
  target bytes, and gates the
  result on `equiv.py`. Builds the gate's control-flow graph from resolved branch addresses
  (including jal relocations folded back to their symbol) so branchy functions and calls verify
  correctly.
- `tools/batch.py` - batch driver over the solver. Takes a JSON job list or a directory of
  `f<FUNC>.c` stubs, reports verified / bytes-only / not-solved / error per function, and writes
  the verified wins to `solved_manifest.json`. A bytes-only result forces a nonzero exit so a
  would-be fake match can never pass a CI run silently.
- `tools/asmlib.py` - toolchain harness (compile C to cc1 assembly, assemble to instruction
  words, read splat-style target `.s`). Configured by environment variables so it plugs into
  any project's toolchain; see the module docstring.
- `tools/classify_divergence.py` - Phase A classifier. Parses a project's parked-function journal
  into a structured divergence taxonomy and ranks the solver build order by wall-family
  frequency. Adapt the paths to your project.
- `tools/phase_f.py` - build integration with the local honesty gate. Splice-normalizes a
  translation unit's cc1 `.s` in place: it splits the file into `.ent NAME .. .end NAME` spans and,
  for each function named in a manifest, applies the same solver `.s`-text operators to that span
  only, leaving every other function byte-for-byte verbatim. An empty manifest rewrites the file
  byte-identically (a round-trip proof). `honesty_gate` refuses to proceed unless the solver reports
  a verified match for every manifest function, and the final linked-image checksum is the ship gate.
- `tools/asm_normalizer.py` - the zero-dependency version of the same seam, meant to be dropped
  into a project's build tree and committed. It carries its own MIPS register table and the
  `.s`-text operators, imports neither the solver nor the verifier, and takes the toolchain paths
  and retail-asm root from the caller, so no project symbols or paths are baked in. The manifest is
  `{"<function>": {"passes": [<pass name>, ...]}}` and lists the ordered rewrites to replay
  deterministically (no search). Pass names include `reg_realloc` (apply the derived register
  correspondence), `web_realloc` (rename one register live range at a time, interference-checked),
  `commutative_swap`, `operand_recolor`, `save_slot` (re-lay callee-saved stack slots),
  `sched_match` (reorder independent units to the target order), `taken_fill` /
  `fallthrough_fill` (delay-slot fill from either path), `dead_code`, `zero_remat` (constant
  re-materialization), `shift_const_fold`, `exit_merge`, `un_hi_cse`, `laform`, and the word-level
  `delay_fill`, `epilogue_unfill`, `reorder_indep`, `padnop`. `assemble_words` is memoized on the
  assembled file's bytes, which makes a recipe search 8-50x faster.
  Keep the verified honesty gate in `phase_f.py`/`solve.py` as a pre-commit check; the shipped build
  just replays the recorded passes and relies on the linked checksum.

## Integrating into a build (Phase F)

The solver proves a rewrite per function; Phase F applies those proven rewrites inside a real
build, between cc1 and the assembler, without forking the build script. Two ways to wire it:

- Non-invasive: import your project's build module and reassign its per-file compile step to a
  wrapper that calls `normalize_s(s_path, manifest)` after cc1 and before the assembler, then run
  the build unchanged. Python resolves a module global at call time, so reassigning the build's
  compile function takes effect and the tracked build file stays byte-untouched. Good when the
  normalizer must stay out of the shipped tree.
- Upstream: once stable, call `normalize_s` from the build directly so matches build from a clean
  checkout.

Two things that quietly cost bytes and are handled here: read and write the cc1 `.s` as raw bytes
(it is frequently CRLF; a universal-newline read plus a `\n` write strips every `\r`, so unchanged
regions stop round-tripping), and prove the normalizer is load-bearing by confirming the plain
build without it produces a different checksum on the same tree. A match is only real when the
normalizer is what closes the gap, not a function that already matched.

## License

MIT. See [LICENSE](LICENSE). Credit required, modification and redistribution allowed.
