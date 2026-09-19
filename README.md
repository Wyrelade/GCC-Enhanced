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

Early. Phase A (divergence taxonomy) has a working classifier; Phases B and C are next. The tool
was started while decompiling Digimon World 2 (PSX, SLUS-01193), which is the motivating case
study, but the method is general to any PSX-era GCC 2.x matching decomp.

## Tools

- `tools/classify_divergence.py` - Phase A classifier. Parses a project's parked-function journal
  into a structured divergence taxonomy and ranks the solver build order by wall-family
  frequency. Adapt the paths to your project.

## License

MIT. See [LICENSE](LICENSE). Credit required, modification and redistribution allowed.
