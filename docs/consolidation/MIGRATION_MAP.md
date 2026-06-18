# Consolidation map

Date: 2026-06-18. morphos is now the single home for the engine work. This records
what was pulled in from the sibling directories under `ai-agent/`, where it landed,
and what was deliberately left out.

## Decision context

The bio-inspired framing is parked for now. The near-term goal is one excellent,
general physics-to-geometry engine (the Noyron-shaped move): make the engine
genuinely good first, then point it at whatever domain we choose. So absorbed
material is preserved but kept off the engine's public surface, and the package
`src/morphos/` is untouched and stays clean.

## What landed where

| Source dir | Landed in morphos as | Status |
|---|---|---|
| `PicoGK/` (Leap71 voxel kernel, upstream clone) | `vendor/PicoGK/` (`.git` stripped, LICENSE kept) | Vendored dependency. morphos already binds its native runtime via ctypes (`src/morphos/geometry/_picogk_native.py`). Heavy native binaries are gitignored. |
| `biomimic/` | `reference/biomimic/` | Preserved, NOT wired. `src/biomimic/generative/` is engine-grade (real solvers) and the candidate-backend goldmine; the rest is the deprioritized bio-copilot product. See `reference/biomimic/INDEX.md`. |
| `biomimicry-ventures/engine/` (biotransport screens) | `reference/biotransport/` | Preserved, NOT wired. Reduced-order kill-test screens. Candidate screening/analytic backends. See `reference/biotransport/INDEX.md`. |
| `biomimicry-ventures/*.md` (opportunity maps, stress tests) | `docs/strategy/` | Internal strategy. Gitignored, never ships in a public repo. |
| `manta-filter-sim/` (minus the LBM clone) | `applications/manta-filter/` | Worked proof-of-method case study: reports, parametric geometry, print/bench specs, figures. Gitignored for now. |

## What was deliberately left out

- `manta-filter-sim/engine_lbm/` (289M): an upstream LBM solver clone. Not copied.
  It can become a morphos fluid backend later; until then it stays where it is rather
  than bloating the engine tree. It still lives at `ai-agent/manta-filter-sim/engine_lbm/`.
- `manta-filter-sim/graphify-out/` (1.8M): tooling cache artifact, not content.
- `mcp/fermat-mcp/` (251M): a separate, self-contained published MCP tool (sympy/numpy/
  matplotlib). It is dev tooling, not engine content, and keeps its own repo. Left in place.
- `longevity-transfer.zip` (4.7G, at `ai-agent/` root): the deferred medicine/phenotype
  vision. Untouched.

## The `reference/` rule

`reference/` is preserved-but-not-wired code, kept OUTSIDE `src/morphos/` on purpose so
the engine never imports half-baked backends. This honors the engine principle that
integration points stay loud and explicit, never silently wrong. Promoting any of this
into a real backend means re-implementing it behind a morphos interface
(`PhysicsOracle`, `Optimizer`, `GeometryKernel`, `Objective`) with its own
finite-difference-gated tests, not importing it as-is.

## Originals

The source directories under `ai-agent/` were COPIED, not moved. The originals are
untouched and still on disk pending an explicit decision to delete them.
