# Changelog

All notable changes to morphos are documented here. This project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-28

### Added
- **Level-set topology optimizer** (`optimize/levelset.py`) — boundary-evolution
  design via a level-set function on the density grid (Allaire 2002; Wang, Mei &
  Wang 2003), with smooth-Heaviside conversion to SIMP density.
- **Robust three-field / delta-p optimizer** (`optimize/robust.py`) — eroded /
  nominal / dilated worst-case optimization for LPBF print tolerance (Wang,
  Lazarov & Sigmund 2011; Lazarov, Wang & Sigmund 2016).
- **Bio-inspired density seeding** (`initialize/`) — `bio_seed.py` (Murray's-law
  branching network rasterization) and `reaction_diffusion.py` (PDE-based seeds).
- **Catalog expansion** — 10 additional builders with capability metadata, plus a
  feasibility gate that rejects under-specified or contradictory requests.
- **LLMInterpreter** — natural-language intent parsing behind the `--llm` CLI flag.
- **Oracle calibration** — `OracleCalibrator` wired into the engine with a
  polynomial gain model, exposed via the `calibrate` CLI subcommand.
- Test coverage for level-set, robust, MMA, minimum-length-scale, and
  initialization modules.

### Changed
- **Field hardening** — added `Field.validate()`, input validation, and a
  rigid-rotation objectivity regression test.
- **AMG solver** — preconditioner now rebuilt every 10 iterations
  (`rebuild_amg_every`, configurable on specs) instead of every solve.
- **Minimum-length-scale filter** — corrected erosion threshold and switched to a
  ramped `beta` parameter to prevent optimizer collapse.
- LLM-router tests skip cleanly when the `anthropic` package is absent.

## [0.1.0] - 2026-06-26

Initial public release: physics-driven topology-optimization engine with
elasticity, heat, Darcy, Stokes, thermoelastic, conjugate-heat, modal, and EM
oracles; OC and MMA optimizers; and the CLI build pipeline.
