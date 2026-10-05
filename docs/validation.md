# Local validation

The following checks were completed in the existing research environments. No new environment was installed and no full closed-loop experiment was run.

- 12 existing core tests passed: replay buffer, command release, and asynchronous neural updates.
- 19 additional tests passed: delayed transition alignment, numerical linearization, two-timescale updates, and SSI RK4 behavior.
- Both public launchers support standard-library-only help and dry-run.
- Quadrotor argument construction was checked against the actual controller parsers. All 120 selected historical configurations were compared with the public protocol; output tags and destinations differ.
- All 120 Go2 configurations round-trip exactly through their configuration classes. All ten field checksums match their originals, and 93,010 stored three-component friction samples match the public field adapter exactly.
- Both experiment implementations import successfully with the existing native dependencies. The Go2 adapter bindings were checked without constructing a world.
- The public Python files pass syntax compilation, local source imports have no missing modules, and local documentation links resolve.
- After removing the dependency patch from the release, Go2 module imports and active gait/reference/foothold helpers were checked with the original pinned upstream `com_trajectory.py`. The public entrypoint does not call its locally extended features; no simulation was run for this check.

These checks establish source extraction, configuration construction, and selected numerical properties. Fresh installation, native solver generation from this checkout, closed-loop smoke runs, complete paper-result reproduction, and hardware execution have not been validated.

Copied files and their original working-tree hashes are recorded in [source provenance](source_provenance.json). Third-party revisions are recorded in [dependency provenance](dependencies.json). Original source formatting, including existing whitespace, is retained.

## Homepage and setup revision (2026-10-03)

The README uses existing paper figures and a new `environment.yaml` / `scripts/setup.sh` installation entrypoint. The setup helper was checked for Bash syntax, help, invalid arguments, and read-only previews for quadrotor, Go2, and both systems. Its previews did not create `external/` or change repository files. Both README experiment examples were checked in standard-library-only dry-run mode. These checks do not constitute a fresh package installation or a closed-loop trial.

## 2026-10-04: custom experiments and batch simulation reproduction

- Added separate README contents, demonstration video, custom-run instructions, and fixed paper-simulation reproduction instructions.
- Full test suite: **88 passed** in the existing `go2-t2s` research environment. Tests cover default configuration preservation, custom disturbance propagation into actual MuJoCo world properties, input validation, fixed batch planning, artifact aggregation, and interruption/resume behavior. No closed-loop trials are launched by these tests.
- Both custom README examples passed dependency-free dry-run. `bash scripts/reproduce_paper.sh --dry-run` enumerates 240 unique trials; Bash syntax and `git diff --check` pass.
- Full MP4 is an unchanged copy of the existing final narrated video; the README GIF contains three explicitly documented excerpts.
- Fresh installation and a complete 240-trial rerun have not been performed. The batch script covers simulation; it does not reproduce hardware acquisition.
