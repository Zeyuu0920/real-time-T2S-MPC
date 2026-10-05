# Local validation

The initial extraction checks below were completed in the existing research environments. Later validation is recorded by date. No fresh environment installation or complete paper evaluation has been performed.

- 12 existing core tests passed: replay buffer, command release, and asynchronous neural updates.
- 19 additional tests passed: delayed transition alignment, numerical linearization, two-timescale updates, and SSI RK4 behavior.
- Both public launchers support standard-library-only help and dry-run.
- Quadrotor argument construction was checked against the actual controller parsers. All 120 selected historical configurations were compared with the public protocol; output tags and destinations differ.
- All 120 Go2 configurations round-trip exactly through their configuration classes. All ten field checksums match their originals, and 93,010 stored three-component friction samples match the public field adapter exactly.
- Both experiment implementations import successfully with the existing native dependencies. The Go2 adapter bindings were checked without constructing a world.
- The public Python files pass syntax compilation, local source imports have no missing modules, and local documentation links resolve.
- After removing the dependency patch from the release, Go2 module imports and active gait/reference/foothold helpers were checked with the original pinned upstream `com_trajectory.py`. The public entrypoint does not call its locally extended features; no simulation was run for this check.

These initial checks established source extraction, configuration construction, and selected numerical properties. They did not cover native solver generation or closed-loop execution; the short interface checks below subsequently cover those paths. Fresh installation, complete paper-result reproduction, and hardware execution remain unvalidated.

Copied files and their original working-tree hashes are recorded in [source provenance](source_provenance.json). Third-party revisions are recorded in [dependency provenance](dependencies.json). Original source formatting, including existing whitespace, is retained.

## Homepage and setup revision (2026-10-03)

The README uses existing paper figures and a new `environment.yaml` / `scripts/setup.sh` installation entrypoint. The setup helper was checked for Bash syntax, help, invalid arguments, and read-only previews for quadrotor, Go2, and both systems. Its previews did not create `external/` or change repository files. Both README experiment examples were checked in standard-library-only dry-run mode. These checks do not constitute a fresh package installation or a closed-loop trial.

## 2026-10-04: custom experiments and batch simulation reproduction

- Added separate README contents, demonstration video, custom-run instructions, and fixed paper-simulation reproduction instructions.
- Full test suite: **88 passed** in the existing `go2-t2s` research environment. Tests cover default configuration preservation, custom disturbance propagation into actual MuJoCo world properties, input validation, fixed batch planning, artifact aggregation, and interruption/resume behavior. No closed-loop trials are launched by these tests.
- Both custom README examples passed dependency-free dry-run. `bash scripts/reproduce_paper.sh --dry-run` enumerates 240 unique trials; Bash syntax and `git diff --check` pass.
- Full MP4 is an unchanged copy of the existing final narrated video; the README GIF contains three explicitly documented excerpts.
- Fresh installation and a complete 240-trial rerun have not been performed. The batch script covers simulation; it does not reproduce hardware acquisition.


## 2026-10-05: configurable observation and actuator interfaces

- Full test suite: **165 passed** in the existing `go2-t2s` research environment. New checks cover strict JSON profiles, effective settings, unchanged default quadrotor arguments across 120 paper configurations, profile hashes across worker launch, and the actual observation and actuator components.
- Go2 observation-history tests cover delays of 0, 1, and 3 control samples and causal transition construction. The new Go2 driver uses the separately identified `go2_sim2real_intra_period_v1` protocol; the paper driver and batch protocol are unchanged.
- Two short nominal-controller closed-loop runs completed with native solver generation from this checkout: quadrotor `combined`, seed 42, 0.12 s (6 control periods); Go2 `combined`, seed 42, 0.08 s (4 control periods). Both used their checked-in example profiles.
- The quadrotor run recorded 40 ms measurement delay, the requested observation noise, 40 ms motor time constant, 8% gain range, and 2% relative thrust noise. It produced four causal delayed transitions and distinct commanded/applied thrust traces. The Go2 run completed with its 20 ms observation delay and joint-response profile, retaining observation source indices and requested/applied torque traces.
- These runs used already available native libraries and external checkouts. The first quadrotor launch lacked an importable `l4acados`; the successful launch used the existing pinned source through `PYTHONPATH`. This was not a fresh-install test.
- A further Go2 run intentionally stopped after one 20 ms interval. It exited with the expected STOP error while preserving the true 0.02 s endpoint, two coherent boundary records, and the complete interval of actuator commands. Regression checks also cover commands published at 19 ms that cannot reach WBC until the next period.
- These brief nominal runs verify interface wiring and output creation. They do not establish long-horizon stability, adaptation performance, hardware calibration, or agreement with paper results. No full 240-trial rerun was needed or performed for this interface change.
