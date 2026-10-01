# Real-Time T2S-MPC

Research code for **T2S-MPC: Real-Time Online Adaptive Model Predictive Control for Continuously Changing Dynamics**.

T2S-MPC learns a time-conditioned neural residual model online. Fast updates adapt the output layer to recent observations; slower updates refine the hidden representation using recent and historical samples. A local linear approximation incorporates the learned model into MPC.

This repository brings together the three-dimensional quadrotor and Unitree Go2 simulation implementations, with explicit experiment configurations and separate entrypoints. It is a release preparation version: installation in a clean environment and full paper-result reproduction remain to be validated. The included source is extracted from the current research working tree; matching the selected experiment settings does not establish that every source file matches the historical run.

## Experiments

| System | Scenarios | Controllers |
| --- | --- | --- |
| Quadrotor (PyBullet) | Increasing mean wind, increasing turbulence, combined | Nominal, SSI, STGP, T2S |
| Unitree Go2 (MuJoCo) | Liquid drain, varying friction, combined | Nominal, SSI, STGP, T2S |

The simulation uses measured controller computation time to determine command release. Timing results depend on the host and its load. These experiments do not establish hard real-time performance on a physical robot.

## Start here

Inspect the selected configuration without installing simulation dependencies or starting an experiment:

```bash
python scripts/run_quadrotor.py --method t2s --scenario combined --seed 42 --dry-run
python scripts/run_go2.py --method t2s --scenario combined --seed 42 --dry-run
```

See [installation](docs/installation.md) for the recorded environment and external dependencies, and [experiment protocols](docs/experiments.md) for the selected settings and scope of reproduction.

After installing the corresponding simulation environment, omit `--dry-run` to execute a single trial. Each entrypoint also provides `--help`.

## Layout

```text
configs/       Explicit experiment settings
assets/        Small experiment inputs, including stored Go2 friction maps
src/           Controller, learning, dynamics, and simulation implementation
scripts/       Public experiment entrypoints and required adapters
tests/         Algorithm and numerical consistency tests
requirements/  Recorded Python dependency versions
docs/          Installation and experiment documentation
```

Generated solvers and experiment output are ignored by Git. Third-party projects are installed separately; see [dependency provenance](docs/dependencies.json).

## Core tests

The replay-buffer, command-release, and asynchronous-update tests run without either simulator:

```bash
python -m pip install -r requirements/test.txt
python -m pytest tests/test_hybrid_replay.py tests/test_deadline_control.py tests/test_async_neural_update.py -q
```

Additional tests require the simulation dependencies. Full experiment reruns are separate from unit tests. See the [local validation record](docs/validation.md) for completed checks and their limits.

## License and citation

A project license and finalized citation have not yet been added. Dependency licenses remain with their respective projects. The paper title above identifies the work; this draft does not claim a publication venue or acceptance status.
