# Experiment protocols

The public entrypoints make the selected settings explicit. They launch fresh trials from the current source extraction and do not reconstruct the paper tables from the original logs. Original raw outputs, manuscript drafts, diagnostic batches, and hardware code are outside this initial simulation release.

## Quadrotor

The selected comparison uses three-dimensional tracking with four controllers: Nominal, SSI, STGP, and T2S. The three scenarios increase mean wind speed, turbulence intensity, or both. The evaluation uses seeds 42–51, 20 seconds per trial, 50 Hz control, 500 Hz physics, and a fixed 20 ms measurement delay with noisy observations.

T2S uses a 50-entry recent FIFO and a 50-entry reservoir of FIFO-evicted history. Fast updates occur every two control steps using the four most recent samples and five epochs. Slow updates occur every 50 steps using 32 samples from each partition and 20 epochs. Both learning rates are 0.001. Sampling is without replacement; source timestamps and applied command histories are preserved.

The selected SSI baseline uses six residual outputs and an author-style full-model RK4 prediction-error update with normalized commanded-thrust features. The selected STGP baseline uses fixed preset hyperparameters and posterior-mean adaptation. Offline-calibrated and artificial-latency diagnostic variants are not the selected STGP comparison.

Controller computations completing within the deadline may release a command within the control period; candidates exceeding 20 ms are discarded. This is simulation with measured computation latency. Source versions and host load can affect new results.

The paper reports each trial's time-mean Euclidean position error and time-mean SO(3) attitude angle, followed by the mean and sample standard deviation (`ddof=1`) across seeds. These are not RMSE. Deadline misses count samples with `control_critical_ms > 20`; the separately recorded command-release misses use a different definition. The current paper's STGP reference uses the `stgp_preset_repeat_10seeds_0911b` cohort; the public STGP arguments match that cohort apart from output paths and labels.

The configuration is in [quadrotor.json](../configs/quadrotor.json). In particular, the selected T2S entrypoint explicitly enables hybrid replay; an older FIFO default is not the paper setting.

## Go2

The selected configuration uses seeds 42–51 and targets 60 seconds per trial, at 0.5 m/s forward speed and 0.30 m body height. Frequencies are 50 Hz MPC, 500 Hz low-level control, and 1000 Hz physics. The MPC friction-cone coefficient is fixed at 0.5.

| Scenario | Payload | Ground |
| --- | --- | --- |
| Liquid drain | 0.6 kg container; liquid decreases from 3.4 to 1.4 kg during 5–45 s, with the retained slosh model | Fixed contact friction vector `[0.8, 0.02, 0.01]` |
| Varying friction | Fixed 4 kg | Stored spatial sliding-friction field in `[0.50, 0.80]`, with the retained torsional/rolling fields |
| Combined | Same drain model | Same spatial field as the varying-friction scenario for each seed |

The controller does not read the true ground friction. Sharing a spatial field does not guarantee that controllers following different trajectories encounter the same friction at the same time.

The T2S model has 24 physical inputs (12 state entries and 12 commanded contact forces), 32 time features, two 64-unit hidden layers, and six residual wrench outputs. Output scales are `[40, 40, 40, 4, 4, 4]`. Fast updates occur every three MPC steps using four recent valid samples and five epochs. Slow updates occur every 50 steps using 32 recent plus 32 historical samples and 20 epochs. Both learning rates are 0.0002; replay capacity is 50 recent plus 50 historical samples. SSI and STGP retain their own selected settings.

Timing uses `latency_replay_pipeline`: measured controller computation is mapped to simulation time, with a one-period command-release delay. Late results are discarded and the previous command is held. Busy periods do not launch overlapping solves. Physics integration time is not part of the controller deadline.

Report planned control periods, launched tasks, overdue tasks, and busy periods separately. Completion rates use all ten planned seeds. Full-trajectory RMSE uses completed trials; failure times are reported separately. A failed prefix is not a completed trajectory.

The historical Go2 comparison combined runs from different batches. Its seeds were used during development and are not an independent held-out generalization set. The public launcher does not promise identical wall-clock timings or bitwise reproduction of these records.

The Go2 configuration is in [final.json](../configs/go2/final.json); the associated field files are checksum-verified before each run.

Run trials serially from a checkout: native solver generation uses shared locations, so simultaneous processes can overwrite generated solver artifacts. Use separate checkouts for concurrent experiments.

## Validation boundary

Dry-run verifies entrypoint configuration construction without importing simulation libraries. Unit tests check selected algorithmic and numerical properties. Neither replaces a fresh environment installation or complete closed-loop experiment rerun. Such reruns must record dependency versions, source hashes, seeds, host resources, timing, failures, and all planned trials.

## Custom disturbance experiments

The public runners expose disturbance overrides for exploratory comparisons. Quadrotor `--wind-scale` and `--turbulence-scale` multiply the initial and final three-axis mean wind and turbulence standard deviations, respectively. They preserve ramp duration, spatial gradients, advection, and correlation-length parameters. A scale of one retains the selected settings; zero disables that wind component.

Go2 `--payload-mass` changes the fixed payload only in the `friction` scenario. `--liquid-mass-scale` changes liquid density and therefore the draining mass history in `drain` or `combined`, with the container fixed at 0.6 kg. `--friction-min` and `--friction-max` remap the frozen sliding-friction field from [0.50, 0.80] to the requested interval while retaining its spatial pattern. Torsional and rolling friction fields and the MPC cone coefficient remain unchanged. `--ground-friction` changes the uniform sliding coefficient in `drain`.

Custom overrides are written into the run metadata. They define new experiments and do not change the fixed configurations used by `scripts/reproduce_paper.sh`.

## Batch simulation reproduction

`bash scripts/reproduce_paper.sh` schedules 120 quadrotor trials and 120 Go2 trials from the fixed configuration files: three scenarios, four methods, and seeds 42–51 per system. This reruns the simulation evaluation; it does not execute the hardware experiment or recover historical outputs. `--dry-run` inspects the schedule without creating output files. See the README for the command and output summary.
