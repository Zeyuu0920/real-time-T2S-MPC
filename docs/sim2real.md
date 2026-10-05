# Sim-to-real configuration

Use JSON profiles to study how observation errors and actuator response affect the simulated controllers. The parameters are synthetic model settings; the example profiles are not hardware calibrations. This interface does not connect to a physical quadrotor or Go2.

## Run a profile

Install the corresponding simulator using the [installation instructions](../README.md#installation), then run from the repository root:

```bash
python scripts/run_quadrotor.py --method t2s --scenario combined --seed 42 \
  --sim2real-config configs/sim2real/quadrotor_example.json \
  --output-dir outputs/quadrotor_sim2real

python scripts/run_go2_sim2real.py --method t2s --scenario combined --seed 42 \
  --sim2real-config configs/sim2real/go2_example.json \
  --output-dir outputs/go2_sim2real
```

Append `--dry-run` to inspect the effective configuration without loading numerical dependencies or creating outputs. Both runners support `nominal`, `ssi`, `stgp`, and `t2s`. The Go2 runner accepts seeds 42–51, defaults to 60 s, and requires at least three available CPU cores for execution. Use a new output directory for a new comparison. Keep the profile and seed fixed when comparing controllers; a fixed seed controls the random draws, while measured computation latency still depends on the host and its load.

The quadrotor option overrides observation and motor parameters in its existing simulation backend. Omitting `--sim2real-config` retains the selected quadrotor configuration. Go2 uses a separate observation/actuator experiment with command release within the control period. It is a different timing protocol from the Go2 paper runner, even when its parameter values resemble the paper defaults.

## Profile format and units

A profile identifies its schema and system and groups settings under `observation` and `actuator`:

```json
{
  "schema_version": 1,
  "system": "quadrotor",
  "observation": {
    "position_noise_std_m": 0.015
  },
  "actuator": {
    "gain_range": 0.08
  }
}
```

This minimal example changes position-estimate noise to 1.5 cm per axis and the fixed motor-gain half-width to 8%. `schema_version: 1` and the matching `system` are required. Either section, or individual fields within it, may be omitted: quadrotor inherits the effective base-protocol settings; Go2 fills its observation/actuator defaults. Unknown fields, Boolean values used as numbers, and nonfinite numeric values are rejected.

Observation noise values are stationary one-standard-deviation errors per axis. A scalar applies to each of the three components in its state group. The [quadrotor example](../configs/sim2real/quadrotor_example.json) increases delay, noise, and lag relative to the selected quadrotor defaults. The [Go2 example](../configs/sim2real/go2_example.json) explicitly lists the separate Go2 observation/actuator model's defaults; it is not the paper Go2 observation/timing configuration.

| Observation field | Unit | Quadrotor baseline | Quadrotor example | Go2 baseline/example |
| --- | --- | --- | --- | --- |
| `delay_steps` | MPC/control samples | `1` | `2` | `1` |
| `position_noise_std_m` | m | `0.01` | `0.02` | `0.01` |
| `velocity_noise_std_mps` | m/s | `0.02` | `0.04` | `0.02` |
| `attitude_noise_std_rad` | rad | `0.01` | `0.02` | `0.01` |
| `body_rate_noise_std_radps` | rad/s | `0.02` | `0.04` | `0.02` |
| `noise_correlation_time_s` | s | `0.10` | `0.15` | `0.10` |

At the selected 50 Hz control rate, one observation-delay step is 20 ms; the quadrotor example therefore uses 40 ms. Both systems accept a nonnegative JSON integer for `delay_steps`; `0` removes the added observation delay. All observation standard deviations must be nonnegative. Quadrotor accepts a nonnegative correlation time; zero correlation time produces independent errors at successive control samples. Go2 requires a strictly positive noise correlation time, including when all noise standard deviations are zero.

| Actuator field | Unit | Quadrotor baseline | Quadrotor example | Go2 baseline/example |
| --- | --- | --- | --- | --- |
| `time_constant_s` | s | `0.025` | `0.04` | `0.005` |
| `gain_range` | dimensionless | `0.05` | `0.08` | `0.03` |
| `noise_std` | dimensionless | `0.01` | `0.02` | unsupported |
| `noise_correlation_time_s` | s | `0.05` | `0.08` | unsupported |

`gain_range` must lie in `[0, 1)`. Each rotor or joint draws one gain uniformly from `[1 - gain_range, 1 + gain_range]` for the trial. Quadrotor requires `time_constant_s > 0`; its actuator noise standard deviation and correlation time may be zero. Go2 permits `time_constant_s = 0` to remove torque lag while retaining any gain mismatch and torque limits.

## Where the effects enter

### Quadrotor

At each control sample, a temporally correlated error is added to the simulated position, velocity, Euler angles, and body rates. Angles are wrapped after perturbation. The noisy packet enters the measurement-delay queue with its source timestamp; noise is not redrawn while the packet is delayed. The controller and online learner use these packets. Tracking metrics use the simulator's true state.

Quadrotor profiles configure the `first_order` actuator model, which evolves rotor speed at the 500 Hz physics rate. Its time constant is used in the plant, nominal MPC dynamics, and nominal motor-state observer. Changing `time_constant_s` therefore changes a matched model parameter; it does not create an unknown lag mismatch by itself. Fixed thrust-gain errors and multiplicative thrust noise act only on the plant. `noise_std` is the stationary relative thrust-noise standard deviation, not an additive force in newtons. Zero actuator noise correlation time gives independent draws at successive physics updates.

### Go2

At each 50 Hz MPC boundary, noise perturbs one coherent base configuration and velocity packet. Position and translational-velocity errors are expressed in world coordinates. Attitude noise is a world-frame rotation-vector perturbation applied to the quaternion; angular-velocity errors affect body rates. Joint positions and velocities remain ideal at the packet's source time and travel with the same delayed MPC packet. The MPC state and foot geometry are then derived from the same noisy packet and delivered according to its source timestamp. Online learning uses aligned delayed transitions and dispatched command histories.

The 500 Hz whole-body controller (WBC) keeps its separate ideal simulated proprioception. The profile does not add joint-encoder noise or delay to that low-level feedback. After WBC computes requested joint torques, the actuator model applies per-joint gain mismatch, torque limits, and a first-order torque response at the 1 kHz physics rate. It adds no separate random torque-noise process. This response is a synthetic model, not an identified Go2 motor transfer function.

## Measurement age and computation time

`observation.delay_steps` controls how old a measurement is when it reaches MPC, including any estimator or sensor-transport delay represented by that value. Solver runtime and command-release latency are recorded separately.

The quadrotor retains its measured-computation command-release policy. The Go2 observation/actuator experiment measures controller response time separately and may publish an accepted command within the current 20 ms period, on the 1 ms physics grid; WBC consumes it on its 2 ms update grid. Late candidates are discarded. This differs from the fixed Go2 paper pipeline's one-period command-release delay. Adding a 20 ms observation age is not equivalent to adding a 20 ms command delay.

## Recorded settings and diagnostics

The dry-run output and each trial's `run_protocol.json` contain the effective profile, so omitted fields can be checked before comparing results.

- Quadrotor records `sim2real.input.path` and `sim2real.input.sha256`, the requested overrides, effective observation/actuator values, control frequency, measurement delay in milliseconds, and the shared scope of the motor time constant.
- Go2 records `sim2real_profile.path`, `sim2real_profile.sha256`, and `sim2real_profile.effective`, plus the effective timing and environmental disturbances. `interfaces.json` contains the numerical interface settings; `actuator_gain.json` stores the sampled per-joint gains. `boundary_data.npz` retains observations, source indices, and requested/applied torque histories. Its metrics identify `go2_sim2real_intra_period_v1`.

The SHA-256 values identify the input profile bytes; effective values show the settings after defaults are filled. Retain these records with the seed and trial outputs. They establish which simulation parameters were used, not whether those parameters were measured on hardware.

## Keep profile experiments separate from paper reproduction

[`scripts/reproduce_paper.sh`](../scripts/reproduce_paper.sh) uses the fixed paper configurations and never loads sim-to-real profiles. The dedicated Go2 runner is outside that batch. Profile overrides define custom experiments and must not be mixed into the paper summaries; the batch summarizer checks protocol metadata and effective settings against its planned trials. See [experiment protocols](experiments.md) for the paper timing and metrics.
