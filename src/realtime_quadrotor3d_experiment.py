"""Shared full-3-D stabilization experiment for Neural MPC and T2S-MPC."""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybullet as pb
import torch
from l4casadi.realtime import RealTimeL4CasADi
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import QuadType

from src.aligned_quadrotor_transition import CausalAlignedTransitionBuffer
from src.hybrid_replay import RecentReservoirReplay
from src.models import MLP, TwoSpeedMLP, time_embedding_np
from src.deadline_control import ComputationAwareControlRelease
from src.actuated_quadrotor_dynamics import (
    RIGID_BODY_STATE_DIM,
    motor_amplitude_from_thrust,
)
from src.motor_actuator import (
    add_motor_actuator_arguments,
    make_motor_actuator,
)
from src.mpc import MPC
from src.multirate_quadrotor import MultirateWindQuadrotor
from src.physical_wind_3d import (
    add_physical_wind_3d_arguments,
    make_physical_wind_3d,
)
from src.realtime_dynamics_3d import (
    RESIDUAL_DERIVATIVE_INDICES,
    Quadrotor3DRealTimeDynamics,
    Quadrotor3DRealTimeT2SDynamics,
)
from src.realtime_neural import (
    AsyncRealtimeTrainer,
    RealtimeTrainingJob,
    evaluate_first_order,
    exact_model_output,
    first_order_parameters,
    load_numpy_state_dict,
    relu_activation_patterns,
)
from src.realtime_knode import (
    AsyncRealtimeKNODETrainer,
    RealtimeKNODEJob,
    TaskMatchedKNODEResidual,
)
from src.realtime_t2s import (
    AsyncRealtimeT2STrainer,
    RealtimeT2SJob,
    sample_replay_without_replacement,
)
from src.ssi_mpc import (
    Quadrotor3DNominalDynamics,
    Quadrotor3DSSIDynamics,
    SSIOnlineLearner,
    SSIRandomFeatures,
    SSI_THRUST_FEATURE_MODES,
)
from src.spatiotemporal_gp_mpc import (
    QuadrotorSTGPResidualAdapter,
    QuadrotorSpatioTemporalGPLearner,
    discrete_acceleration_residual_map,
)
from src.state_measurement import (
    DelayedNoisyStateEstimator,
    state_noise_standard_deviations,
    wrap_angles,
)
from src.utils import get_mass, get_pb_handles


N = 20
T_HORIZON = 1.0
CONTROL_FREQUENCY = 50  # Backward-compatible default; CLI may select 100 Hz.
PHYSICS_FREQUENCY = 500
PHYSICS_DT = 1.0 / PHYSICS_FREQUENCY
TIME_FEAT_DIM = 16
TIME_SCALE = 1.0
REPLAY_MAX = 100
GRAVITY = 9.81
STATE_NAMES = (
    "x", "x_dot", "y", "y_dot", "z", "z_dot",
    "roll", "pitch", "yaw", "p", "q", "r",
)
RESIDUAL_NAMES = ("x_ddot", "y_ddot", "z_ddot", "p_dot", "q_dot", "r_dot")


def parse_args(controller):
    parser = argparse.ArgumentParser(
        description=f"Full-3-D quadrotor control with real-time {controller} MPC"
    )
    parser.add_argument(
        "--task",
        choices=("stabilization", "circle"),
        default="stabilization",
    )
    parser.add_argument("--tracking-period", type=float, default=8.0)
    parser.add_argument("--tracking-radius", type=float, default=1.0)
    parser.add_argument("--tracking-altitude", type=float, default=1.0)
    parser.add_argument("--control-frequency", type=int, default=CONTROL_FREQUENCY)
    parser.add_argument("--measurement-delay-steps", type=int, default=1)
    parser.add_argument("--position-noise-std", type=float, default=0.01)
    parser.add_argument("--velocity-noise-std", type=float, default=0.02)
    parser.add_argument("--attitude-noise-std", type=float, default=0.01)
    parser.add_argument("--body-rate-noise-std", type=float, default=0.02)
    parser.add_argument(
        "--measurement-noise-correlation-time", type=float, default=0.10
    )
    parser.add_argument("--wind", choices=("dryden", "none"), default="dryden")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for this run's result CSVs and plots")
    parser.add_argument(
        "--result-tag", default="",
        help="Optional short identifier for a separately retained experiment",
    )
    parser.add_argument(
        "--compact-result-filenames",
        action="store_true",
        help=(
            "Use a shorter controller prefix in result filenames. This only "
            "changes file names; the complete configuration is still written "
            "to the summary CSV."
        ),
    )
    parser.add_argument(
        "--training-epochs", type=int,
        default=60 if controller == "knode" else 20,
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--control-cpu-core",
        type=int,
        default=None,
        help="pin the control/simulation process to this logical CPU",
    )
    parser.add_argument(
        "--trainer-cpu-core",
        type=int,
        default=None,
        help="pin the asynchronous neural trainer to this logical CPU",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--deadline-aware",
        action="store_true",
        help=(
            "Apply a hard-real-time release policy: the current "
            "control is held during computation, on-time candidates become "
            "active for the remaining 500 Hz substeps, and late candidates "
            "are discarded"
        ),
    )
    parser.add_argument(
        "--control-release-mode",
        choices=ComputationAwareControlRelease.VALID_MODES,
        default="intra_period",
        help=(
            "intra_period applies an on-time candidate at the first 500 Hz "
            "substep after computation; period_boundary holds the previous "
            "command for the full current control period and publishes an "
            "on-time candidate at the next control boundary"
        ),
    )
    if controller == "neural":
        parser.add_argument("--update-every", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=8)
    elif controller == "t2s":
        parser.add_argument("--fast-update-every", type=int, default=5)
        parser.add_argument("--fast-batch-size", type=int, default=8)
        parser.add_argument("--slow-update-every", type=int, default=50)
        parser.add_argument("--slow-batch-size", type=int, default=64)
        parser.add_argument("--fast-epochs", type=int, default=10)
        parser.add_argument("--slow-epochs", type=int, default=20)
        parser.add_argument(
            "--t2s-replay-mode", choices=("fifo", "fifo_reservoir"),
            default="fifo",
            help="100 FIFO entries, or 50 recent FIFO plus 50 historical reservoir entries",
        )
    elif controller == "ssi":
        parser.add_argument("--ssi-features", type=int, default=50)
        parser.add_argument("--ssi-learning-rate", type=float, default=0.25)
        parser.add_argument("--ssi-kernel-std", type=float, default=0.01)
        parser.add_argument(
            "--ssi-thrust-feature",
            choices=SSI_THRUST_FEATURE_MODES,
            default="lagged_physical",
            help=(
                "lagged_physical uses the current motor-state thrust a_i^2; "
                "normalized_command uses u_i/u_i,max as in the published "
                "quadrotor SSI feature design"
            ),
        )
        parser.add_argument(
            "--ssi-residual-dimension",
            type=int,
            choices=(3, 6),
            default=3,
            help=(
                "SSI output dimension: translational acceleration only (3) "
                "or translational plus angular acceleration (6)"
            ),
        )
    elif controller == "stgp":
        parser.add_argument("--stgp-inducing-points", type=int, default=80)
        parser.add_argument(
            "--stgp-inducing-seed",
            type=int,
            default=0,
            help=(
                "fixed Sobol inducing-point seed, independent of the plant "
                "and disturbance seed"
            ),
        )
        parser.add_argument(
            "--stgp-spatial-lengthscale", type=float, default=1.5
        )
        parser.add_argument(
            "--stgp-temporal-lengthscale", type=float, default=5.0
        )
        parser.add_argument(
            "--stgp-linear-output-variance", type=float, default=1.0
        )
        parser.add_argument(
            "--stgp-angular-output-variance", type=float, default=4.0
        )
        parser.add_argument(
            "--stgp-observation-noise-variance", type=float, default=0.36
        )
        parser.add_argument(
            "--stgp-process-noise-variances",
            type=float,
            nargs=6,
            default=(1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3),
            metavar=("WX", "WY", "WZ", "WP", "WQ", "WR"),
            help=(
                "irreducible six-axis acceleration process-noise variances "
                "used by stochastic zoRO covariance propagation"
            ),
        )
        parser.add_argument(
            "--stgp-backoff-scaling",
            type=float,
            default=0.0,
            help=(
                "must be zero in the shared tracking benchmark: posterior "
                "variance is recorded but does not tighten constraints"
            ),
        )
    elif controller == "knode":
        parser.add_argument("--collection-duration", type=float, default=2.0)
        parser.add_argument("--data-start-delay", type=float, default=3.0)
    add_motor_actuator_arguments(parser)
    add_physical_wind_3d_arguments(parser)
    args = parser.parse_args()
    if len(args.result_tag) > 16 or any(
        not (character.isascii() and (character.isalnum() or character in "_-"))
        for character in args.result_tag
    ):
        parser.error("--result-tag must contain at most 16 ASCII letters, digits, '_' or '-'")
    positive = [
        "duration", "training_epochs", "torch_threads", "tracking_period",
        "tracking_radius", "tracking_altitude",
    ]
    if controller == "neural":
        positive.extend(("update_every", "batch_size"))
    elif controller == "t2s":
        positive.extend(
            (
                "fast_update_every", "fast_batch_size", "slow_update_every",
                "slow_batch_size", "fast_epochs", "slow_epochs",
            )
        )
    elif controller == "ssi":
        positive.extend(
            ("ssi_features", "ssi_learning_rate", "ssi_kernel_std")
        )
    elif controller == "stgp":
        positive.extend(
            (
                "stgp_inducing_points",
                "stgp_spatial_lengthscale",
                "stgp_temporal_lengthscale",
                "stgp_linear_output_variance",
                "stgp_angular_output_variance",
                "stgp_observation_noise_variance",
            )
        )
    elif controller == "knode":
        positive.extend(("collection_duration", "data_start_delay"))
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("durations, periods, batch sizes, epochs and threads must be positive")
    nonnegative = (
        "measurement_delay_steps", "position_noise_std", "velocity_noise_std",
        "attitude_noise_std", "body_rate_noise_std",
        "measurement_noise_correlation_time", "motor_gain_range",
        "motor_noise_std", "motor_noise_correlation_time",
    )
    if any(getattr(args, name) < 0 for name in nonnegative):
        parser.error("measurement delay and noise parameters cannot be negative")
    if args.control_frequency <= 0:
        parser.error("--control-frequency must be positive")
    available_cpu_count = os.cpu_count() or 1
    for name in ("control_cpu_core", "trainer_cpu_core"):
        value = getattr(args, name)
        if value is not None and not 0 <= value < available_cpu_count:
            parser.error(
                f"--{name.replace('_', '-')} must be in "
                f"[0, {available_cpu_count - 1}]"
            )
    if (
        args.control_cpu_core is not None
        and args.trainer_cpu_core is not None
        and args.control_cpu_core == args.trainer_cpu_core
    ):
        parser.error("control and trainer CPU cores must be different")
    if args.actuator_model == "first_order" and args.motor_time_constant <= 0:
        parser.error("--motor-time-constant must be positive")
    if args.motor_gain_range >= 1.0:
        parser.error("--motor-gain-range must be smaller than one")
    if controller == "stgp":
        if any(value < 0.0 for value in args.stgp_process_noise_variances):
            parser.error("STGP process-noise variances cannot be negative")
        if not np.isclose(args.stgp_backoff_scaling, 0.0):
            parser.error(
                "--stgp-backoff-scaling must be 0: the shared tracking "
                "benchmark does not use chance-constraint tightening"
            )
    if PHYSICS_FREQUENCY % args.control_frequency:
        parser.error(
            f"--control-frequency must divide the {PHYSICS_FREQUENCY} Hz "
            "physics frequency exactly"
        )
    if controller == "neural" and args.batch_size > REPLAY_MAX:
        parser.error(f"--batch-size cannot exceed replay capacity {REPLAY_MAX}")
    if controller == "t2s" and (
        args.fast_batch_size > REPLAY_MAX or args.slow_batch_size > REPLAY_MAX
    ):
        parser.error(
            f"batch sizes cannot exceed replay capacity {REPLAY_MAX} when "
            "sampling without replacement"
        )
    if controller == "t2s" and args.t2s_replay_mode == "fifo_reservoir":
        if args.fast_batch_size > REPLAY_MAX // 2:
            parser.error("Fast batch cannot exceed the 50-entry recent FIFO")
        if args.slow_batch_size % 2:
            parser.error("Hybrid replay requires an even slow batch size for equal contributions")
    return args


def _initial_state(seed, args):
    if args.task == "circle":
        return reference_state(args, 0.0, np.zeros(12, dtype=float))

    rng = np.random.default_rng(seed)
    state = np.zeros(12, dtype=float)
    state[[0, 2]] = rng.uniform(-0.25, 0.25, size=2)
    state[4] = rng.uniform(0.75, 1.25)
    state[6:8] = rng.uniform(-0.08, 0.08, size=2)
    state[8] = rng.uniform(-0.10, 0.10)
    state[9:12] = rng.uniform(-0.03, 0.03, size=3)
    return state


def make_environment(args, seed):
    return MultirateWindQuadrotor(
        quad_type=QuadType.THREE_D,
        task_info={
            "stabilization_goal": [0.0, 0.0, 1.0],
            "stabilization_goal_tolerance": 0.05,
        },
        init_state=_initial_state(seed, args),
        randomized_init=False,
        gui=args.gui,
        ctrl_freq=args.control_frequency,
        pyb_freq=PHYSICS_FREQUENCY,
        episode_len_sec=max(1, int(np.ceil(args.duration))),
        done_on_out_of_bound=False,
        adversary_disturbance=None,
        seed=seed,
    )


def _zero_initial_residual(model):
    with torch.no_grad():
        if isinstance(model, TwoSpeedMLP):
            model.layer3.weight.zero_()
            model.layer3.bias.zero_()
        elif isinstance(model, TaskMatchedKNODEResidual):
            model.queue_weights.zero_()
            model.hidden_masks.zero_()
            model.active_count.zero_()
        else:
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
    model.eval()


def initialize_solver(solver, state, env):
    hover = np.asarray(env.U_GOAL, dtype=float)
    for stage in range(N):
        solver.set(stage, "x", state)
        solver.set(stage, "u", hover)
    solver.set(N, "x", state)


def horizon_features(solver, current_time, controller, measured_state=None):
    state_dim = int(np.asarray(solver.get(0, "x")).size)
    base_input_dim = state_dim + 4
    input_dim = base_input_dim + (
        TIME_FEAT_DIM if controller == "t2s" else 0
    )
    features = np.empty((N, input_dim), dtype=np.float64)
    embeddings = None
    if controller == "t2s":
        embeddings = np.empty((N, TIME_FEAT_DIM), dtype=np.float64)
    for stage in range(N):
        features[stage, :state_dim] = solver.get(stage, "x")
        features[stage, state_dim:base_input_dim] = solver.get(stage, "u")
        if controller == "t2s":
            tau = (current_time + stage * T_HORIZON / N) / TIME_SCALE
            embeddings[stage] = time_embedding_np(tau, d=TIME_FEAT_DIM)
            features[stage, base_input_dim:] = embeddings[stage]
    if measured_state is not None:
        features[0, :state_dim] = measured_state
    return features, embeddings


def euler_zyx_to_rotation(euler_angles):
    """Return the body-to-world rotation for ZYX roll, pitch, yaw angles."""
    roll, pitch, yaw = np.asarray(euler_angles, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def rotation_to_euler_zyx(rotation):
    """Convert a nonsingular body-to-world rotation to ZYX Euler angles."""
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def circle_reference_rotation(args, time_value):
    """Flatness-consistent circle attitude with fixed zero yaw.

    The desired body z axis follows the total specific-force direction
    ``a_d + g e_3``.  A fixed world-x heading completes the desired frame, so
    the resulting ZYX yaw reference remains zero while roll and pitch supply
    the horizontal centripetal acceleration.
    """
    omega = 2.0 * np.pi / args.tracking_period
    phase = omega * time_value
    acceleration = np.array(
        [
            -args.tracking_radius * omega**2 * np.cos(phase),
            -args.tracking_radius * omega**2 * np.sin(phase),
            0.0,
        ],
        dtype=float,
    )
    body_z = acceleration + np.array([0.0, 0.0, GRAVITY])
    body_z /= np.linalg.norm(body_z)
    heading = np.array([1.0, 0.0, 0.0])
    body_y = np.cross(body_z, heading)
    body_y /= np.linalg.norm(body_y)
    body_x = np.cross(body_y, body_z)
    return np.column_stack((body_x, body_y, body_z))


def circle_reference_body_rates(args, time_value):
    """Numerically differentiate the desired rotation into body rates."""
    epsilon = 1e-5
    rotation = circle_reference_rotation(args, time_value)
    rotation_dot = (
        circle_reference_rotation(args, time_value + epsilon)
        - circle_reference_rotation(args, time_value - epsilon)
    ) / (2.0 * epsilon)
    skew = rotation.T @ rotation_dot
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=float)


def attitude_geodesic_error(actual_euler, reference_euler):
    """SO(3) geodesic orientation error in radians."""
    actual_rotation = euler_zyx_to_rotation(actual_euler)
    reference_rotation = euler_zyx_to_rotation(reference_euler)
    cosine = (np.trace(reference_rotation.T @ actual_rotation) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def reference_state(args, time_value, stabilization_goal):
    if args.task == "stabilization":
        return stabilization_goal.copy()

    omega = 2.0 * np.pi / args.tracking_period
    phase = omega * time_value
    reference = np.zeros(12, dtype=float)
    reference[0] = args.tracking_radius * np.cos(phase)
    reference[1] = -args.tracking_radius * omega * np.sin(phase)
    reference[2] = args.tracking_radius * np.sin(phase)
    reference[3] = args.tracking_radius * omega * np.cos(phase)
    reference[4] = args.tracking_altitude
    desired_rotation = circle_reference_rotation(args, time_value)
    reference[6:9] = rotation_to_euler_zyx(desired_rotation)
    # ``circle_reference_rotation`` fixes the ZYX yaw to zero.  Suppress only
    # floating-point roundoff so that the declared task is exact in the CSV.
    reference[8] = 0.0
    reference[9:12] = circle_reference_body_rates(args, time_value)
    return reference


def set_references(solver, args, current_time, stabilization_goal, hover):
    state_dim = int(np.asarray(solver.get(0, "x")).size)
    motor_reference = motor_amplitude_from_thrust(hover)

    def controller_reference(physical_reference):
        if state_dim == RIGID_BODY_STATE_DIM:
            return physical_reference
        return np.concatenate((physical_reference, motor_reference))

    for stage in range(N):
        future_time = current_time + stage * T_HORIZON / N
        state_reference = controller_reference(
            reference_state(args, future_time, stabilization_goal)
        )
        solver.set(stage, "yref", np.concatenate((state_reference, hover)))
    terminal_reference = controller_reference(
        reference_state(args, current_time + T_HORIZON, stabilization_goal)
    )
    solver.set(N, "yref", terminal_reference)


def _make_stem(args, controller, seed):
    method = {
        "nominal": "Nominal_MPC_3D_Final",
        "neural": "RealTime_Neural_MPC_3D",
        "t2s": "RealTime_T2S_MPC_3D",
        "ssi": "SSI_MPC_3D_Final",
        "stgp": "STGP_MPC_3D",
        "knode": "RealTime_KNODE_MPC_3D_TaskMatched",
    }[controller]
    if args.compact_result_filenames and controller == "t2s":
        method = "T2S_MPC_3D"
    if controller == "ssi" and args.ssi_residual_dimension == 6:
        method = (
            "SSI6" if args.compact_result_filenames else "SSI6_MPC_3D"
        )
    scenario_tag = _wind_scenario_file_tag(args)
    if args.compact_result_filenames:
        scenario_tag = {
            "mean_wind_increase": "mwi",
            "turbulence_intensity_increase": "tii",
            "combined_wind_increase": "comb",
            "stationary": "stat",
            "nowind": "nw",
        }[scenario_tag]
    parts = [
        method,
        args.task,
        args.wind,
        scenario_tag,
        f"physics{PHYSICS_FREQUENCY}",
        f"control{args.control_frequency}",
        "a1",
        "ib" if controller == "stgp" else "inputbounds_only",
    ]
    if args.deadline_aware:
        parts.append(
            "rtzoh"
            if args.control_release_mode == "intra_period"
            else "rtboundary"
        )
    if args.task == "circle":
        parts.extend(
            (
                f"r{args.tracking_radius:g}p{args.tracking_period:g}",
                "attref",
            )
        )
    parts.extend(
        (
            f"measd{args.measurement_delay_steps}",
            (
                f"n{args.position_noise_std:g}-{args.velocity_noise_std:g}-"
                f"{args.attitude_noise_std:g}-{args.body_rate_noise_std:g}"
            ),
            f"tau{args.measurement_noise_correlation_time:g}",
        )
    )
    if args.actuator_model == "first_order":
        parts.append(
            "actF"
            f"{1000.0 * args.motor_time_constant:g}"
            f"g{100.0 * args.motor_gain_range:g}"
            f"n{100.0 * args.motor_noise_std:g}"
            f"c{1000.0 * args.motor_noise_correlation_time:g}"
        )
    else:
        parts.append("actideal")
    if args.wind == "dryden":
        parts.extend(
            f"grad{axis}{getattr(args, f'wind_gradient_{axis}'):g}"
            for axis in "xyz"
        )
        parts.append(
            "L" + "-".join(
                f"{getattr(args, f'length_scale_{axis}'):g}"
                for axis in "xyz"
            )
        )
    if controller == "neural":
        parts.extend(
            (f"replay{REPLAY_MAX}", f"batch{args.batch_size}",
             f"update{args.update_every}")
        )
    elif controller == "t2s":
        if args.t2s_replay_mode == "fifo_reservoir":
            parts.append("fr50")
        if args.deadline_aware:
            # Keep diagnostic-image paths below Linux's 255-byte component
            # limit.  The complete configuration remains in the summary CSV.
            parts.extend(
                (
                    f"t2f{args.fast_batch_size}-{args.fast_update_every}-"
                    f"{args.fast_epochs}",
                    f"s{args.slow_batch_size}-{args.slow_update_every}-"
                    f"{args.slow_epochs}",
                )
            )
        else:
            parts.extend(
                (
                    f"r{REPLAY_MAX}", f"tf{TIME_FEAT_DIM}",
                    f"fb{args.fast_batch_size}",
                    f"fu{args.fast_update_every}",
                    f"fe{args.fast_epochs}",
                    f"sb{args.slow_batch_size}",
                    f"su{args.slow_update_every}",
                    f"se{args.slow_epochs}", "nr",
                )
            )
    elif controller == "ssi":
        if args.ssi_thrust_feature == "normalized_command":
            parts.append("featNormCmd")
        if args.deadline_aware:
            parts.extend(
                (
                    f"rff{args.ssi_features}",
                    f"lr{args.ssi_learning_rate:g}",
                    f"k{args.ssi_kernel_std:g}",
                    "u1",
                    "nr",
                )
            )
        else:
            parts.extend(
                (
                    f"rff{args.ssi_features}",
                    f"lr{args.ssi_learning_rate:g}",
                    f"kernel{args.ssi_kernel_std:g}",
                    "online_every1",
                    "noreplay",
                )
            )
    elif controller == "stgp":
        parts.extend(
            (
                f"ncphys{args.stgp_inducing_seed}",
                f"gp{args.stgp_inducing_points}",
                (
                    f"l{args.stgp_spatial_lengthscale:g}-"
                    f"{args.stgp_temporal_lengthscale:g}"
                ),
                (
                    f"v{args.stgp_linear_output_variance:g}-"
                    f"{args.stgp_angular_output_variance:g}"
                ),
                f"n{args.stgp_observation_noise_variance:g}",
            )
        )
    elif controller == "nominal":
        parts.append("nolearning")
    else:
        collection_points = int(
            round(args.collection_duration * args.control_frequency)
        )
        parts.extend(
            (
                "queue3", "noreplay", f"batch{collection_points}",
                f"collect{args.collection_duration:g}s",
                f"delay{args.data_start_delay:g}s",
                f"epochs{args.training_epochs}",
            )
        )
    if args.result_tag:
        parts.append(args.result_tag)
    parts.append(f"seed{seed}")
    return "_".join(parts)


def _wind_scenario(args):
    if args.wind == "none":
        return "nowind"
    mean_wind_increases = any(
        not np.isclose(
            getattr(args, f"mean_wind_{axis}"),
            getattr(args, f"mean_wind_{axis}_end"),
        )
        for axis in "xyz"
    )
    turbulence_intensity_increases = any(
        not np.isclose(
            getattr(args, f"turbulence_sigma_{axis}"),
            getattr(args, f"turbulence_sigma_{axis}_end"),
        )
        for axis in "xyz"
    )
    if mean_wind_increases and turbulence_intensity_increases:
        return "mean_wind_and_turbulence_intensity_increase"
    if mean_wind_increases:
        return "mean_wind_increase"
    if turbulence_intensity_increases:
        return "turbulence_intensity_increase"
    return "stationary"


def _wind_scenario_file_tag(args):
    """Return a filesystem-safe shorthand without changing result metadata."""
    scenario = _wind_scenario(args)
    if scenario == "mean_wind_and_turbulence_intensity_increase":
        return "combined_wind_increase"
    return scenario


def _record_training_result(result, step, activated, activation_time=None, load_ms=np.nan):
    if result.error is not None or result.state_dict is None:
        raise RuntimeError("Background 3-D training failed:\n" + str(result.error))
    record = {
        "version": result.version,
        "trigger_step": result.trigger_step,
        "ready_step": step,
        "model_age_steps": step - result.trigger_step if activated else np.nan,
        "train_ms": 1000.0 * result.train_seconds,
        "trigger_to_ready_ms": 1000.0 * (
            result.ready_wall_time - result.trigger_wall_time
        ),
        "trigger_to_activate_ms": (
            1000.0 * (activation_time - result.trigger_wall_time)
            if activated else np.nan
        ),
        "state_load_ms": load_ms,
        "activated": activated,
    }
    if hasattr(result, "fast_seconds"):
        record.update(
            {
                "contains_fast": bool(result.fast_losses),
                "contains_slow": bool(result.slow_losses),
                "fast_train_ms": 1000.0 * result.fast_seconds,
                "slow_train_ms": 1000.0 * result.slow_seconds,
            }
        )
    return record


def run_once(args, controller, run_id):
    seed = args.seed + run_id
    control_frequency = args.control_frequency
    dt = 1.0 / control_frequency
    physics_steps_per_control = PHYSICS_FREQUENCY // control_frequency
    np.random.seed(seed)
    torch.manual_seed(seed)
    replay_rng = np.random.default_rng(seed + 100_000)
    try:
        pb.resetSimulation()
    except Exception:
        pass

    env = make_environment(args, seed)
    observation, _ = env.reset()
    true_state = np.asarray(observation[:12], dtype=float)
    measurement_std = state_noise_standard_deviations(
        args.position_noise_std,
        args.velocity_noise_std,
        args.attitude_noise_std,
        args.body_rate_noise_std,
    )
    estimator_delay_steps = args.measurement_delay_steps
    state_estimator = DelayedNoisyStateEstimator(
        seed=seed + 200_000,
        dt=dt,
        standard_deviations=measurement_std,
        correlation_time=args.measurement_noise_correlation_time,
        delay_steps=estimator_delay_steps,
    )
    # The 12 physical states are estimated with noise/delay.  For
    # nominal/SSI/STGP/T2S, a command-driven first-order motor observer is
    # appended at the same source timestamp; hidden plant gain/noise and
    # PyBullet motor truth never enter the controller or online labels.
    measurement_packet = state_estimator.reset_packet(true_state)
    measured_physical_state = measurement_packet.state.copy()
    goal = np.asarray(env.X_GOAL, dtype=float)
    hover = np.asarray(env.U_GOAL, dtype=float)
    physics_client, robot_id = get_pb_handles(env)
    mass = get_mass(physics_client, robot_id)
    actuator = make_motor_actuator(
        env, args, PHYSICS_DT, seed + 300_000
    )
    env.set_substep_actuator_model(actuator, hover)
    controller_motor_time_constant = (
        args.motor_time_constant
        if args.actuator_model == "first_order"
        and controller in ("nominal", "ssi", "stgp", "t2s")
        else None
    )
    controller_state_dim = (
        RIGID_BODY_STATE_DIM + 4
        if controller_motor_time_constant is not None
        else RIGID_BODY_STATE_DIM
    )
    initial_motor_amplitude = motor_amplitude_from_thrust(hover)
    if controller_state_dim == RIGID_BODY_STATE_DIM + 4:
        state = np.concatenate(
            (
                measured_physical_state,
                initial_motor_amplitude,
            )
        )
    else:
        state = measured_physical_state.copy()
    model_input_dim = controller_state_dim + 4
    wind_model = (
        make_physical_wind_3d(env, args, PHYSICS_DT, seed)
        if args.wind == "dryden" else None
    )

    residual_model = None
    realtime_model = None
    ssi_features = None
    ssi_learner = None
    stgp_learner = None
    stgp_residual_adapter = None
    if controller == "neural":
        residual_model = MLP(
            input_dim=model_input_dim,
            output_dim=6,
            hidden_dim=64,
            num_layers=3,
        )
    elif controller == "t2s":
        residual_model = TwoSpeedMLP(
            input_dim=model_input_dim + TIME_FEAT_DIM,
            hidden_dim=64,
            output_dim=6,
        )
    elif controller == "knode":
        residual_model = TaskMatchedKNODEResidual()
    if residual_model is not None:
        _zero_initial_residual(residual_model)
        realtime_model = RealTimeL4CasADi(
            residual_model,
            approximation_order=1,
            name=f"rt_{controller}_3d_run{run_id}_seed{seed}",
        )
    if controller == "nominal":
        dynamics = Quadrotor3DNominalDynamics(
            env, motor_time_constant=controller_motor_time_constant
        ).model()
    elif controller == "ssi":
        ssi_features = SSIRandomFeatures.sample(
            seed,
            count=args.ssi_features,
            kernel_std=args.ssi_kernel_std,
            thrust_feature_mode=args.ssi_thrust_feature,
            command_thrust_scale=np.asarray(
                env.physical_action_bounds[1], dtype=float
            ),
        )
        dynamics = Quadrotor3DSSIDynamics(
            env,
            ssi_features,
            motor_time_constant=controller_motor_time_constant,
            residual_dimension=args.ssi_residual_dimension,
        ).model()
    elif controller == "stgp":
        stgp_learner = QuadrotorSpatioTemporalGPLearner(
            state_dimension=controller_state_dim,
            physical_velocity_limit=float(env.MAX_SPEED_KMH) / 3.6,
            physical_attitude_limits=np.maximum(
                np.abs(np.asarray(env.state_space.low[6:9], dtype=float)),
                np.abs(np.asarray(env.state_space.high[6:9], dtype=float)),
            ),
            physical_body_rate_limits=np.maximum(
                np.abs(np.asarray(env.state_space.low[9:12], dtype=float)),
                np.abs(np.asarray(env.state_space.high[9:12], dtype=float)),
            ),
            physical_action_bounds=env.physical_action_bounds,
            inducing_point_seed=args.stgp_inducing_seed,
            control_dt=dt,
            horizon_dt=T_HORIZON / N,
            inducing_point_count=args.stgp_inducing_points,
            spatial_lengthscale=args.stgp_spatial_lengthscale,
            temporal_lengthscale=args.stgp_temporal_lengthscale,
            linear_output_variance=args.stgp_linear_output_variance,
            angular_output_variance=args.stgp_angular_output_variance,
            observation_noise_variance=(
                args.stgp_observation_noise_variance
            ),
        )
        stgp_residual_adapter = QuadrotorSTGPResidualAdapter(stgp_learner)
        dynamics = Quadrotor3DNominalDynamics(
            env, motor_time_constant=controller_motor_time_constant
        ).model()
    elif controller == "t2s":
        dynamics = Quadrotor3DRealTimeT2SDynamics(
            env,
            realtime_model,
            time_feat_dim=TIME_FEAT_DIM,
            motor_time_constant=controller_motor_time_constant,
        ).model()
    else:
        dynamics = Quadrotor3DRealTimeDynamics(env, realtime_model).model()
    dynamics.name = f"quadrotor3D_rt_{controller}_run{run_id}_seed{seed}"
    if controller == "stgp" and N != 20:
        dynamics.name += f"_N{N}"
    nominal_function = cs.Function(
        f"nominal_3d_{controller}_{run_id}_{seed}",
        [dynamics.x, dynamics.u],
        [dynamics.f_nominal],
    )
    transition_buffer = CausalAlignedTransitionBuffer(
        nominal_derivative=nominal_function,
        state_dimension=controller_state_dim,
        derivative_indices=RESIDUAL_DERIVATIVE_INDICES,
        control_dt=dt,
        physics_dt=PHYSICS_DT,
        motor_time_constant=controller_motor_time_constant,
    )
    transition_buffer.reset(
        measurement_packet,
        initial_motor_amplitude=(
            initial_motor_amplitude
            if controller_state_dim == RIGID_BODY_STATE_DIM + 4 else None
        ),
    )
    project_root = Path(__file__).resolve().parents[1]
    mpc_builder = MPC(
        model=dynamics,
        N=N,
        t_horizon=T_HORIZON,
        external_shared_lib_dir=str(
            Path(os.environ.get("ACADOS_SOURCE_DIR", project_root / "external" / "acados")) / "lib"
        ),
        external_shared_lib_name="acados",
    )
    if controller == "stgp":
        # Preserve l4acados' zero-order residual-linearization and one-step
        # SQP-RTI architecture. The online STGP posterior mean/Jacobian enters
        # the OCP. Variance is evaluated separately as a diagnostic because
        # this benchmark intentionally has no chance constraints.
        from l4acados.controllers import ResidualLearningMPC

        ocp = mpc_builder.ocp()
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.levenberg_marquardt = 1e-1
        ocp.solver_options.tol = 1e-4
        residual_map = discrete_acceleration_residual_map(
            controller_state_dim, T_HORIZON / N
        )
        build_tag = f"stgp_noncautious_run{run_id}_seed{seed}"
        if N != 20:
            build_tag += f"_N{N}"
        build_directory = project_root / "c_generated_code" / build_tag
        ocp.code_export_directory = str(build_directory)
        solver = ResidualLearningMPC(
            ocp=ocp,
            B=residual_map,
            residual_model=stgp_residual_adapter,
            use_cython=True,
            path_json_ocp=str(
                project_root / "c_generated_code" / f"{build_tag}_ocp.json"
            ),
            path_json_sim=str(
                project_root / "c_generated_code" / f"{build_tag}_sim.json"
            ),
            build_c_code=True,
        )
    else:
        solver = mpc_builder.solver
    initialize_solver(solver, state, env)

    trainer = None
    if controller == "ssi":
        ssi_continuous = cs.Function(
            f"ssi_continuous_{run_id}_{seed}",
            [dynamics.x, dynamics.u, dynamics.p],
            [dynamics.f_expl],
        )
        ssi_learner = SSIOnlineLearner(
            ssi_features,
            ssi_continuous,
            learning_rate=args.ssi_learning_rate,
            residual_dimension=args.ssi_residual_dimension,
        )
    elif controller == "neural":
        trainer = AsyncRealtimeTrainer(
            residual_model,
            input_dim=model_input_dim,
            output_dim=6,
            hidden_dim=64,
            num_layers=3,
            learning_rate=1e-3,
            epochs=args.training_epochs,
            torch_threads=args.torch_threads,
        )
    elif controller == "t2s":
        trainer = AsyncRealtimeT2STrainer(
            residual_model,
            input_dim=model_input_dim + TIME_FEAT_DIM,
            hidden_dim=64,
            output_dim=6,
            fast_learning_rate=1e-3,
            slow_learning_rate=1e-3,
            fast_epochs=args.fast_epochs,
            slow_epochs=args.slow_epochs,
            torch_threads=args.torch_threads,
            cpu_core=args.trainer_cpu_core,
        )
    elif controller == "knode":
        trainer = AsyncRealtimeKNODETrainer(
            residual_model,
            epochs=args.training_epochs,
            learning_rate=1e-2,
            regularization=1e-7,
            torch_threads=args.torch_threads,
        )

    if residual_model is not None:
        warmup, _ = horizon_features(
            solver, 0.0, controller, measured_state=state
        )
        _, warm_values, warm_jacobians = first_order_parameters(
            residual_model, warmup
        )
        evaluate_first_order(warmup, warm_values, warm_jacobians, warmup)
        exact_model_output(residual_model, warmup)
        if controller != "knode":
            relu_activation_patterns(residual_model, warmup)

    steps = int(args.duration / dt)
    states = [true_state.copy()]
    measured_states = [measured_physical_state.copy()]
    controls = []
    candidate_controls = []
    applied_controls = []
    actuator_substep_records = []
    references = []
    wind_velocities = []
    wind_forces = []
    wind_torques = []
    wind_means = []
    wind_sigmas = []
    wind_substep_records = []
    replay_inputs = []
    replay_targets = []
    hybrid_replay = (
        RecentReservoirReplay(50, 50, seed=seed + 600_000)
        if controller == "t2s" and args.t2s_replay_mode == "fifo_reservoir"
        else None
    )
    replay_batch_records = []
    timing_records = []
    accuracy_records = []
    training_records = []
    neural_losses = []
    fast_losses = []
    slow_losses = []
    knode_losses = []
    ssi_update_records = []
    stgp_update_records = []
    residual_transition_records = []
    ssi_pending_transition = None
    stgp_pending_input = None
    stgp_pending_target = None
    stgp_pending_time = None
    deadline_release = (
        ComputationAwareControlRelease(
            hover, mode=args.control_release_mode
        )
        if args.deadline_aware else None
    )
    knode_inputs = []
    knode_targets = []
    knode_collection_steps = (
        int(round(args.collection_duration * control_frequency))
        if controller == "knode" else 0
    )
    knode_start_step = (
        int(round(args.data_start_delay * control_frequency))
        if controller == "knode" else 0
    )
    submitted = {"neural": 0, "fast": 0, "slow": 0, "knode": 0}
    skipped = {"neural": 0, "fast": 0, "slow": 0, "knode": 0}
    active_version = 0
    next_version = 1
    gc_was_enabled = gc.isenabled()
    # Cyclic GC can pause a Python control loop for hundreds of milliseconds
    # after enough short-lived CasADi/PyTorch wrapper objects accumulate.  All
    # one-time allocation and autodiff warmup is complete at this point.  Keep
    # deterministic reference-count cleanup during flight, defer cyclic GC,
    # and restore the interpreter setting at shutdown.
    if gc_was_enabled:
        gc.disable()
    run_start = time.perf_counter()

    try:
        for step in range(steps):
            target_start = run_start + step * dt
            remaining = target_start - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            step_start = time.perf_counter()
            start_lateness_ms = 1000.0 * max(0.0, step_start - target_start)

            completed = trainer.poll() if trainer is not None else None
            if completed is not None:
                if completed.error is not None or completed.state_dict is None:
                    raise RuntimeError(
                        "Background 3-D training failed:\n" + str(completed.error)
                    )
                load_start = time.perf_counter()
                load_numpy_state_dict(residual_model, completed.state_dict)
                activation_time = time.perf_counter()
                training_records.append(
                    _record_training_result(
                        completed, step, True, activation_time,
                        1000.0 * (activation_time - load_start),
                    )
                )
                active_version = completed.version
                if controller == "neural":
                    neural_losses.extend(completed.losses)
                elif controller == "t2s":
                    fast_losses.extend(completed.fast_losses)
                    slow_losses.extend(completed.slow_losses)
                else:
                    knode_losses.extend(completed.losses)

            current_time = step * dt
            parameter_start = time.perf_counter()
            online_update_ms = 0.0
            ssi_error_norm = np.nan
            stgp_error_norm = np.nan
            stgp_inference_ms = np.nan
            stgp_variance_ms = np.nan
            stgp_posterior_mean_norm = np.nan
            stgp_update_target = np.full(6, np.nan)
            stgp_update_prediction = np.full(6, np.nan)
            stgp_update_error = np.full(6, np.nan)
            if residual_model is not None:
                expansion, embeddings = horizon_features(
                    solver, current_time, controller, measured_state=state
                )
                parameters, expansion_values, jacobians = first_order_parameters(
                    residual_model, expansion
                )
                for stage in range(N):
                    packed = parameters[stage]
                    if controller == "t2s":
                        packed = np.concatenate((embeddings[stage], packed))
                    solver.set(stage, "p", packed)
            elif controller == "ssi":
                update_start = time.perf_counter()
                if ssi_pending_transition is not None:
                    ssi_error = ssi_learner.update_residual_target(
                        ssi_pending_transition.state,
                        ssi_pending_transition.equivalent_control,
                        ssi_pending_transition.target,
                    )
                    ssi_error_norm = float(np.linalg.norm(ssi_error))
                    ssi_pending_transition = None
                online_update_ms = 1000.0 * (
                    time.perf_counter() - update_start
                )
                ssi_parameter = ssi_learner.parameter_vector
                for stage in range(N):
                    solver.set(stage, "p", ssi_parameter)
                ssi_update_records.append(
                    {
                        "step": step,
                        "time": current_time,
                        "update_ms": online_update_ms,
                        "prediction_error_l2": ssi_error_norm,
                        "alpha_l2": float(np.linalg.norm(ssi_learner.alpha)),
                        "alpha_max_abs": float(
                            np.max(np.abs(ssi_learner.alpha))
                        ),
                    }
                )
            elif controller == "stgp":
                update_start = time.perf_counter()
                if stgp_pending_input is not None:
                    stgp_error = stgp_learner.update(
                        stgp_pending_input,
                        stgp_pending_target,
                        stgp_pending_time,
                    )
                    stgp_update_target = stgp_pending_target.copy()
                    stgp_update_error = stgp_error.copy()
                    stgp_update_prediction = (
                        stgp_update_target + stgp_update_error
                    )
                    stgp_error_norm = float(np.linalg.norm(stgp_error))
                    stgp_pending_input = None
                    stgp_pending_target = None
                    stgp_pending_time = None
                online_update_ms = 1000.0 * (
                    time.perf_counter() - update_start
                )
                stgp_residual_adapter.set_query_time(current_time)
            parameter_ms = 1000.0 * (time.perf_counter() - parameter_start)

            set_references(solver, args, current_time, goal, hover)
            solver.set(0, "lbx", state)
            solver.set(0, "ubx", state)
            solve_start = time.perf_counter()
            if controller == "stgp":
                # l4acados separates residual/nominal linearization in the RTI
                # preparation phase from the feedback QP. Exactly one
                # preparation/feedback iteration is used.
                solver.preparation()
                solver_status = solver.feedback()
                stgp_inference_ms = 1000.0 * float(
                    solver.time_residual[solver.num_iter]
                )
                stgp_posterior_mean_norm = float(
                    np.linalg.norm(
                        stgp_residual_adapter.current_prediction[0]
                    )
                )
            else:
                solver_status = solver.solve()
            solve_ms = 1000.0 * (time.perf_counter() - solve_start)
            candidate_control = np.clip(
                solver.get(0, "u"), dynamics.u_min, dynamics.u_max
            )
            control_ready = time.perf_counter()
            critical_ms = 1000.0 * (control_ready - step_start)
            release_latency_ms = 1000.0 * (control_ready - target_start)
            if deadline_release is not None:
                release_decision = deadline_release.release(
                    candidate_control,
                    compute_ms=critical_ms,
                    deadline_ms=1000.0 * dt,
                    substep_ms=1000.0 * PHYSICS_DT,
                    substep_count=physics_steps_per_control,
                    candidate_step=step,
                    solver_success=(solver_status == 0),
                )
                control = release_decision.applied_control
                candidate_accepted = release_decision.candidate_accepted
                candidate_discarded = release_decision.candidate_discarded
                applied_control_source_step = (
                    release_decision.applied_source_step
                )
                applied_control_age_steps = release_decision.applied_age_steps
                command_schedule = release_decision.command_schedule
                held_substeps = release_decision.held_substeps
                candidate_substeps = release_decision.candidate_substeps
            else:
                control = candidate_control.copy()
                candidate_accepted = solver_status == 0
                candidate_discarded = False
                applied_control_source_step = step
                applied_control_age_steps = 0
                command_schedule = None
                held_substeps = 0
                candidate_substeps = physics_steps_per_control
            candidate_controls.append(candidate_control.copy())
            controls.append(control.copy())

            diagnostic_start = time.perf_counter()
            if residual_model is not None:
                query, _ = horizon_features(solver, current_time, controller)
                approximation = evaluate_first_order(
                    expansion, expansion_values, jacobians, query
                )
                exact = exact_model_output(residual_model, query)
                error = approximation - exact
                if controller == "knode":
                    # Tanh has no piecewise-linear activation regions.  Taylor
                    # accuracy is still measured directly by ``error_l2``.
                    crossed = np.zeros(N, dtype=bool)
                else:
                    crossed = np.any(
                        relu_activation_patterns(residual_model, expansion)
                        != relu_activation_patterns(residual_model, query), axis=1
                    )
                relative = np.linalg.norm(error, axis=1) / np.maximum(
                    np.linalg.norm(exact, axis=1), 1e-6
                )
                for node in range(N):
                    record = {
                        "step": step,
                        "node": node,
                        "model_version": active_version,
                        "delta_input_l2": np.linalg.norm(
                            query[node] - expansion[node]
                        ),
                        "error_l2": np.linalg.norm(error[node]),
                        "relative_error_l2": relative[node],
                        "relu_region_crossed": crossed[node],
                    }
                    for index, name in enumerate(RESIDUAL_NAMES):
                        record[f"exact_{name}"] = exact[node, index]
                        record[f"approx_{name}"] = approximation[node, index]
                        record[f"error_{name}"] = error[node, index]
                    accuracy_records.append(record)
            diagnostic_ms = 1000.0 * (time.perf_counter() - diagnostic_start)

            substep_samples = []
            if wind_model is not None:
                def substep_wind(substep):
                    _, orientation = pb.getBasePositionAndOrientation(
                        robot_id, physicsClientId=physics_client
                    )
                    velocity, angular_velocity = pb.getBaseVelocity(
                        robot_id, physicsClientId=physics_client
                    )
                    sample = wind_model.step(
                        velocity,
                        orientation,
                        env.current_applied_thrust,
                        angular_velocity=angular_velocity,
                    )
                    substep_samples.append(sample)
                    substep_index = step * physics_steps_per_control + substep
                    wind_substep_records.append(
                        {
                            "time": substep_index * PHYSICS_DT,
                            "control_step": step,
                            "physics_substep": substep,
                            **{
                                f"wind_v{axis}": sample.velocity[index]
                                for index, axis in enumerate("xyz")
                            },
                            **{
                                f"wind_f{axis}": sample.force[index]
                                for index, axis in enumerate("xyz")
                            },
                            **{
                                f"wind_tau_{axis}": sample.torque_body[index]
                                for index, axis in enumerate("xyz")
                            },
                            **{
                                f"wind_mean_v{axis}": sample.local_mean[index]
                                for index, axis in enumerate("xyz")
                            },
                            **{
                                f"wind_sigma_{axis}": sample.local_sigma[index]
                                for index, axis in enumerate("xyz")
                            },
                        }
                    )
                    return sample

                env.set_substep_disturbance_callback(substep_wind)
            else:
                env.set_substep_disturbance_callback(None)
            next_observation, _, _, _ = env.step(
                control, command_schedule=command_schedule
            )
            actuator_samples = env.last_actuator_samples
            if len(actuator_samples) != physics_steps_per_control:
                raise RuntimeError(
                    "Actuator model did not run once per physics substep: "
                    f"expected {physics_steps_per_control}, got "
                    f"{len(actuator_samples)}"
                )
            command_thrust_schedule = np.stack(
                [sample.command_thrust for sample in actuator_samples]
            )
            transition_buffer.record_command_interval(
                step, command_thrust_schedule
            )
            applied_controls.append(
                np.mean(
                    [sample.applied_thrust for sample in actuator_samples],
                    axis=0,
                )
            )
            for substep, sample in enumerate(actuator_samples):
                substep_index = step * physics_steps_per_control + substep
                record = {
                    "time": substep_index * PHYSICS_DT,
                    "control_step": step,
                    "physics_substep": substep,
                }
                for motor in range(4):
                    index = motor + 1
                    record.update(
                        {
                            f"u{index}_command_N": sample.command_thrust[motor],
                            f"rpm{index}_command": sample.command_rpm[motor],
                            f"rpm{index}_lagged": sample.lagged_rpm[motor],
                            f"u{index}_lagged_N": sample.lagged_thrust[motor],
                            f"gain{index}": sample.motor_gain[motor],
                            f"noise{index}": sample.relative_noise[motor],
                            f"u{index}_applied_N": sample.applied_thrust[motor],
                            f"rpm{index}_applied": sample.applied_rpm[motor],
                        }
                    )
                actuator_substep_records.append(record)
            true_state = np.asarray(next_observation[:12], dtype=float)
            measurement_packet = state_estimator.observe_packet(true_state)
            measured_physical_state = measurement_packet.state.copy()
            transition_buffer.add_measurement(measurement_packet)
            state = transition_buffer.state_for_packet(measurement_packet)
            ready_transitions = transition_buffer.pop_ready()

            if controller == "stgp":
                # This diagnostic is deliberately outside the control-critical
                # path: posterior variance is not consumed by this OCP. A
                # single current-query value is sufficient to report online
                # uncertainty without evaluating unused horizon backoffs.
                variance_start = time.perf_counter()
                stgp_variance = (
                    stgp_residual_adapter.evaluate_current_variance()
                )
                stgp_variance_ms = 1000.0 * (
                    time.perf_counter() - variance_start
                )
                stgp_stdev = np.sqrt(np.maximum(stgp_variance, 0.0))
                stgp_update_records.append(
                    {
                        "step": step,
                        "time": current_time,
                        "update_ms": online_update_ms,
                        "inference_ms": stgp_inference_ms,
                        "variance_diagnostic_ms": stgp_variance_ms,
                        "prediction_error_l2": stgp_error_norm,
                        "posterior_mean_l2": stgp_posterior_mean_norm,
                        "posterior_stdev_l2": float(
                            np.linalg.norm(stgp_stdev[0])
                        ),
                        "observations": stgp_learner.update_count,
                        **{
                            f"target_{name}": stgp_update_target[index]
                            for index, name in enumerate(RESIDUAL_NAMES)
                        },
                        **{
                            f"prediction_{name}": (
                                stgp_update_prediction[index]
                            )
                            for index, name in enumerate(RESIDUAL_NAMES)
                        },
                        **{
                            f"error_{name}": stgp_update_error[index]
                            for index, name in enumerate(RESIDUAL_NAMES)
                        },
                        **{
                            f"mpc_mean_{name}": (
                                stgp_residual_adapter.current_prediction[0, index]
                            )
                            for index, name in enumerate(RESIDUAL_NAMES)
                        },
                        **{
                            f"posterior_stdev_{name}": stgp_stdev[0, index]
                            for index, name in enumerate(RESIDUAL_NAMES)
                        },
                    }
                )
            states.append(true_state.copy())
            measured_states.append(measured_physical_state.copy())
            references.append(reference_state(args, current_time + dt, goal))

            if substep_samples:
                wind_velocities.append(
                    np.mean([sample.velocity for sample in substep_samples], axis=0)
                )
                wind_forces.append(
                    np.mean([sample.force for sample in substep_samples], axis=0)
                )
                wind_torques.append(
                    np.mean(
                        [sample.torque_body for sample in substep_samples],
                        axis=0,
                    )
                )
                wind_means.append(
                    np.mean([sample.local_mean for sample in substep_samples], axis=0)
                )
                wind_sigmas.append(
                    np.mean([sample.local_sigma for sample in substep_samples], axis=0)
                )
            else:
                wind_velocities.append(np.zeros(3))
                wind_forces.append(np.zeros(3))
                wind_torques.append(np.zeros(3))
                wind_means.append(np.zeros(3))
                wind_sigmas.append(np.zeros(3))

            for transition in ready_transitions:
                network_parts = [
                    transition.state,
                    transition.equivalent_control,
                ]
                if controller == "t2s":
                    network_parts.append(
                        time_embedding_np(
                            transition.source_time / TIME_SCALE,
                            d=TIME_FEAT_DIM,
                        )
                    )
                sample_input = np.concatenate(network_parts).astype(np.float32)
                sample_target = transition.target.astype(np.float32)

                command_changes = int(
                    np.count_nonzero(
                        np.any(
                            np.diff(transition.command_schedule, axis=0)
                            != 0.0,
                            axis=1,
                        )
                    )
                )
                record = {
                    "source_step": transition.source_step,
                    "source_time": transition.source_time,
                    "end_step": transition.end_step,
                    "end_time": transition.end_time,
                    "available_step": transition.available_step,
                    "available_time": transition.available_time,
                    "start_arrival_step": transition.start_arrival_step,
                    "end_arrival_step": transition.end_arrival_step,
                    "label_delay_steps": transition.label_delay_steps,
                    "label_delay_seconds": transition.label_delay_seconds,
                    "physics_substeps": transition.command_schedule.shape[0],
                    "command_changes": command_changes,
                }
                for motor in range(4):
                    name = motor + 1
                    record[f"u{name}_first_N"] = (
                        transition.command_schedule[0, motor]
                    )
                    record[f"u{name}_last_N"] = (
                        transition.command_schedule[-1, motor]
                    )
                    record[f"u{name}_equivalent_N"] = (
                        transition.equivalent_control[motor]
                    )
                for index, name in enumerate(RESIDUAL_NAMES):
                    derivative_index = RESIDUAL_DERIVATIVE_INDICES[index]
                    record[f"target_{name}"] = transition.target[index]
                    record[f"observed_next_{name}"] = (
                        transition.next_state[derivative_index]
                    )
                    record[f"nominal_next_{name}"] = (
                        transition.nominal_next_state[derivative_index]
                    )
                residual_transition_records.append(record)

                if controller == "knode":
                    if transition.end_step >= knode_start_step:
                        knode_inputs.append(sample_input)
                        knode_targets.append(sample_target)
                elif controller == "ssi":
                    ssi_pending_transition = transition
                elif controller == "stgp":
                    stgp_pending_input = sample_input.astype(float, copy=True)
                    stgp_pending_target = sample_target.astype(float, copy=True)
                    stgp_pending_time = transition.source_time
                elif controller in ("neural", "t2s"):
                    if hybrid_replay is not None:
                        hybrid_replay.append(
                            sample_input, sample_target,
                            transition.source_step, transition.source_time,
                        )
                    else:
                        replay_inputs.append(sample_input)
                        replay_targets.append(sample_target)
                        if len(replay_inputs) > REPLAY_MAX:
                            replay_inputs.pop(0)
                            replay_targets.pop(0)

            if controller == "neural":
                due = (
                    step > 0 and step % args.update_every == 0
                    and step + args.update_every < steps
                    and len(replay_inputs) >= args.batch_size
                )
                if due:
                    job = RealtimeTrainingJob(
                        version=next_version,
                        trigger_step=step,
                        trigger_wall_time=time.perf_counter(),
                        inputs=np.asarray(replay_inputs[-args.batch_size:]).copy(),
                        targets=np.asarray(replay_targets[-args.batch_size:]).copy(),
                    )
                    if trainer.submit(job):
                        submitted["neural"] += 1
                        next_version += 1
                    else:
                        skipped["neural"] += 1
            elif controller == "t2s":
                fast_samples_ready = (
                    len(hybrid_replay.recent) >= args.fast_batch_size
                    if hybrid_replay is not None
                    else len(replay_inputs) >= args.fast_batch_size
                )
                slow_samples_ready = (
                    hybrid_replay.can_sample(
                        args.slow_batch_size // 2, args.slow_batch_size // 2
                    )
                    if hybrid_replay is not None
                    else len(replay_inputs) >= args.slow_batch_size
                )
                fast_due = (
                    step > 0 and step % args.fast_update_every == 0
                    and step + args.fast_update_every < steps
                    and fast_samples_ready
                )
                slow_due = (
                    step > 0 and step % args.slow_update_every == 0
                    and step + args.slow_update_every < steps
                    and slow_samples_ready
                )
                # Reserve the worker immediately before a slow boundary.  With
                # fast=3 and slow=50, step 99 would otherwise launch a fast
                # job that is still in flight when the slow job becomes due at
                # step 100.  Slow representation learning has priority over
                # one output-layer refresh at that boundary.
                steps_to_slow = (-step) % args.slow_update_every
                if fast_due and 0 < steps_to_slow <= 1:
                    fast_due = False
                if fast_due or slow_due:
                    fast_inputs = fast_targets = slow_inputs = slow_targets = None
                    fast_entries = recent_entries = history_entries = []
                    if fast_due:
                        if hybrid_replay is not None:
                            fast_entries = hybrid_replay.latest(args.fast_batch_size)
                            fast_inputs, fast_targets = hybrid_replay.arrays(fast_entries)
                        else:
                            fast_inputs = np.asarray(
                                replay_inputs[-args.fast_batch_size:]
                            ).copy()
                            fast_targets = np.asarray(
                                replay_targets[-args.fast_batch_size:]
                            ).copy()
                    if slow_due:
                        if hybrid_replay is not None:
                            recent_entries, history_entries = hybrid_replay.sample(
                                replay_rng, args.slow_batch_size // 2,
                                args.slow_batch_size // 2,
                            )
                            slow_inputs, slow_targets = hybrid_replay.arrays(
                                recent_entries + history_entries
                            )
                        else:
                            slow_inputs, slow_targets = sample_replay_without_replacement(
                                replay_rng,
                                replay_inputs,
                                replay_targets,
                                args.slow_batch_size,
                            )
                    job = RealtimeT2SJob(
                        version=next_version,
                        trigger_step=step,
                        trigger_wall_time=time.perf_counter(),
                        fast_inputs=fast_inputs,
                        fast_targets=fast_targets,
                        slow_inputs=slow_inputs,
                        slow_targets=slow_targets,
                    )
                    accepted_job = trainer.submit(job)
                    if hybrid_replay is not None:
                        batch_record = {
                            "version": job.version,
                            "trigger_step": step,
                            "submitted": accepted_job,
                            "recent_size": len(hybrid_replay.recent),
                            "reservoir_size": len(hybrid_replay.history),
                            "history_seen": hybrid_replay.history_seen,
                            "recent_oldest_source_step": hybrid_replay.recent[0].source_step,
                            "recent_newest_source_step": hybrid_replay.recent[-1].source_step,
                        }
                        for batch_name, entries in (
                            ("fast", fast_entries), ("slow_recent", recent_entries),
                            ("slow_history", history_entries),
                        ):
                            batch_record[f"{batch_name}_count"] = len(entries)
                            batch_record[f"{batch_name}_source_steps"] = ";".join(
                                str(entry.source_step) for entry in entries
                            )
                            batch_record[f"{batch_name}_source_times_s"] = ";".join(
                                f"{entry.source_time:.8f}" for entry in entries
                            )
                        replay_batch_records.append(batch_record)
                    if accepted_job:
                        submitted["fast"] += int(fast_due)
                        submitted["slow"] += int(slow_due)
                        next_version += 1
                    else:
                        skipped["fast"] += int(fast_due)
                        skipped["slow"] += int(slow_due)
            elif controller == "knode":
                if len(knode_inputs) == knode_collection_steps:
                    job = RealtimeKNODEJob(
                        version=next_version,
                        trigger_step=step,
                        trigger_wall_time=time.perf_counter(),
                        inputs=np.asarray(knode_inputs, dtype=np.float32),
                        targets=np.asarray(knode_targets, dtype=np.float32),
                    )
                    # KNODE consumes each fresh window once and never places
                    # it in a replay buffer.
                    knode_inputs.clear()
                    knode_targets.clear()
                    if trainer.submit(job):
                        submitted["knode"] += 1
                        next_version += 1
                    else:
                        skipped["knode"] += 1

            timing_records.append(
                {
                    "step": step,
                    "model_version": active_version,
                    "parameter_ms": parameter_ms,
                    "online_update_ms": online_update_ms,
                    "ssi_prediction_error_l2": ssi_error_norm,
                    "stgp_prediction_error_l2": stgp_error_norm,
                    "stgp_inference_ms": stgp_inference_ms,
                    "solve_ms": solve_ms,
                    "control_critical_ms": critical_ms,
                    "diagnostic_ms": diagnostic_ms,
                    "loop_ms": 1000.0 * (time.perf_counter() - step_start),
                    "start_lateness_ms": start_lateness_ms,
                    "control_release_latency_ms": release_latency_ms,
                    "candidate_accepted": candidate_accepted,
                    "candidate_discarded": candidate_discarded,
                    "applied_control_source_step": applied_control_source_step,
                    "applied_control_age_steps": applied_control_age_steps,
                    "held_control_substeps": held_substeps,
                    "candidate_control_substeps": candidate_substeps,
                    # Backward-compatible primary deadline definition: the
                    # end-to-end computation performed inside this control
                    # step.  Wall-clock release lateness is retained
                    # separately so OS scheduling does not get attributed to
                    # the controller algorithm.
                    "deadline_miss": critical_ms > 1000.0 * dt,
                    "release_deadline_miss": (
                        release_latency_ms > 1000.0 * dt
                    ),
                    "solver_status": solver_status,
                }
            )
    finally:
        pending = trainer.finish() if trainer is not None else []
        for completed in pending:
            training_records.append(
                _record_training_result(completed, np.nan, False)
            )
            if controller == "neural":
                neural_losses.extend(completed.losses)
            elif controller == "t2s":
                fast_losses.extend(completed.fast_losses)
                slow_losses.extend(completed.slow_losses)
            else:
                knode_losses.extend(completed.losses)
        env.close()
        if gc_was_enabled:
            gc.enable()

    states_array = np.asarray(states[1:])
    measured_states_array = np.asarray(measured_states[1:])
    controls_array = np.asarray(controls)
    candidate_controls_array = np.asarray(candidate_controls)
    applied_controls_array = np.asarray(applied_controls)
    wind_velocity = np.asarray(wind_velocities)
    wind_force = np.asarray(wind_forces)
    wind_torque = np.asarray(wind_torques)
    wind_mean = np.asarray(wind_means)
    wind_sigma = np.asarray(wind_sigmas)
    timing = pd.DataFrame(timing_records)
    accuracy = pd.DataFrame(accuracy_records)
    training = pd.DataFrame(training_records)
    ssi_updates = pd.DataFrame(ssi_update_records)
    stgp_updates = pd.DataFrame(stgp_update_records)
    residual_transitions = pd.DataFrame(residual_transition_records)
    wind_substeps = pd.DataFrame(wind_substep_records)
    actuator_substeps = pd.DataFrame(actuator_substep_records)
    references_array = np.asarray(references)
    actuation_error = applied_controls_array - controls_array
    position = states_array[:, [0, 2, 4]]
    reference_position = references_array[:, [0, 2, 4]]
    position_error = np.linalg.norm(position - reference_position, axis=1)
    attitude_error = np.array(
        [
            attitude_geodesic_error(actual[6:9], reference[6:9])
            for actual, reference in zip(states_array, references_array)
        ],
        dtype=float,
    )
    yaw_error = wrap_angles(states_array[:, 8] - references_array[:, 8])
    estimation_error = measured_states_array - states_array
    estimation_error[:, 6:9] = wrap_angles(estimation_error[:, 6:9])

    trajectory_data = {"time": (np.arange(steps) + 1) * dt}
    trajectory_data.update(
        {name: states_array[:, index] for index, name in enumerate(STATE_NAMES)}
    )
    trajectory_data.update(
        {
            f"{name}_measured": measured_states_array[:, index]
            for index, name in enumerate(STATE_NAMES)
        }
    )
    trajectory_data.update(
        {
            f"{name}_ref": references_array[:, index]
            for index, name in enumerate(STATE_NAMES)
        }
    )
    trajectory_data.update(
        {f"u{index + 1}": controls_array[:, index] for index in range(4)}
    )
    trajectory_data.update(
        {
            f"u{index + 1}_candidate": candidate_controls_array[:, index]
            for index in range(4)
        }
    )
    trajectory_data.update(
        {
            f"u{index + 1}_applied": applied_controls_array[:, index]
            for index in range(4)
        }
    )
    trajectory_data.update(
        {
            f"u{index + 1}_actuation_error": (
                applied_controls_array[:, index] - controls_array[:, index]
            )
            for index in range(4)
        }
    )
    for index, axis in enumerate("xyz"):
        trajectory_data[f"wind_v{axis}"] = wind_velocity[:, index]
        trajectory_data[f"wind_f{axis}"] = wind_force[:, index]
        trajectory_data[f"wind_tau_{axis}"] = wind_torque[:, index]
        trajectory_data[f"wind_mean_v{axis}"] = wind_mean[:, index]
        trajectory_data[f"wind_sigma_{axis}"] = wind_sigma[:, index]
    trajectory_data["position_error_xyz"] = position_error
    trajectory_data["attitude_error"] = attitude_error
    trajectory_data["yaw_error"] = yaw_error
    trajectory_data["position_estimation_error"] = np.linalg.norm(
        estimation_error[:, [0, 2, 4]], axis=1
    )
    trajectory_data["velocity_estimation_error"] = np.linalg.norm(
        estimation_error[:, [1, 3, 5]], axis=1
    )
    trajectory_data["attitude_estimation_error"] = np.linalg.norm(
        estimation_error[:, 6:9], axis=1
    )
    trajectory_data["body_rate_estimation_error"] = np.linalg.norm(
        estimation_error[:, 9:12], axis=1
    )
    trajectory_data["model_version"] = timing.model_version.to_numpy()
    trajectory_data["control_compute_ms"] = timing.control_critical_ms.to_numpy()
    trajectory = pd.DataFrame(trajectory_data)

    result_folder = (
        "results_stabilization_3d"
        if args.task == "stabilization"
        else "results_tracking_3d"
    )
    output_dir = (
        args.output_dir if args.output_dir is not None
        else Path(__file__).resolve().parents[1] / result_folder
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / _make_stem(args, controller, seed)
    trajectory.to_csv(str(stem) + ".csv", index=False)
    timing.to_csv(str(stem) + "_timing.csv", index=False)
    accuracy.to_csv(str(stem) + "_taylor_accuracy.csv", index=False)
    training.to_csv(str(stem) + "_training_timing.csv", index=False)
    if hybrid_replay is not None:
        pd.DataFrame(replay_batch_records).to_csv(str(stem) + "_replay_batches.csv", index=False)
    ssi_updates.to_csv(str(stem) + "_ssi_updates.csv", index=False)
    stgp_updates.to_csv(str(stem) + "_stgp_updates.csv", index=False)
    residual_transitions.to_csv(
        str(stem) + "_residual_transitions.csv", index=False
    )
    wind_substeps.to_csv(str(stem) + "_wind_500hz.csv", index=False)
    actuator_substeps.to_csv(
        str(stem) + "_actuator_500hz.csv", index=False
    )

    summary = pd.DataFrame(
        [{
            "controller": controller,
            "seed": seed,
            "result_tag": args.result_tag,
            "steps": steps,
            "task": args.task,
            "wind_scenario": _wind_scenario(args),
            "tracking_period_s": (
                args.tracking_period if args.task == "circle" else np.nan
            ),
            "tracking_radius_m": (
                args.tracking_radius if args.task == "circle" else np.nan
            ),
            "tracking_altitude_m": (
                args.tracking_altitude if args.task == "circle" else np.nan
            ),
            "circle_speed_mps": (
                args.tracking_radius * 2.0 * np.pi / args.tracking_period
                if args.task == "circle" else np.nan
            ),
            "circle_centripetal_acceleration_mps2": (
                args.tracking_radius * (2.0 * np.pi / args.tracking_period) ** 2
                if args.task == "circle" else np.nan
            ),
            "circle_nominal_tilt_rad": (
                np.arctan2(
                    args.tracking_radius
                    * (2.0 * np.pi / args.tracking_period) ** 2,
                    GRAVITY,
                )
                if args.task == "circle" else np.nan
            ),
            "yaw_reference_rad": 0.0 if args.task == "circle" else np.nan,
            "attitude_reference_method": (
                "flatness_total_specific_force_fixed_yaw"
                if args.task == "circle" else "stabilization_goal"
            ),
            "attitude_error_metric": "SO3_geodesic_rad",
            "control_state_source": "delayed_noisy_state_estimate",
            "evaluation_state_source": "pybullet_ground_truth",
            "deadline_aware_control": args.deadline_aware,
            "control_release_policy": (
                (
                    "intra_period_zoh_apply_on_completion_"
                    "discard_after_deadline"
                )
                if (
                    args.deadline_aware
                    and args.control_release_mode == "intra_period"
                )
                else (
                    "period_boundary_zoh_publish_next_period_"
                    "discard_after_deadline"
                )
                if args.deadline_aware
                else "synchronous_zero_latency"
            ),
            "control_command_delay_steps": (
                np.nan if args.deadline_aware else 0
            ),
            "control_latency_resolution_ms": (
                1000.0 * PHYSICS_DT if args.deadline_aware else 0.0
            ),
            "measurement_delay_steps": args.measurement_delay_steps,
            "measurement_delay_ms": 1000.0 * args.measurement_delay_steps * dt,
            "effective_measurement_delay_steps": estimator_delay_steps,
            "effective_measurement_delay_ms": 1000.0 * estimator_delay_steps * dt,
            "residual_label_protocol": (
                "source_timestamp_aligned_discrete_nominal_transition"
            ),
            "residual_label_command_history": (
                "actual_sent_command_at_each_500hz_substep"
            ),
            "residual_label_nominal_integrator": "rk4_500hz",
            "residual_label_samples": len(residual_transitions),
            "residual_label_delay_steps": (
                float(residual_transitions.label_delay_steps.mean())
                if len(residual_transitions) else np.nan
            ),
            "learner_motor_state_source": (
                "causal_nominal_command_driven_observer"
                if controller_state_dim == RIGID_BODY_STATE_DIM + 4
                else "not_in_controller_model"
            ),
            "ssi_parameter_packing": (
                "casadi_column_major_matched"
            ) if controller == "ssi" else "not_applicable",
            "position_noise_std_m": args.position_noise_std,
            "velocity_noise_std_mps": args.velocity_noise_std,
            "attitude_noise_std_rad": args.attitude_noise_std,
            "body_rate_noise_std_radps": args.body_rate_noise_std,
            "measurement_noise_correlation_time_s": (
                args.measurement_noise_correlation_time
            ),
            "physical_state_dimension": RIGID_BODY_STATE_DIM,
            "controller_state_dimension": controller_state_dim,
            # Backward-compatible alias used by older summary scripts.
            "state_dimension": controller_state_dim,
            "motor_state_representation": (
                "sqrt_thrust_four_state_first_order"
                if controller_state_dim == RIGID_BODY_STATE_DIM + 4
                else "not_in_controller_model"
            ),
            "control_dimension": 4,
            "control_input_constraint": "per_rotor_physical_thrust_bounds",
            "control_input_lower_bound_N": float(dynamics.u_min[0]),
            "control_input_upper_bound_N": float(dynamics.u_max[0]),
            "state_path_constraints": "none",
            "residual_dimension": (
                args.ssi_residual_dimension if controller == "ssi" else 6
            ),
            "physics_frequency_hz": PHYSICS_FREQUENCY,
            "control_frequency_hz": control_frequency,
            "control_cpu_core": (
                args.control_cpu_core
                if args.control_cpu_core is not None else np.nan
            ),
            "trainer_cpu_core": (
                args.trainer_cpu_core
                if args.trainer_cpu_core is not None else np.nan
            ),
            "torch_threads": args.torch_threads,
            "physics_steps_per_control": physics_steps_per_control,
            "actuator_model": args.actuator_model,
            "actuator_update_frequency_hz": PHYSICS_FREQUENCY,
            "actuator_substep_samples": len(actuator_substeps),
            "motor_time_constant_s": (
                args.motor_time_constant
                if args.actuator_model == "first_order" else 0.0
            ),
            "motor_gain_range_fraction": (
                args.motor_gain_range
                if args.actuator_model == "first_order" else 0.0
            ),
            "motor_noise_std_fraction": (
                args.motor_noise_std
                if args.actuator_model == "first_order" else 0.0
            ),
            "motor_noise_correlation_time_s": (
                args.motor_noise_correlation_time
                if args.actuator_model == "first_order" else 0.0
            ),
            **{
                f"motor_{index + 1}_gain": actuator.motor_gains[index]
                for index in range(4)
            },
            "actuator_thrust_error_rms_N": float(
                np.sqrt(np.mean(actuation_error**2))
            ),
            "actuator_thrust_error_mean_abs_N": float(
                np.mean(np.abs(actuation_error))
            ),
            "actuator_relative_thrust_error_rms": float(
                np.sqrt(np.mean(actuation_error**2))
                / max(float(np.mean(controls_array)), 1e-12)
            ),
            **{
                f"motor_{index + 1}_thrust_error_rms_N": float(
                    np.sqrt(np.mean(actuation_error[:, index] ** 2))
                )
                for index in range(4)
            },
            "mpc_horizon_s": T_HORIZON,
            "mpc_nodes": N,
            "control_deadline_ms": 1000.0 * dt,
            "wind_update_frequency_hz": (
                PHYSICS_FREQUENCY if wind_model is not None else np.nan
            ),
            "training_sample_frequency_hz": (
                control_frequency if controller != "nominal" else np.nan
            ),
            "wind_substep_samples": len(wind_substeps),
            **{
                f"mean_wind_{axis}_start_mps": getattr(
                    args, f"mean_wind_{axis}"
                )
                for axis in "xyz"
            },
            **{
                f"mean_wind_{axis}_end_mps": getattr(
                    args, f"mean_wind_{axis}_end"
                )
                for axis in "xyz"
            },
            **{
                f"turbulence_sigma_{axis}_start_mps": getattr(
                    args, f"turbulence_sigma_{axis}"
                )
                for axis in "xyz"
            },
            **{
                f"turbulence_sigma_{axis}_end_mps": getattr(
                    args, f"turbulence_sigma_{axis}_end"
                )
                for axis in "xyz"
            },
            **{
                f"length_scale_{axis}_m": getattr(
                    args, f"length_scale_{axis}"
                )
                for axis in "xyz"
            },
            "advection_speed_mps": args.advection_speed,
            "wind_ramp_duration_s": args.wind_ramp_duration,
            **{
                f"wind_gradient_{axis}_per_m": getattr(
                    args, f"wind_gradient_{axis}"
                )
                for axis in "xyz"
            },
            **{
                f"wind_torque_{axis}_rms_Nm": float(
                    np.sqrt(np.mean(wind_torque[:, index] ** 2))
                )
                for index, axis in enumerate("xyz")
            },
            "wind_torque_norm_max_Nm": float(
                np.linalg.norm(wind_torque, axis=1).max()
            ),
            "fast_learning_rate": 1e-3 if controller == "t2s" else np.nan,
            "slow_learning_rate": 1e-3 if controller == "t2s" else np.nan,
            "fast_epochs": args.fast_epochs if controller == "t2s" else np.nan,
            "slow_epochs": args.slow_epochs if controller == "t2s" else np.nan,
            "fast_batch_size": (
                args.fast_batch_size if controller == "t2s" else np.nan
            ),
            "fast_update_every_steps": (
                args.fast_update_every if controller == "t2s" else np.nan
            ),
            "slow_batch_size": (
                args.slow_batch_size if controller == "t2s" else np.nan
            ),
            "slow_update_every_steps": (
                args.slow_update_every if controller == "t2s" else np.nan
            ),
            "replay_capacity": (
                REPLAY_MAX if controller in ("neural", "t2s") else 0
            ),
            "slow_sampling_with_replacement": (
                False if controller == "t2s" else np.nan
            ),
            "slow_sampling_strategy": (
                ("equal_recent_history_without_replacement"
                 if hybrid_replay is not None else "uniform_without_replacement")
                if controller == "t2s" else "not_applicable"
            ),
            "t2s_replay_mode": args.t2s_replay_mode if controller == "t2s" else "not_applicable",
            "replay_recent_capacity": 50 if hybrid_replay is not None else (
                REPLAY_MAX if controller in ("neural", "t2s") else 0
            ),
            "replay_reservoir_capacity": 50 if hybrid_replay is not None else 0,
            "reservoir_sampling_seed": seed + 600_000 if hybrid_replay is not None else np.nan,
            "reservoir_population": "fifo_evicted_history" if hybrid_replay is not None else "not_applicable",
            "reservoir_history_seen": hybrid_replay.history_seen if hybrid_replay is not None else 0,
            "replay_recent_final_size": len(hybrid_replay.recent) if hybrid_replay is not None else len(replay_inputs),
            "replay_reservoir_final_size": len(hybrid_replay.history) if hybrid_replay is not None else 0,
            "slow_recent_batch_size": args.slow_batch_size // 2 if hybrid_replay is not None else np.nan,
            "slow_history_batch_size": args.slow_batch_size // 2 if hybrid_replay is not None else np.nan,
            "ssi_feature_count": (
                args.ssi_features if controller == "ssi" else np.nan
            ),
            "ssi_learning_rate": (
                args.ssi_learning_rate if controller == "ssi" else np.nan
            ),
            "ssi_kernel_std": (
                args.ssi_kernel_std if controller == "ssi" else np.nan
            ),
            "ssi_residual_dimension": (
                args.ssi_residual_dimension if controller == "ssi" else np.nan
            ),
            "ssi_thrust_feature": (
                args.ssi_thrust_feature
                if controller == "ssi" else "not_applicable"
            ),
            "ssi_command_thrust_scale_N": (
                ";".join(
                    f"{value:.10g}"
                    for value in np.asarray(
                        env.physical_action_bounds[1], dtype=float
                    )
                )
                if controller == "ssi" else "not_applicable"
            ),
            "ssi_update_every_steps": 1 if controller == "ssi" else np.nan,
            "ssi_update_mean_ms": (
                ssi_updates.update_ms.iloc[1:].mean()
                if controller == "ssi" and len(ssi_updates) > 1 else np.nan
            ),
            "ssi_update_p95_ms": (
                ssi_updates.update_ms.iloc[1:].quantile(0.95)
                if controller == "ssi" and len(ssi_updates) > 1 else np.nan
            ),
            "ssi_prediction_error_mean": (
                ssi_updates.prediction_error_l2.mean()
                if controller == "ssi" and len(ssi_updates) else np.nan
            ),
            "stgp_inducing_points": (
                args.stgp_inducing_points
                if controller == "stgp" else np.nan
            ),
            "stgp_inducing_point_seed": (
                args.stgp_inducing_seed
                if controller == "stgp" else np.nan
            ),
            "stgp_inducing_domain": (
                "normalized_hypercube_[-1,1]^13"
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_feature_normalization": (
                "predeclared_vehicle_physical_work_envelope"
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_velocity_limit_mps": (
                float(stgp_learner.physical_velocity_limit[0])
                if controller == "stgp" else np.nan
            ),
            "stgp_attitude_limits_rad": (
                ";".join(
                    f"{value:.10g}"
                    for value in stgp_learner.physical_attitude_limits
                )
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_body_rate_limits_radps": (
                ";".join(
                    f"{value:.10g}"
                    for value in stgp_learner.physical_body_rate_limits
                )
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_action_bounds_N": (
                ";".join(
                    f"{value:.10g}"
                    for value in (
                        stgp_learner.physical_action_lower[0],
                        stgp_learner.physical_action_upper[0],
                    )
                )
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_spatial_kernel": (
                "ARD RBF" if controller == "stgp" else "not_applicable"
            ),
            "stgp_temporal_kernel": (
                "Matern 1.5" if controller == "stgp" else "not_applicable"
            ),
            "stgp_spatial_lengthscale": (
                args.stgp_spatial_lengthscale
                if controller == "stgp" else np.nan
            ),
            "stgp_temporal_lengthscale_s": (
                args.stgp_temporal_lengthscale
                if controller == "stgp" else np.nan
            ),
            "stgp_linear_output_variance": (
                args.stgp_linear_output_variance
                if controller == "stgp" else np.nan
            ),
            "stgp_angular_output_variance": (
                args.stgp_angular_output_variance
                if controller == "stgp" else np.nan
            ),
            "stgp_observation_noise_variance": (
                args.stgp_observation_noise_variance
                if controller == "stgp" else np.nan
            ),
            "stgp_process_noise_variances": (
                ";".join(
                    f"{value:.10g}"
                    for value in args.stgp_process_noise_variances
                )
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_backoff_scaling_gamma": (
                args.stgp_backoff_scaling
                if controller == "stgp" else np.nan
            ),
            "stgp_ocp_formulation": (
                "noncautious_zero_order_gp_mean"
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_solver_architecture": (
                "l4acados_ResidualLearningMPC_SQP_RTI_one_iteration"
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_state_covariance_propagation": (
                False if controller == "stgp" else np.nan
            ),
            "stgp_constraint_tightening": (
                "none"
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_posterior_variance_role": (
                "diagnostic_only"
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_hyperparameter_source": (
                "predeclared_vehicle_bounds_not_evaluation_seed_fitted"
                if controller == "stgp" else "not_applicable"
            ),
            "stgp_prior_mean": (
                "zero" if controller == "stgp" else "not_applicable"
            ),
            "stgp_update_every_steps": (
                1 if controller == "stgp" else np.nan
            ),
            "stgp_update_mean_ms": (
                stgp_updates.update_ms.iloc[1:].mean()
                if controller == "stgp" and len(stgp_updates) > 1
                else np.nan
            ),
            "stgp_update_p95_ms": (
                stgp_updates.update_ms.iloc[1:].quantile(0.95)
                if controller == "stgp" and len(stgp_updates) > 1
                else np.nan
            ),
            "stgp_inference_mean_ms": (
                stgp_updates.inference_ms.mean()
                if controller == "stgp" and len(stgp_updates)
                else np.nan
            ),
            "stgp_inference_p95_ms": (
                stgp_updates.inference_ms.quantile(0.95)
                if controller == "stgp" and len(stgp_updates)
                else np.nan
            ),
            "stgp_prediction_error_mean": (
                stgp_updates.prediction_error_l2.mean()
                if controller == "stgp" and len(stgp_updates)
                else np.nan
            ),
            "stgp_observations": (
                stgp_learner.update_count
                if controller == "stgp" else np.nan
            ),
            "knode_collection_duration_s": (
                args.collection_duration if controller == "knode" else np.nan
            ),
            "knode_collection_points": (
                knode_collection_steps if controller == "knode" else np.nan
            ),
            "knode_data_start_delay_s": (
                args.data_start_delay if controller == "knode" else np.nan
            ),
            "knode_training_epochs": (
                args.training_epochs if controller == "knode" else np.nan
            ),
            "knode_learning_rate": 1e-2 if controller == "knode" else np.nan,
            "knode_regularization": 1e-7 if controller == "knode" else np.nan,
            "knode_queue_capacity": 3 if controller == "knode" else np.nan,
            "knode_replay_buffer": False if controller == "knode" else np.nan,
            "mean_position_error_m": position_error.mean(),
            "p95_position_error_m": np.quantile(position_error, 0.95),
            "mean_attitude_error_rad": trajectory.attitude_error.mean(),
            "p95_attitude_error_rad": trajectory.attitude_error.quantile(0.95),
            "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_error ** 2))),
            "position_estimation_error_mean_m": (
                trajectory.position_estimation_error.mean()
            ),
            "velocity_estimation_error_mean_mps": (
                trajectory.velocity_estimation_error.mean()
            ),
            "taylor_error_mean": (
                accuracy.error_l2.mean() if len(accuracy) else np.nan
            ),
            "taylor_error_p95": (
                accuracy.error_l2.quantile(0.95) if len(accuracy) else np.nan
            ),
            "control_critical_mean_ms": timing.control_critical_ms.mean(),
            "control_critical_p95_ms": timing.control_critical_ms.quantile(0.95),
            "solve_mean_ms": timing.solve_ms.mean(),
            "solve_p95_ms": timing.solve_ms.quantile(0.95),
            "solve_max_ms": timing.solve_ms.max(),
            "control_release_mean_ms": timing.control_release_latency_ms.mean(),
            "control_release_p95_ms": timing.control_release_latency_ms.quantile(0.95),
            "control_release_max_ms": timing.control_release_latency_ms.max(),
            "start_lateness_p95_ms": timing.start_lateness_ms.quantile(0.95),
            "loop_mean_ms": timing.loop_ms.mean(),
            "loop_p95_ms": timing.loop_ms.quantile(0.95),
            "loop_max_ms": timing.loop_ms.max(),
            "control_deadline_misses": int(timing.deadline_miss.sum()),
            "control_candidates_accepted": int(
                timing.candidate_accepted.sum()
            ),
            "control_candidates_discarded": int(
                timing.candidate_discarded.sum()
            ),
            "maximum_applied_control_age_steps": int(
                timing.applied_control_age_steps.max()
            ),
            "control_release_deadline_misses": int(
                timing.release_deadline_miss.sum()
            ),
            "solver_failures": int((timing.solver_status != 0).sum()),
            "neural_submitted": submitted["neural"],
            "fast_submitted": submitted["fast"],
            "slow_submitted": submitted["slow"],
            "knode_submitted": submitted["knode"],
            "neural_skipped_busy": skipped["neural"],
            "fast_skipped_busy": skipped["fast"],
            "slow_skipped_busy": skipped["slow"],
            "knode_skipped_busy": skipped["knode"],
        }]
    )
    summary.to_csv(str(stem) + "_summary.csv", index=False)

    print(f"\n3-D {controller.upper()} MPC run {run_id}, seed {seed}")
    print(
        "Plant: 12-state full 3-D rigid body + four 500 Hz motor states; "
        f"MPC prediction state={controller_state_dim}D"
    )
    print(f"Task: {args.task}; wind scenario: {_wind_scenario(args)}")
    print(
        f"Frequencies: physics={PHYSICS_FREQUENCY}Hz, "
        f"MPC/data={control_frequency}Hz, "
        f"wind force={PHYSICS_FREQUENCY}Hz "
        f"({physics_steps_per_control} physics substeps/control)"
    )
    print(
        f"Mean XYZ position error: {position_error.mean():.6f} m; "
        f"p95={np.quantile(position_error, 0.95):.6f} m"
    )
    print(
        "SO(3) attitude error: "
        f"mean={attitude_error.mean():.6f}rad, "
        f"p95={np.quantile(attitude_error, 0.95):.6f}rad; "
        f"yaw RMSE={np.sqrt(np.mean(yaw_error ** 2)):.6f}rad"
    )
    print(
        "Controller measurement: "
        f"delay={args.measurement_delay_steps} step "
        f"({1000.0 * args.measurement_delay_steps * dt:.1f}ms), "
        "stationary std="
        f"[position {args.position_noise_std:g}m, "
        f"velocity {args.velocity_noise_std:g}m/s, "
        f"attitude {args.attitude_noise_std:g}rad, "
        f"body rate {args.body_rate_noise_std:g}rad/s]"
    )
    print(
        "Residual labels: "
        f"{len(residual_transitions)} causal source-time transitions; "
        f"RK4 nominal rollout uses all {physics_steps_per_control} "
        "sent 500 Hz commands; "
        "mean endpoint-to-availability delay="
        f"{residual_transitions.label_delay_seconds.mean():.3f}s"
        if len(residual_transitions)
        else "Residual labels: no complete delayed transition available"
    )
    print(
        "500 Hz motor actuator: "
        f"model={args.actuator_model}, "
        f"tau={actuator.time_constant:g}s, "
        f"gain={actuator.motor_gains}, "
        f"relative-noise std={actuator.noise_std:g}; "
        "command-to-applied thrust RMS="
        f"{np.sqrt(np.mean(actuation_error**2)):.6f}N"
    )
    print(
        "Control critical path: "
        f"mean={timing.control_critical_ms.mean():.3f}ms, "
        f"p95={timing.control_critical_ms.quantile(0.95):.3f}ms, "
        f"deadline misses={int(timing.deadline_miss.sum())}/{steps}"
    )
    print(
        "Control release policy: "
        f"{summary.control_release_policy.iloc[0]}; "
        f"accepted={int(timing.candidate_accepted.sum())}, "
        f"discarded={int(timing.candidate_discarded.sum())}, "
        "maximum applied-command age="
        f"{int(timing.applied_control_age_steps.max())} steps"
    )
    print(
        "MPC solve: "
        f"mean={timing.solve_ms.mean():.3f}ms, "
        f"p95={timing.solve_ms.quantile(0.95):.3f}ms, "
        f"max={timing.solve_ms.max():.3f}ms"
    )
    if controller == "ssi" and len(ssi_updates) > 1:
        print(
            "SSI online update: "
            f"mean={ssi_updates.update_ms.iloc[1:].mean():.3f}ms, "
            f"p95={ssi_updates.update_ms.iloc[1:].quantile(0.95):.3f}ms"
        )
    if controller == "stgp" and len(stgp_updates) > 1:
        print(
            "STGP Kalman update: "
            f"mean={stgp_updates.update_ms.iloc[1:].mean():.3f}ms, "
            f"p95={stgp_updates.update_ms.iloc[1:].quantile(0.95):.3f}ms; "
            "horizon mean/Jacobian: "
            f"mean={stgp_updates.inference_ms.mean():.3f}ms, "
            f"p95={stgp_updates.inference_ms.quantile(0.95):.3f}ms"
        )
        print(
            "STGP non-cautious OCP: l4acados residual-learning SQP-RTI; "
            "posterior mean used by the dynamics; posterior variance "
            "recorded diagnostically; no state bounds or constraint "
            "tightening; physical motor-thrust bounds retained"
        )
    print(
        "Wall-clock control release: "
        f"mean={timing.control_release_latency_ms.mean():.3f}ms, "
        f"p95={timing.control_release_latency_ms.quantile(0.95):.3f}ms, "
        f"max={timing.control_release_latency_ms.max():.3f}ms"
    )
    print(
        f"Training submitted: neural={submitted['neural']}, "
        f"fast={submitted['fast']}, slow={submitted['slow']}, "
        f"knode={submitted['knode']}"
    )
    print(
        "Wind moment about body centre: "
        f"RMS xyz={np.sqrt(np.mean(wind_torque ** 2, axis=0))} N m; "
        f"max norm={np.linalg.norm(wind_torque, axis=1).max():.6e} N m"
    )
    print(f"Saved trajectory: {stem}.csv")
    print(f"Saved 500 Hz wind trace: {stem}_wind_500hz.csv")
    print(f"Saved 500 Hz actuator trace: {stem}_actuator_500hz.csv")
    plot_results(
        trajectory, timing, accuracy, training, ssi_updates, stgp_updates,
        neural_losses, fast_losses, slow_losses, knode_losses,
        mass, controller, args, stem,
    )
    return {
        "position_error": float(position_error.mean()),
        "deadline_misses": int(timing.deadline_miss.sum()),
    }


def plot_results(
    trajectory, timing, accuracy, training, ssi_updates, stgp_updates,
    neural_losses, fast_losses, slow_losses, knode_losses,
    mass, controller, args, stem,
):
    dt = 1.0 / args.control_frequency
    time_axis = trajectory.time.to_numpy()
    figures = []

    figure, axis = plt.subplots(figsize=(12, 7))
    for component, label in zip("xyz", (r"$F_{w,x}/m$", r"$F_{w,y}/m$", r"$F_{w,z}/m$")):
        axis.plot(time_axis, trajectory[f"wind_f{component}"] / mass, label=label)
    axis.set(xlabel="Time [s]", ylabel="Equivalent acceleration [m/s²]",
             title="Three-axis wind acceleration applied to the plant")
    axis.grid(True); axis.legend(); figure.tight_layout()
    figures.append((figure, "_wind_acceleration.png"))

    figure, axis = plt.subplots(figsize=(12, 7))
    for component, label in zip(
        "xyz", (r"$\tau_{w,x}$", r"$\tau_{w,y}$", r"$\tau_{w,z}$")
    ):
        axis.plot(
            time_axis,
            1000.0 * trajectory[f"wind_tau_{component}"],
            label=label,
        )
    axis.set(
        xlabel="Time [s]",
        ylabel="Aerodynamic moment [mN m]",
        title="Three-axis wind moment applied to the plant",
    )
    axis.grid(True); axis.legend(); figure.tight_layout()
    figures.append((figure, "_wind_torque.png"))

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for component in "xyz":
        axes[0].plot(time_axis, trajectory[f"wind_mean_v{component}"], label=rf"$\mu_{component}(t)$")
        axes[1].plot(time_axis, trajectory[f"wind_sigma_{component}"], label=rf"$\sigma_{component}(t)$")
    axes[0].set_ylabel("Local mean [m/s]")
    axes[1].set(xlabel="Time [s]", ylabel="Local standard deviation [m/s]")
    for axis in axes: axis.grid(True); axis.legend()
    figure.suptitle("Prescribed non-stationary Dryden statistics")
    figure.tight_layout(); figures.append((figure, "_wind_parameters.png"))

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, component in zip(axes, "xyz"):
        axis.plot(time_axis, trajectory[component], label=component)
        axis.plot(time_axis, trajectory[f"{component}_ref"], "--", label="reference")
        axis.set_ylabel(f"{component} [m]"); axis.grid(True); axis.legend()
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle(f"3-D position {args.task}")
    figure.tight_layout(); figures.append((figure, "_position_tracking.png"))

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for axis, name in zip(axes, ("roll", "pitch", "yaw")):
        axis.plot(time_axis, trajectory[name], label="true")
        axis.plot(
            time_axis,
            trajectory[f"{name}_ref"],
            "--",
            label="reference",
        )
        axis.set_ylabel(f"{name} [rad]"); axis.grid(True)
        axis.legend()
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("Full attitude tracking with flatness-consistent reference")
    figure.tight_layout(); figures.append((figure, "_attitude.png"))

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    measurement_errors = (
        ("position_estimation_error", "Position error [m]"),
        ("velocity_estimation_error", "Velocity error [m/s]"),
        ("attitude_estimation_error", "Attitude error [rad]"),
        ("body_rate_estimation_error", "Body-rate error [rad/s]"),
    )
    for axis, (column, label) in zip(axes, measurement_errors):
        axis.plot(time_axis, trajectory[column])
        axis.set_ylabel(label)
        axis.grid(True)
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("Delayed noisy state estimate versus PyBullet truth")
    figure.tight_layout(); figures.append((figure, "_measurement_errors.png"))

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(trajectory.x, trajectory.y, trajectory.z, label="quadrotor")
    if args.task == "circle":
        axis.plot(
            trajectory.x_ref,
            trajectory.y_ref,
            trajectory.z_ref,
            "--",
            label="reference",
        )
    else:
        axis.scatter([trajectory.x_ref.iloc[0]], [trajectory.y_ref.iloc[0]],
                     [trajectory.z_ref.iloc[0]], marker="x", s=100, label="goal")
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="3-D trajectory")
    axis.legend(); figure.tight_layout(); figures.append((figure, "_trajectory_3d.png"))

    figure, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for index, axis in enumerate(axes):
        motor = index + 1
        axis.plot(
            time_axis,
            trajectory[f"u{motor}"],
            label=rf"$u_{{{motor}}}^{{released}}$",
        )
        if args.deadline_aware:
            axis.plot(
                time_axis,
                trajectory[f"u{motor}_candidate"],
                ":",
                alpha=0.8,
                label=rf"$u_{{{motor}}}^{{candidate}}$",
            )
        axis.plot(
            time_axis,
            trajectory[f"u{motor}_applied"],
            "--",
            label=rf"$u_{{{motor}}}^{{applied}}$ (500 Hz mean)",
        )
        axis.set_ylabel("Thrust [N]")
        axis.grid(True)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time [s]")
    figure.suptitle("MPC motor commands versus realized actuator thrust")
    figure.tight_layout()
    figures.append((figure, "_controls.png"))

    if controller == "neural":
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(neural_losses); axis.set(xlabel="Training epoch", ylabel="MSE", title="Online residual training")
        axis.grid(True)
    elif controller == "t2s":
        figure, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].plot(fast_losses); axes[0].set_ylabel("Fast MSE")
        axes[1].plot(slow_losses); axes[1].set(xlabel="Training epoch", ylabel="Slow MSE")
        for axis in axes: axis.grid(True)
        figure.suptitle("Online two-timescale residual training")
    elif controller == "knode":
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.plot(knode_losses)
        axis.set(
            xlabel="Training epoch",
            ylabel="MSE + regularization",
            title="Fresh-window KNODE queue training",
        )
        axis.grid(True)
    elif controller == "ssi":
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(ssi_updates.time, ssi_updates.prediction_error_l2)
        axes[0].set_ylabel("Velocity prediction error [m/s²]")
        axes[1].plot(ssi_updates.time, ssi_updates.alpha_l2)
        axes[1].set(xlabel="Time [s]", ylabel=r"$\|\alpha\|_2$")
        for axis in axes:
            axis.grid(True)
        figure.suptitle("SSI-MPC online RFF least-squares update")
    elif controller == "stgp":
        figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(
            stgp_updates.time, stgp_updates.prediction_error_l2
        )
        axes[0].set_ylabel("Residual prediction error")
        axes[1].plot(
            stgp_updates.time, stgp_updates.posterior_mean_l2
        )
        axes[1].set(
            xlabel="Time [s]", ylabel=r"$\|\mu_{GP}\|_2$"
        )
        for axis in axes:
            axis.grid(True)
        figure.suptitle("Spatio-temporal GP recursive Kalman update")
    else:
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.text(
            0.5, 0.5, "Nominal MPC: no online model update",
            horizontalalignment="center", verticalalignment="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
    figure.tight_layout(); figures.append((figure, "_online_learning.png"))

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    if len(accuracy):
        max_error = accuracy.groupby("step").error_l2.max()
        axes[0].plot(np.arange(len(max_error)) * dt, max_error)
        axes[0].set_ylabel("Max Taylor error")
    elif controller == "ssi" and len(ssi_updates):
        axes[0].plot(
            ssi_updates.time, ssi_updates.prediction_error_l2
        )
        axes[0].set_ylabel("SSI prediction error")
    elif controller == "stgp" and len(stgp_updates):
        axes[0].plot(
            stgp_updates.time, stgp_updates.prediction_error_l2
        )
        axes[0].set_ylabel("STGP prediction error")
    else:
        axes[0].text(
            0.5, 0.5, "No learned-model approximation",
            horizontalalignment="center", verticalalignment="center",
            transform=axes[0].transAxes,
        )
    axes[1].plot(time_axis, timing.control_release_latency_ms)
    axes[1].axhline(
        1000.0 * dt,
        color="r",
        linestyle="--",
        label=f"{1000.0 * dt:g} ms deadline",
    )
    axes[1].set_ylabel("Release latency [ms]"); axes[1].legend()
    if len(training):
        active = training[training.activated]
        axes[2].plot(active.trigger_step * dt, active.trigger_to_activate_ms, marker=".")
    if controller == "neural":
        deadline_steps = args.update_every
    elif controller == "t2s":
        deadline_steps = args.fast_update_every
    elif controller == "knode":
        deadline_steps = int(
            round(args.collection_duration * args.control_frequency)
        )
    else:
        deadline_steps = 1
        axes[2].plot(
            timing.step * dt,
            timing.online_update_ms,
            label=(
                "online update"
                if controller in ("ssi", "stgp") else "no update"
            ),
        )
    axes[2].axhline(1000.0 * dt * deadline_steps, color="r", linestyle="--")
    axes[2].set(xlabel="Experiment time [s]", ylabel="Activation latency [ms]")
    for axis in axes: axis.grid(True)
    figure.suptitle(f"3-D {controller.upper()} MPC real-time diagnostics")
    figure.tight_layout(); figures.append((figure, "_realtime_diagnostics.png"))

    for figure, suffix in figures:
        figure.savefig(str(stem) + suffix, dpi=150)
    print(f"Saved {len(figures)} figures with prefix: {stem}")
    plt.show()
    for figure, _ in figures: plt.close(figure)


def main(controller):
    if controller not in (
        "nominal", "neural", "t2s", "ssi", "stgp", "knode"
    ):
        raise ValueError(
            "controller must be 'nominal', 'neural', 't2s', 'ssi', "
            "'stgp' or 'knode'"
        )
    args = parse_args(controller)
    if args.control_cpu_core is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU affinity requires os.sched_setaffinity")
        os.sched_setaffinity(0, {args.control_cpu_core})
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    print(
        "Wind model: three-axis Dryden filters + spatially distributed "
        "Crazyflie rotor drag"
    )
    results = [run_once(args, controller, run_id) for run_id in range(args.num_runs)]
    if len(results) > 1:
        print(
            "Across runs: mean XYZ error="
            f"{np.mean([result['position_error'] for result in results]):.6f} m"
        )
