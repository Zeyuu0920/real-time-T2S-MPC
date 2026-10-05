"""Single-seed Go2 experiment using the external paper's controller stack."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import mujoco as mj
import numpy as np

from convex_mpc.com_trajectory import ComTraj
from convex_mpc.gait import Gait
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.leg_controller import LegController

from src.go2_friction import (
    MuJoCoRandomFrictionGo2Model,
    RigidPayloadConfig,
    SmoothRandomFrictionConfig,
)
from src.go2_paper_baseline import (
    AuthorCodeRandomFeatures,
    AuthorCodeSSIOnlineLearner,
    PAPER_FORCE_REFERENCE_MASS,
    PAPER_FRICTION_COEFFICIENT,
    PAPER_HORIZON_SECONDS,
    PAPER_INTERVALS,
    PAPER_LOW_LEVEL_HZ,
    PAPER_MAX_STANCE_FZ,
    PAPER_MIN_STANCE_FZ,
    PAPER_MODEL_MASS,
    PAPER_MPC_HZ,
    PAPER_NODE_DT,
    PAPER_NODES,
    PAPER_RFF_COUNT,
    PAPER_RFF_KERNEL_STD,
    PAPER_RFF_LEARNING_RATE,
    PAPER_STATE_COST_DIAGONAL,
    PAPER_INPUT_COST_DIAGONAL,
    PaperGo2NMPC,
    paper_first_element_duration,
    paper_terms_numpy,
    rotation_zyx_numpy,
    world_state_to_paper_state,
)


SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
SIM_STEPS_PER_MPC = SIM_HZ // PAPER_MPC_HZ
SIM_STEPS_PER_LOW_LEVEL = SIM_HZ // PAPER_LOW_LEVEL_HZ
TARGET_HEIGHT = 0.30
AUTHOR_GAIT_PERIOD = 0.54
AUTHOR_GAIT_DUTY = 0.50
ADAPTER_GAIT_PERIOD = 1.0 / 3.0
ADAPTER_GAIT_DUTY = 0.60
STAND_WARMUP_SECONDS = 1.0
LEGS = ("FL", "FR", "RL", "RR")
LEG_SLICES = {
    "FL": slice(0, 3),
    "FR": slice(3, 6),
    "RL": slice(6, 9),
    "RR": slice(9, 12),
}
TORQUE_LIMIT = np.array([23.7, 23.7, 45.0] * 4, dtype=float)


@dataclass(frozen=True)
class PaperBaselineConfig:
    controller: str = "nominal"
    duration: float = 8.0
    seed: int = 42
    target_speed: float = 0.5
    payload_mass: float = 4.0
    execution_gait: str = "author"
    output_dir: str = "results_go2_external_paper_baseline"


class _StandingGait:
    """All-feet stance used by the authors' required stand-before-run step."""

    gait_period = AUTHOR_GAIT_PERIOD
    stance_time = AUTHOR_GAIT_PERIOD
    swing_time = AUTHOR_GAIT_PERIOD

    @staticmethod
    def compute_contact_table(time_now: float, dt: float, count: int) -> np.ndarray:
        del time_now, dt
        return np.ones((4, count), dtype=np.int32)

    def compute_current_mask(self, time_now: float) -> np.ndarray:
        return self.compute_contact_table(time_now, 0.0, 1)[:, 0]


class _AuthorDiscreteTrottingGait(Gait):
    """The source's 18-node trot, initialized with the local planner."""

    _period_nodes = int(round(AUTHOR_GAIT_PERIOD / PAPER_NODE_DT))
    _phase_offsets = np.array([0.0, 0.5, 0.5, 0.0], dtype=float)

    def __init__(self, start_time: float, target_speed: float) -> None:
        super().__init__(1.0 / AUTHOR_GAIT_PERIOD, AUTHOR_GAIT_DUTY)
        self.start_time = float(start_time)
        self.target_speed = float(target_speed)
        self._planned_touchdowns: dict[str, np.ndarray] = {}

    def set_planned_touchdowns(
        self, planned_touchdowns: dict[str, np.ndarray]
    ) -> None:
        """Share the latest footstep-plan touchdowns with the leg controller.

        The authors' stack sends one common foot plan to both the centroidal
        MPC and the inverse-dynamics controller.  Keeping these targets on the
        gait object gives the reconstructed MuJoCo bridge the same data flow.
        """

        self._planned_touchdowns = {
            leg: np.asarray(position, dtype=float).reshape(3).copy()
            for leg, position in planned_touchdowns.items()
        }

    def compute_contact_table(
        self, t0: float, dt: float, count: int
    ) -> np.ndarray:
        elapsed = np.maximum(
            0.0, t0 - self.start_time + np.arange(count) * dt
        )
        nodes = np.floor(elapsed / PAPER_NODE_DT + 1e-9).astype(int)
        phase = np.mod(
            self._phase_offsets[:, None]
            + nodes[None, :] / self._period_nodes,
            1.0,
        )
        return (phase < AUTHOR_GAIT_DUTY).astype(np.int32)

    def _touchdown_world(self, go2: PinGo2Model, leg: str) -> np.ndarray:
        """Flat-ground version of the author's mid-stance/Raibert rule."""

        if leg in self._planned_touchdowns:
            return self._planned_touchdowns[leg].copy()

        base_position = go2.current_config.base_pos
        desired_velocity = go2.R_z @ np.array(
            [self.target_speed, 0.0, 0.0], dtype=float
        )
        current_velocity = np.asarray(go2.vel_com_world, dtype=float)
        time_to_midstance = self.swing_time + 0.5 * self.stance_time
        hip_world = (
            np.array([base_position[0], base_position[1], 0.0])
            + go2.R_z @ go2.get_hip_offset(leg)
            + desired_velocity * time_to_midstance
        )
        height = max(float(base_position[2]), 0.0)
        velocity_tracking = np.sqrt(height / 9.81) * (
            current_velocity - desired_velocity
        )
        touchdown = hip_world + velocity_tracking
        touchdown[2] = 0.02
        return touchdown

    def compute_touchdown_world_for_traj_purpose_only(
        self, go2: PinGo2Model, leg: str
    ) -> np.ndarray:
        return self._touchdown_world(go2, leg)

    def compute_swing_traj_and_touchdown(
        self, go2: PinGo2Model, leg: str
    ):
        foot_position, _ = go2.get_single_foot_state_in_world(leg)
        touchdown = self._touchdown_world(go2, leg)
        trajectory = self._make_author_swing_trajectory(
            foot_position, touchdown, self.swing_time, apex_height=0.07
        )
        return trajectory, touchdown

    @staticmethod
    def _make_author_swing_trajectory(
        start: np.ndarray,
        touchdown: np.ndarray,
        duration: float,
        *,
        apex_height: float,
    ):
        """Port the public planner's cubic-Hermite swing interpolation.

        Its x/y coordinates use one zero-end-velocity cubic.  The z coordinate
        uses two cubics through an *absolute* 7 cm apex.  The inherited MuJoCo
        gait instead adds a 7 cm minimum-jerk bump to the toe height, reaching
        9 cm and producing different acceleration feedforward.
        """

        start = np.asarray(start, dtype=float).reshape(3).copy()
        touchdown = np.asarray(touchdown, dtype=float).reshape(3).copy()
        duration = float(duration)

        def cubic(p0: float, p1: float, t: float, total: float):
            phase = np.clip(t / total, 0.0, 1.0)
            position = p0 + (3.0 * phase**2 - 2.0 * phase**3) * (p1 - p0)
            velocity = (
                (6.0 * phase - 6.0 * phase**2) * (p1 - p0) / total
            )
            acceleration = (
                (6.0 - 12.0 * phase) * (p1 - p0) / total**2
            )
            return position, velocity, acceleration

        def evaluate(time_since_takeoff: float):
            time_since_takeoff = float(
                np.clip(time_since_takeoff, 0.0, duration)
            )
            position = np.zeros(3)
            velocity = np.zeros(3)
            acceleration = np.zeros(3)
            for axis in (0, 1):
                (
                    position[axis],
                    velocity[axis],
                    acceleration[axis],
                ) = cubic(
                    start[axis],
                    touchdown[axis],
                    time_since_takeoff,
                    duration,
                )
            half = 0.5 * duration
            if time_since_takeoff < half:
                position[2], velocity[2], acceleration[2] = cubic(
                    start[2], apex_height, time_since_takeoff, half
                )
            else:
                position[2], velocity[2], acceleration[2] = cubic(
                    apex_height,
                    touchdown[2],
                    time_since_takeoff - half,
                    half,
                )
            return position, velocity, acceleration

        return evaluate


def author_twist_reference(
    measured_world_state: np.ndarray,
    *,
    target_speed: float,
    lateral_speed: float = 0.0,
    yaw_rate: float = 0.0,
    first_element_duration: float = PAPER_NODE_DT,
) -> np.ndarray:
    """Port the author's rolling twist reference over 20 prediction nodes."""

    measured = np.asarray(measured_world_state, dtype=float).reshape(12)
    reference = np.zeros((12, PAPER_NODES + 1), dtype=float)
    reference[0:2, 0] = measured[0:2]
    reference[2, :] = TARGET_HEIGHT
    reference[5, 0] = measured[5]
    for node in range(PAPER_NODES):
        yaw = reference[5, node]
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        velocity_world = np.array(
            [
                cosine * target_speed - sine * lateral_speed,
                sine * target_speed + cosine * lateral_speed,
                0.0,
            ],
            dtype=float,
        )
        reference[6:9, node] = velocity_world
        reference[11, node] = yaw_rate
        step = first_element_duration if node == 0 else PAPER_NODE_DT
        reference[0:3, node + 1] = (
            reference[0:3, node] + step * velocity_world
        )
        reference[2, node + 1] = TARGET_HEIGHT
        reference[5, node + 1] = reference[5, node] + step * yaw_rate
    terminal_yaw = reference[5, -1]
    terminal_cosine = np.cos(terminal_yaw)
    terminal_sine = np.sin(terminal_yaw)
    reference[6:9, -1] = np.array(
        [
            terminal_cosine * target_speed - terminal_sine * lateral_speed,
            terminal_sine * target_speed + terminal_cosine * lateral_speed,
            0.0,
        ],
        dtype=float,
    )
    reference[11, -1] = yaw_rate
    return reference


def _negative_body_foot_positions(
    world_levers: np.ndarray, reference_rpy: np.ndarray
) -> np.ndarray:
    """Return the source's relative-world lever despite its body-frame name."""

    world_levers = np.asarray(world_levers, dtype=float).reshape(4, 3)
    del reference_rpy
    return -world_levers.reshape(12)


def _horizon_negative_body_feet(
    trajectory: ComTraj,
    references: np.ndarray,
    current_contact: np.ndarray,
    com_minus_base_world: np.ndarray,
) -> np.ndarray:
    arrays = (
        trajectory.r_fl_foot_world,
        trajectory.r_fr_foot_world,
        trajectory.r_rl_foot_world,
        trajectory.r_rr_foot_world,
    )
    contact = np.asarray(
        trajectory.contact_table[:, :PAPER_INTERVALS], dtype=bool
    )
    current_contact = np.asarray(current_contact, dtype=bool).reshape(4)
    carries_com_lever = current_contact.copy()
    result = np.zeros((12, PAPER_INTERVALS), dtype=float)
    for stage in range(PAPER_INTERVALS):
        world_levers = np.stack([array[:, stage] for array in arrays])
        for leg in range(4):
            if not contact[leg, stage]:
                carries_com_lever[leg] = False
            if contact[leg, stage] and carries_com_lever[leg]:
                world_levers[leg] += com_minus_base_world
        result[:, stage] = _negative_body_foot_positions(
            world_levers, references[3:6, stage + 1]
        )
    return result


def _friction_config(seed: int) -> SmoothRandomFrictionConfig:
    """Keep the previously agreed random-strength red/blue plant unchanged."""

    return SmoothRandomFrictionConfig(seed=seed)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(config: PaperBaselineConfig, records: list[dict], status: str) -> dict:
    # The one-second standing phase is never logged, so every record belongs
    # to the walking evaluation.  Do not silently discard another second.
    evaluation = records
    speed_error = np.array([row["vx_error"] for row in evaluation])
    height_error = np.array([row["z_error"] for row in evaluation])
    x_error = np.array([row["x_error"] for row in evaluation])
    y_error = np.array([row["y_error"] for row in evaluation])
    roll_pitch = np.array(
        [[row["roll"], row["pitch"]] for row in evaluation]
    )
    solve_ms = np.array([row["solve_ms"] for row in records], dtype=float)
    end_to_end_ms = np.array(
        [row["mpc_end_to_end_ms"] for row in records], dtype=float
    )
    deadline_ms = 1000.0 / PAPER_MPC_HZ
    return {
        **asdict(config),
        "status": status,
        "simulated_seconds": records[-1]["time"] if records else 0.0,
        "samples": len(records),
        "evaluation_start_seconds": 0.0,
        "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
        "height_rmse_m": float(np.sqrt(np.mean(height_error**2))),
        "position_x_rmse_m": float(np.sqrt(np.mean(x_error**2))),
        "position_y_rmse_m": float(np.sqrt(np.mean(y_error**2))),
        "roll_pitch_rmse_rad": float(np.sqrt(np.mean(roll_pitch**2))),
        "final_x_error_m": float(x_error[-1]),
        "final_y_error_m": float(y_error[-1]),
        "mean_solve_ms": float(np.mean(solve_ms)),
        "p95_solve_ms": float(np.percentile(solve_ms, 95)),
        "max_solve_ms": float(np.max(solve_ms)),
        "mean_mpc_end_to_end_ms": float(np.mean(end_to_end_ms)),
        "p95_mpc_end_to_end_ms": float(np.percentile(end_to_end_ms, 95)),
        "mpc_deadline_ms": deadline_ms,
        "mpc_deadline_miss_count": int(np.count_nonzero(end_to_end_ms > deadline_ms)),
        "mpc_deadline_miss_rate": float(np.mean(end_to_end_ms > deadline_ms)),
        "mean_ipopt_iterations": float(
            np.mean([row["ipopt_iterations"] for row in records])
        ),
        "ssi_updates": int(sum(row["ssi_updated"] for row in records)),
        "ssi_skipped_contact_change": int(
            sum(row["ssi_skipped_contact_change"] for row in records)
        ),
        "final_alpha_l2": float(records[-1]["alpha_l2"]),
        "final_predicted_residual_wrench": (
            [records[-1][f"pred_residual_{axis}"] for axis in ("fx", "fy", "fz", "tx", "ty", "tz")]
        ),
        "controller_provenance": {
            "repository": "UM-iRaL/Adaptive-Legged-Locomotion",
            "commit": "9b0bda7912f1863d4b40244348dd9a9a6c72ca59",
            "port_scope": "Go2 centroidal NMPC and RFF-SSI online update",
            "plant_and_low_level": "existing Go2 MuJoCo and Pinocchio leg-control adapter",
        },
        "paper_controller": {
            "mpc_hz": PAPER_MPC_HZ,
            "low_level_hz": PAPER_LOW_LEVEL_HZ,
            "nodes": PAPER_NODES,
            "intervals_in_author_code": PAPER_INTERVALS,
            "node_dt_seconds": PAPER_NODE_DT,
            "nominal_horizon_label_seconds": PAPER_HORIZON_SECONDS,
            "discretization": "backward_euler_as_public_source_code",
            "solver": "CasADi_IPOPT",
            "friction_cone_mu": PAPER_FRICTION_COEFFICIENT,
            "minimum_stance_fz_N": PAPER_MIN_STANCE_FZ,
            "maximum_stance_fz_N": PAPER_MAX_STANCE_FZ,
            "dynamics_mass_kg": PAPER_MODEL_MASS,
            "grf_cost_reference_mass_kg": PAPER_FORCE_REFERENCE_MASS,
            "Q_diagonal": PAPER_STATE_COST_DIAGONAL.tolist(),
            "R_diagonal": PAPER_INPUT_COST_DIAGONAL.tolist(),
        },
        "ssi_source_code_settings": {
            "physical_input_dim": 15,
            "physical_input_layout": "rpy3_world_velocity3_body_omega3_JT_u_over_100_6",
            "output": "residual_generalized_force_and_torque_6",
            "random_features": PAPER_RFF_COUNT,
            "kernel_std": PAPER_RFF_KERNEL_STD,
            "learning_rate": PAPER_RFF_LEARNING_RATE,
            "feature_normalization": "one_over_sqrt_M_as_public_source_code",
            "contact_transition_updates": "skipped_as_public_source_code",
        },
        "plant_protocol": {
            "payload_kg": config.payload_mass,
            "friction": "same_previously_agreed_random_strength_red_blue_field",
            "red_sliding_range": [0.45, 0.50],
            "blue_sliding_range": [0.05, 0.10],
            "actuator": "ideal_joint_torque_no_added_noise_or_lag",
            "stand_warmup_seconds_excluded_from_metrics": STAND_WARMUP_SECONDS,
            "execution_gait_period_seconds": (
                AUTHOR_GAIT_PERIOD
                if config.execution_gait == "author"
                else ADAPTER_GAIT_PERIOD
            ),
            "execution_gait_duty": (
                AUTHOR_GAIT_DUTY
                if config.execution_gait == "author"
                else ADAPTER_GAIT_DUTY
            ),
            "execution_gait_phase_offsets": (
                [0.0, 0.5, 0.5, 0.0]
                if config.execution_gait == "author"
                else [0.5, 0.0, 0.0, 0.5]
            ),
        },
    }


def run_paper_baseline(config: PaperBaselineConfig) -> tuple[dict, Path]:
    if config.controller not in {"nominal", "ssi"}:
        raise ValueError("controller must be nominal or ssi")
    if config.duration <= 0.0 or config.target_speed <= 0.0:
        raise ValueError("duration and target_speed must be positive")
    if config.payload_mass < 0.0:
        raise ValueError("payload_mass must be nonnegative")
    if config.execution_gait not in {"author", "adapter"}:
        raise ValueError("execution_gait must be author or adapter")

    np.random.seed(config.seed)
    go2 = PinGo2Model()
    simulation = MuJoCoRandomFrictionGo2Model(
        _friction_config(config.seed),
        rigid_payload=(
            RigidPayloadConfig(mass=config.payload_mass)
            if config.payload_mass > 0.0
            else None
        ),
    )
    simulation.update_with_q_pin(go2.current_config.get_q())
    simulation.model.opt.timestep = SIM_DT
    simulation.data.qvel[:] = 0.0
    simulation.update_pin_with_mujoco(go2)

    gait = (
        _AuthorDiscreteTrottingGait(
            STAND_WARMUP_SECONDS, config.target_speed
        )
        if config.execution_gait == "author"
        else Gait(1.0 / ADAPTER_GAIT_PERIOD, ADAPTER_GAIT_DUTY)
    )
    standing_gait = _StandingGait()
    trajectory = ComTraj(go2)
    leg_controller = LegController()
    random_features = (
        AuthorCodeRandomFeatures.sample(config.seed)
        if config.controller == "ssi"
        else None
    )
    learner = (
        AuthorCodeSSIOnlineLearner(random_features)
        if random_features is not None
        else None
    )
    solver = PaperGo2NMPC(random_features=random_features)

    force_hold = np.zeros(12, dtype=float)
    torque_hold = np.zeros(12, dtype=float)
    previous_state = None
    previous_force = None
    previous_negative_feet = None
    previous_contact = None
    records: list[dict] = []
    status = "completed"
    initial_x = None
    initial_y = None

    total_seconds = STAND_WARMUP_SECONDS + config.duration
    walking_started = False
    for sim_step in range(int(round(total_seconds * SIM_HZ))):
        time_now = float(simulation.data.time)
        in_warmup = time_now + 0.5 * SIM_DT < STAND_WARMUP_SECONDS
        active_gait = standing_gait if in_warmup else gait
        experiment_time = max(0.0, time_now - STAND_WARMUP_SECONDS)
        if not in_warmup and not walking_started:
            # The paper initializes alpha at zero at the start of the walking
            # trial.  Do not use stand-up transients as learner data.
            previous_state = None
            previous_force = None
            previous_negative_feet = None
            previous_contact = None
            initial_x = None
            initial_y = None
            walking_started = True
        if sim_step % SIM_STEPS_PER_MPC == 0:
            cycle_start = time.perf_counter()
            simulation.update_pin_with_mujoco(go2)
            rotation = go2.R_body_to_world
            world_state = np.concatenate(
                (
                    go2.current_config.base_pos,
                    go2.current_config.compute_euler_angle_world(),
                    rotation @ go2.current_config.base_vel,
                    rotation @ go2.current_config.base_ang_vel,
                )
            )
            state = world_state_to_paper_state(world_state)
            if initial_x is None:
                initial_x = float(state[0])
                initial_y = float(state[1])
            current_com_levers = np.stack(go2.get_foot_lever_world())
            com_minus_base_world = (
                go2.pos_com_world - go2.current_config.base_pos
            )
            current_world_levers = current_com_levers + com_minus_base_world
            current_negative_feet = _negative_body_foot_positions(
                current_world_levers, state[3:6]
            )
            current_contact = active_gait.compute_contact_table(
                time_now, 1.0 / PAPER_MPC_HZ, 1
            )[:, 0].astype(bool)

            update_target = np.full(6, np.nan)
            prediction_error = np.full(6, np.nan)
            ssi_updated = False
            skipped_contact = False
            if (
                learner is not None
                and not in_warmup
                and previous_state is not None
            ):
                contact_changed = not np.array_equal(
                    current_contact, previous_contact
                )
                if contact_changed:
                    skipped_contact = True
                else:
                    update_target, prediction_error = learner.update(
                        previous_state,
                        state,
                        previous_force,
                        previous_negative_feet,
                        1.0 / PAPER_MPC_HZ,
                    )
                    ssi_updated = True

            first_element_duration = paper_first_element_duration(
                experiment_time
            )
            references_world = author_twist_reference(
                world_state,
                target_speed=(0.0 if in_warmup else config.target_speed),
                first_element_duration=first_element_duration,
            )
            trajectory.generate_traj(
                go2,
                active_gait,
                time_now,
                0.0 if in_warmup else config.target_speed,
                0.0,
                TARGET_HEIGHT,
                0.0,
                time_step=PAPER_NODE_DT,
                state_reference_world=references_world,
                time_horizon_seconds=PAPER_HORIZON_SECONDS,
            )
            if trajectory.N != PAPER_NODES:
                raise RuntimeError(
                    f"expected {PAPER_NODES} trajectory nodes, got {trajectory.N}"
                )
            negative_feet_horizon = _horizon_negative_body_feet(
                trajectory,
                references_world,
                current_contact,
                com_minus_base_world,
            )
            contact_horizon = trajectory.contact_table[:, :PAPER_INTERVALS]
            references_paper = references_world[:, :PAPER_NODES].copy()
            # The reference angular velocity is zero here; this assignment is
            # explicit so future nonzero commands cannot mix world/body frames.
            references_paper[9:12] = 0.0
            result = solver.solve(
                state,
                references_paper,
                negative_feet_horizon,
                contact_horizon,
                first_element_duration=first_element_duration,
                alpha=(learner.alpha if learner is not None else None),
            )
            acceptable = result.status in {
                "Solve_Succeeded",
                "Solved_To_Acceptable_Level",
            }
            if not acceptable or not np.all(np.isfinite(result.force)):
                status = f"solver_failed_{result.status}"
                break
            force_hold = result.force
            _, _, model_jacobian, _ = paper_terms_numpy(
                state, negative_feet_horizon[:, 0]
            )
            modeled_wrench = model_jacobian @ force_hold
            predicted_residual = (
                learner.predict(state, force_hold, current_negative_feet)
                if learner is not None
                else np.zeros(6)
            )
            friction_vector = simulation.friction_at(state[0], state[1])
            x_reference = initial_x + config.target_speed * experiment_time
            commanded_moment_world = np.sum(
                np.cross(current_com_levers, force_hold.reshape(4, 3)), axis=0
            )
            commanded_moment_body = (
                rotation_zyx_numpy(state[3:6]).T @ commanded_moment_world
            )
            if not in_warmup:
                records.append(
                    {
                    "time": experiment_time,
                    "x": state[0],
                    "y": state[1],
                    "z": state[2],
                    "roll": state[3],
                    "pitch": state[4],
                    "yaw": state[5],
                    "vx": state[6],
                    "vy": state[7],
                    "vz": state[8],
                    "omega_body_x": state[9],
                    "omega_body_y": state[10],
                    "omega_body_z": state[11],
                    "vx_reference": config.target_speed,
                    "z_reference": TARGET_HEIGHT,
                    "x_reference_diagnostic": x_reference,
                    "y_reference_diagnostic": initial_y,
                    "vx_error": state[6] - config.target_speed,
                    "z_error": state[2] - TARGET_HEIGHT,
                    "x_error": state[0] - x_reference,
                    "y_error": state[1] - initial_y,
                    "friction_mu": friction_vector[0],
                    "net_commanded_fx": np.sum(force_hold[0::3]),
                    "net_commanded_fy": np.sum(force_hold[1::3]),
                    "net_commanded_fz": np.sum(force_hold[2::3]),
                    "commanded_body_mx": commanded_moment_body[0],
                    "commanded_body_my": commanded_moment_body[1],
                    "commanded_body_mz": commanded_moment_body[2],
                    "modeled_generalized_tx": modeled_wrench[3],
                    "modeled_generalized_ty": modeled_wrench[4],
                    "modeled_generalized_tz": modeled_wrench[5],
                    "predicted_next_roll": result.states[3, 1],
                    "predicted_next_pitch": result.states[4, 1],
                    "predicted_next_omega_body_x": result.states[9, 1],
                    "predicted_next_omega_body_y": result.states[10, 1],
                    **{
                        f"force_{leg}_{axis}": force_hold[3 * leg_index + axis_index]
                        for leg_index, leg in enumerate(("fl", "fr", "rl", "rr"))
                        for axis_index, axis in enumerate(("x", "y", "z"))
                    },
                    "solve_ms": result.solve_ms,
                    "mpc_end_to_end_ms": 1000.0
                    * (time.perf_counter() - cycle_start),
                    "ipopt_iterations": result.iterations,
                    "solver_status": result.status,
                    "ssi_updated": int(ssi_updated),
                    "ssi_skipped_contact_change": int(skipped_contact),
                    "alpha_l2": (
                        np.linalg.norm(learner.alpha) if learner is not None else 0.0
                    ),
                    "prediction_error_l2": np.linalg.norm(prediction_error),
                    "target_residual_fx": update_target[0],
                    "target_residual_fy": update_target[1],
                    "target_residual_fz": update_target[2],
                    "target_residual_tx": update_target[3],
                    "target_residual_ty": update_target[4],
                    "target_residual_tz": update_target[5],
                    "pred_residual_fx": predicted_residual[0],
                    "pred_residual_fy": predicted_residual[1],
                    "pred_residual_fz": predicted_residual[2],
                    "pred_residual_tx": predicted_residual[3],
                    "pred_residual_ty": predicted_residual[4],
                    "pred_residual_tz": predicted_residual[5],
                    }
                )
                previous_state = state.copy()
                previous_force = force_hold.copy()
                previous_negative_feet = current_negative_feet.copy()
                previous_contact = current_contact.copy()

            if state[2] < 0.12 or max(abs(state[3]), abs(state[4])) > 1.0:
                status = "fell"
                break

        if sim_step % SIM_STEPS_PER_LOW_LEVEL == 0:
            simulation.update_pin_with_mujoco(go2)
            raw_torque = np.zeros(12, dtype=float)
            for leg in LEGS:
                leg_slice = LEG_SLICES[leg]
                leg_output = leg_controller.compute_leg_torque(
                    leg,
                    go2,
                    active_gait,
                    force_hold[leg_slice],
                    time_now,
                )
                raw_torque[leg_slice] = leg_output.tau
            torque_hold = np.clip(raw_torque, -TORQUE_LIMIT, TORQUE_LIMIT)

        mj.mj_step1(simulation.model, simulation.data)
        simulation.apply_spatial_contact_friction()
        simulation.set_joint_torque(torque_hold)
        mj.mj_step2(simulation.model, simulation.data)

    output = (
        Path(config.output_dir).expanduser().resolve()
        / f"{config.controller}_payload{config.payload_mass:g}_{config.duration:g}s_seed{config.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(config, records, status)
    _write_csv(output / "trajectory.csv", records)
    with (output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    return metrics, output


def save_comparison_plot(outputs: dict[str, Path], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for method, path in outputs.items():
        data = np.genfromtxt(path / "trajectory.csv", delimiter=",", names=True)
        label = "SSI-MPC (author code)" if method == "ssi" else "Nominal MPC (author code)"
        axes[0].plot(data["time"], data["vx_error"], label=label)
        axes[1].plot(data["time"], data["z_error"], label=label)
        axes[2].plot(data["time"], data["pitch"], label=label)
        axes[3].plot(data["time"], data["friction_mu"], label=label)
    for axis in axes[:3]:
        axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("$v_x-0.5$ [m/s]")
    axes[1].set_ylabel("$z-0.3$ [m]")
    axes[2].set_ylabel("pitch [rad]")
    axes[3].set_ylabel("plant $\\mu$")
    axes[3].set_xlabel("time [s]")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=170)
    plt.close(figure)
