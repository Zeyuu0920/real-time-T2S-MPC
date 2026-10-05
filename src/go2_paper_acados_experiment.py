"""Single-seed MuJoCo Nominal/SSI/T2S experiment on the paper protocol."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import mujoco as mj
import numpy as np
import torch
from l4casadi.realtime import RealTimeL4CasADi

from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.gait import Gait
from convex_mpc.leg_controller import LegController
from convex_mpc.mujoco_model import MuJoCo_GO2_Model

from src.go2_paper_acados import (
    PAPER_ACADOS_HORIZON_SECONDS,
    PAPER_ACADOS_INTERVALS,
    PaperAcadosMPC,
)
from src.go2_contact import (
    ContactWrenchIntervalAccumulator,
    Go2FootContactReader,
)
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
    PAPER_INPUT_COST_DIAGONAL,
    PAPER_LOW_LEVEL_HZ,
    PAPER_MAX_STANCE_FZ,
    PAPER_MIN_STANCE_FZ,
    PAPER_MODEL_MASS,
    PAPER_MPC_HZ,
    PAPER_NODE_DT,
    PAPER_RFF_COUNT,
    PAPER_RFF_KERNEL_STD,
    PAPER_RFF_LEARNING_RATE,
    PAPER_STATE_COST_DIAGONAL,
    author_code_features_numpy,
    paper_first_element_duration,
    paper_force_reference,
    paper_residual_wrench_target_numpy,
    rotation_zyx_numpy,
    world_state_to_paper_state,
)
from src.go2_paper_experiment import (
    AUTHOR_GAIT_DUTY,
    AUTHOR_GAIT_PERIOD,
    TARGET_HEIGHT,
    _AuthorDiscreteTrottingGait,
    _StandingGait,
    author_twist_reference,
)
from src.go2_quad_sdk_low_level import PaperQuadSdkLowLevel
from src.go2_reference import path_following_body_command
from src.models import BoundedTwoSpeedMLP, time_embedding_np
from src.realtime_neural import (
    exact_model_output,
    first_order_parameters,
    load_numpy_state_dict,
)
from src.realtime_t2s import (
    AsyncRealtimeT2STrainer,
    RealtimeT2SJob,
    sample_replay_without_replacement,
)


SIM_HZ = 1000
SIM_DT = 1.0 / SIM_HZ
SIM_STEPS_PER_MPC = SIM_HZ // PAPER_MPC_HZ
SIM_STEPS_PER_LOW_LEVEL = SIM_HZ // PAPER_LOW_LEVEL_HZ
# The source's 0.1 command filter and 0.1 m/s STEP threshold keep all four feet
# in stance for the first two 100 Hz command messages; the third message starts
# the trot.  Reproduce that 20 ms state transition without the old one-second
# non-paper warm-up.
PRETRIAL_STAND_SECONDS = 0.02
PAPER_COMMAND_FILTER_NEW_WEIGHT = 0.10
PAPER_COMMAND_PUBLISH_HZ = 100
PAPER_LATERAL_POSITION_GAIN = 1.0
PAPER_HEADING_GAIN = 3.0
LEGS = ("FL", "FR", "RL", "RR")
MUJOCO_TOE_COLLISION_RADIUS = 0.022
PAPER_HIGH_FRICTION_VECTOR = np.array([0.50, 0.50, 0.010], dtype=float)
PAPER_T2S_TIME_EMBEDDING_DIM = 16
PAPER_T2S_INPUT_DIM = 15 + PAPER_T2S_TIME_EMBEDDING_DIM
PAPER_T2S_TIME_SCALE_SECONDS = 5.0
PAPER_T2S_HIDDEN_DIM = 64
PAPER_T2S_REPLAY_CAPACITY = 400
PAPER_PAYLOAD_INERTIA_DIAGONALS = {
    4.0: np.array([0.00234, 0.00304, 0.00414], dtype=float),
    8.0: np.array([0.00503, 0.00655, 0.00889], dtype=float),
}


@dataclass(frozen=True)
class PaperAcadosExperimentConfig:
    controller: str = "nominal"
    duration: float = 8.0
    seed: int = 42
    target_speed: float = 0.5
    output_dir: str = "results_go2_paper_acados_single_seed_aligned"
    low_level: str = "quad_sdk"
    acados_nlp_solver: str = "SQP_RTI"
    rti_iterations_per_cycle: int = 1
    execution_gait: str = "author"
    terrain: str = "constant_paper_friction"
    friction_tile_length: float = 0.02
    friction_red_sliding_min: float = 0.45
    friction_red_sliding_max: float = 0.50
    friction_blue_sliding_min: float = 0.05
    friction_blue_sliding_max: float = 0.10
    friction_red_length_min: float = 3.80
    friction_red_length_max: float = 4.20
    friction_blue_length_min: float = 0.04
    friction_blue_length_max: float = 0.06
    friction_warmup_end_x: float = 0.50
    payload_mass_kg: float = 0.0
    t2s_fast_update_every: int = 10
    t2s_slow_update_every: int = 80
    t2s_fast_batch_size: int = 16
    t2s_slow_batch_size: int = 64
    t2s_fast_epochs: int = 3
    t2s_slow_epochs: int = 8
    t2s_fast_learning_rate: float = 2.0e-4
    t2s_slow_learning_rate: float = 2.0e-4
    t2s_torch_threads: int = 1


def _zero_t2s_output_layer(model: BoundedTwoSpeedMLP) -> None:
    """Make the untrained T2S controller exactly nominal at time zero."""

    with torch.no_grad():
        model.layer3.weight.zero_()
        model.layer3.bias.zero_()
    model.eval()


def _alternating_friction_config(
    config: PaperAcadosExperimentConfig,
) -> SmoothRandomFrictionConfig:
    """Return the previously frozen random red/blue spatial protocol."""

    return SmoothRandomFrictionConfig(
        seed=config.seed + 300_000,
        x_min=-3.0,
        x_max=max(24.0, 1.25 * config.target_speed * config.duration + 6.0),
        tile_length=config.friction_tile_length,
        half_width=3.0,
        sliding_min=0.05,
        sliding_max=0.50,
        red_sliding_min=config.friction_red_sliding_min,
        red_sliding_max=config.friction_red_sliding_max,
        blue_sliding_min=config.friction_blue_sliding_min,
        blue_sliding_max=config.friction_blue_sliding_max,
        red_length_min=config.friction_red_length_min,
        red_length_max=config.friction_red_length_max,
        blue_length_min=config.friction_blue_length_min,
        blue_length_max=config.friction_blue_length_max,
        warmup_end_x=config.friction_warmup_end_x,
    )


def _paper_rigid_payload_config(payload_mass_kg: float) -> RigidPayloadConfig:
    """Construct a trunk payload with the paper-reported 4/8 kg inertia."""

    payload_mass_kg = float(payload_mass_kg)
    if payload_mass_kg <= 0.0:
        raise ValueError("payload_mass_kg must be positive")
    inertia = PAPER_PAYLOAD_INERTIA_DIAGONALS.get(payload_mass_kg)
    if inertia is None:
        return RigidPayloadConfig(mass=payload_mass_kg)
    ixx, iyy, izz = inertia
    half_size_squared = 3.0 / (2.0 * payload_mass_kg) * np.array(
        [
            iyy + izz - ixx,
            ixx + izz - iyy,
            ixx + iyy - izz,
        ]
    )
    if np.any(half_size_squared <= 0.0):
        raise ValueError("payload inertia does not define a physical box")
    return RigidPayloadConfig(
        mass=payload_mass_kg,
        half_size=tuple(np.sqrt(half_size_squared)),
    )


def _rigid_payload_inertia_diagonal(
    payload: RigidPayloadConfig,
) -> np.ndarray:
    """Return the diagonal inertia of the configured uniform box."""

    half_x, half_y, half_z = payload.half_size
    return payload.mass / 3.0 * np.array(
        [
            half_y**2 + half_z**2,
            half_x**2 + half_z**2,
            half_x**2 + half_y**2,
        ]
    )


def _t2s_time_embeddings(
    experiment_time: float,
    first_element_duration: float,
) -> np.ndarray:
    """Time codes at the starts of the paper's 20 shooting intervals."""

    stage_times = np.empty(PAPER_ACADOS_INTERVALS, dtype=float)
    stage_times[0] = experiment_time
    if PAPER_ACADOS_INTERVALS > 1:
        stage_times[1:] = (
            experiment_time
            + first_element_duration
            + PAPER_NODE_DT * np.arange(PAPER_ACADOS_INTERVALS - 1)
        )
    return np.stack(
        [
            time_embedding_np(
                stage_time / PAPER_T2S_TIME_SCALE_SECONDS,
                d=PAPER_T2S_TIME_EMBEDDING_DIM,
            )
            for stage_time in stage_times
        ]
    )


def _paper_t2s_feature(
    state: np.ndarray,
    force: np.ndarray,
    negative_feet: np.ndarray,
    embedding: np.ndarray,
) -> np.ndarray:
    """Paper 15-D physical feature followed by the T2S time code."""

    return np.concatenate(
        (
            author_code_features_numpy(state, force, negative_feet),
            np.asarray(embedding, dtype=float).reshape(
                PAPER_T2S_TIME_EMBEDDING_DIM
            ),
        )
    )


def _paper_t2s_horizon_features(
    solver: PaperAcadosMPC,
    measured_state: np.ndarray,
    references: np.ndarray,
    negative_feet: np.ndarray,
    contact_table: np.ndarray,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Expansion points from the current warm start (or paper reference)."""

    features = np.empty(
        (PAPER_ACADOS_INTERVALS, PAPER_T2S_INPUT_DIM), dtype=float
    )
    for stage in range(PAPER_ACADOS_INTERVALS):
        if solver.initialized:
            state = np.asarray(solver.solver.get(stage, "x"), dtype=float)
            force = np.asarray(solver.solver.get(stage, "u"), dtype=float)
        else:
            state = np.asarray(references[:, stage], dtype=float)
            force = paper_force_reference(contact_table[:, stage])
        if stage == 0:
            state = np.asarray(measured_state, dtype=float)
        features[stage] = _paper_t2s_feature(
            state,
            force,
            negative_feet[:, stage],
            embeddings[stage],
        )
    return features


def _t2s_training_record(result, ready_step: int | float, activated: bool) -> dict:
    return {
        "version": result.version,
        "trigger_step": result.trigger_step,
        "ready_step": ready_step,
        "activated": activated,
        "train_ms": 1000.0 * result.train_seconds,
        "fast_train_ms": 1000.0 * result.fast_seconds,
        "slow_train_ms": 1000.0 * result.slow_seconds,
        "contains_fast": bool(result.fast_losses),
        "contains_slow": bool(result.slow_losses),
        "fast_final_loss": (
            result.fast_losses[-1] if result.fast_losses else np.nan
        ),
        "slow_final_loss": (
            result.slow_losses[-1] if result.slow_losses else np.nan
        ),
    }


def _disable_nonflat_world_geometry(simulation: MuJoCo_GO2_Model) -> int:
    """Retain the plane and remove the upstream demo obstacles."""

    disabled = 0
    for geometry_id in range(simulation.model.ngeom):
        is_world = int(simulation.model.geom_bodyid[geometry_id]) == 0
        is_box = int(simulation.model.geom_type[geometry_id]) == int(
            mj.mjtGeom.mjGEOM_BOX
        )
        if is_world and is_box:
            simulation.model.geom_contype[geometry_id] = 0
            simulation.model.geom_conaffinity[geometry_id] = 0
            simulation.model.geom_pos[geometry_id, 2] = -100.0
            simulation.model.geom_rgba[geometry_id, 3] = 0.0
            disabled += 1
    return disabled


def _set_constant_paper_friction(simulation: MuJoCo_GO2_Model) -> None:
    """Set the paper's high-friction material on the floor and all toes."""

    # The Unitree MJCF gives toe contacts priority one.  Updating only the
    # floor would therefore leave the effective contact at the toe default.
    for geometry_name in ("floor", "FL", "FR", "RL", "RR"):
        geometry_id = mj.mj_name2id(
            simulation.model, mj.mjtObj.mjOBJ_GEOM, geometry_name
        )
        if geometry_id < 0:
            raise RuntimeError(f"missing MuJoCo contact geom: {geometry_name}")
        simulation.model.geom_friction[geometry_id] = PAPER_HIGH_FRICTION_VECTOR


def _paper_world_state(go2: PinGo2Model) -> np.ndarray:
    """Return the body state used by the authors' local planner.

    The paper calls this a centroidal model, but the public controller fills
    its position state with the floating-base/body position.  The upstream
    ``compute_com_x_vec`` helper instead inserts the whole-robot CoM.  In the
    symmetric stance used here that CoM is about 1.9 cm below the base origin,
    and it moves as the legs swing.  Mixing those two conventions changes the
    model state and the meaning of the 0.30 m body-height reference.
    """

    rotation = np.asarray(go2.R_body_to_world, dtype=float)
    return np.concatenate(
        (
            np.asarray(go2.current_config.base_pos, dtype=float),
            go2.current_config.compute_euler_angle_world(),
            rotation @ np.asarray(go2.current_config.base_vel, dtype=float),
            rotation
            @ np.asarray(go2.current_config.base_ang_vel, dtype=float),
        )
    )


def _initialize_paper_standing_pose(
    go2: PinGo2Model, low_level: PaperQuadSdkLowLevel
) -> np.ndarray:
    """Create a symmetric, static 0.30 m body-height initial condition."""

    initial_feet = np.stack(go2.get_foot_placement_in_world())
    # The MJCF collision feet are 22 mm spheres.  The upstream Pinocchio
    # default pose places their centres at roughly 5 mm, i.e. 17 mm inside the
    # floor.  Reusing that z coordinate in IK makes the stand phase lift the
    # body to about 0.317 m and introduces a pitch bias before walking starts.
    # Put the sphere centres tangent to the ground; the paper's separate
    # 20 mm toe-radius correction remains in the centroidal lever calculation.
    initial_feet[:, 2] = MUJOCO_TOE_COLLISION_RADIUS
    q_initial = np.asarray(go2.current_config.get_q(), dtype=float).copy()
    q_initial[2] = TARGET_HEIGHT
    go2.update_model(q_initial, np.zeros(18))
    q_initial[7:19] = low_level._inverse_kinematics(go2, initial_feet)
    go2.update_model(q_initial, np.zeros(18))
    return q_initial


def _negative_world_foot_levers(go2: PinGo2Model) -> np.ndarray:
    """Authors' ``-foot_positions_body`` in world-aligned coordinates."""

    base_position = np.asarray(go2.current_config.base_pos, dtype=float)
    foot_positions = np.stack(go2.get_foot_placement_in_world())
    # Quad-SDK plans the centre of each spherical toe, then shifts its GRF
    # application point down by the published 2 cm toe radius.
    foot_positions[:, 2] -= 0.02
    return -(foot_positions - base_position).reshape(12)


def _interpolate_body_plan(
    body_plan: np.ndarray,
    elapsed: float,
    first_element_duration: float,
) -> np.ndarray:
    """Interpolate the current source-grid body plan for the 500 Hz WBC."""

    body_plan = np.asarray(body_plan, dtype=float).reshape(
        12, PAPER_ACADOS_INTERVALS + 1
    )
    elapsed = max(0.0, float(elapsed))
    first_element_duration = float(first_element_duration)
    if elapsed <= first_element_duration:
        lower = 0
        phase = elapsed / max(first_element_duration, 1.0e-12)
    else:
        remainder = elapsed - first_element_duration
        lower = 1 + int(remainder // PAPER_NODE_DT)
        lower = min(lower, PAPER_ACADOS_INTERVALS - 1)
        phase = (remainder - (lower - 1) * PAPER_NODE_DT) / PAPER_NODE_DT
    phase = float(np.clip(phase, 0.0, 1.0))
    return (1.0 - phase) * body_plan[:, lower] + phase * body_plan[:, lower + 1]


def _shift_body_plan_for_new_grid(
    body_plan: np.ndarray,
    plan_index_difference: int,
) -> np.ndarray:
    """Apply the source local planner's 30 ms trajectory shift."""

    shifted = np.asarray(body_plan, dtype=float).reshape(
        12, PAPER_ACADOS_INTERVALS + 1
    ).copy()
    difference = max(0, int(plan_index_difference))
    for _ in range(min(difference, PAPER_ACADOS_INTERVALS)):
        # Match ``topRows(N - 1) = bottomRows(N - 1)``: advance every
        # available state and hold the old terminal state at the tail.
        shifted[:, :-1] = shifted[:, 1:]
    return shifted


def _horizon_negative_feet(
    go2: PinGo2Model,
    references_world: np.ndarray,
    contact_nodes: np.ndarray,
    body_plan_world: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reproduce the paper planner's stance footholds and body-frame levers.

    A stance toe is fixed in the world while the predicted body moves.  At a
    touchdown we place the toe below the centre of the nominal hip trajectory
    over the following stance, which is the flat, straight-line specialization
    of the public planner's minimum-enclosing-circle rule.
    """

    references_world = np.asarray(references_world, dtype=float).reshape(
        12, PAPER_ACADOS_INTERVALS + 1
    )
    if body_plan_world is None:
        body_plan_world = references_world.copy()
    else:
        body_plan_world = np.asarray(body_plan_world, dtype=float).reshape(
            12, PAPER_ACADOS_INTERVALS + 1
        ).copy()
        # The public local planner overwrites row zero of the shifted previous
        # solution with the measured state before planning the next footholds.
        body_plan_world[:, 0] = _paper_world_state(go2)
    contact_nodes = np.asarray(contact_nodes, dtype=bool).reshape(
        4, PAPER_ACADOS_INTERVALS + 1
    )
    measured_feet = np.stack(go2.get_foot_placement_in_world())
    held_footholds = measured_feet.copy()
    hip_offsets = np.stack(
        [go2.get_hip_offset(leg) for leg in ("FL", "FR", "RL", "RR")]
    )
    result = np.zeros((12, PAPER_ACADOS_INTERVALS), dtype=float)
    legs = ("FL", "FR", "RL", "RR")
    planned_touchdowns: dict[str, np.ndarray] = {}
    for stage in range(PAPER_ACADOS_INTERVALS):
        node = stage + 1
        for leg in range(4):
            new_contact = contact_nodes[leg, node] and (
                not contact_nodes[leg, node - 1]
            )
            if new_contact:
                end = node + 1
                while (
                    end < PAPER_ACADOS_INTERVALS + 1
                    and contact_nodes[leg, end]
                ):
                    end += 1
                nominal_stance_nodes = int(
                    round(AUTHOR_GAIT_PERIOD * AUTHOR_GAIT_DUTY / PAPER_NODE_DT)
                )
                end = max(end, node + nominal_stance_nodes)
                first_body = body_plan_world[0:3, node]
                if end - 1 <= PAPER_ACADOS_INTERVALS:
                    last_body = body_plan_world[0:3, end - 1]
                    last_rpy = body_plan_world[3:6, end - 1]
                else:
                    extra_time = (end - 1 - PAPER_ACADOS_INTERVALS) * PAPER_NODE_DT
                    last_body = (
                        body_plan_world[0:3, -1]
                        + extra_time * body_plan_world[6:9, -1]
                    )
                    last_rpy = body_plan_world[3:6, -1]
                first_hip = (
                    first_body
                    + rotation_zyx_numpy(body_plan_world[3:6, node])
                    @ hip_offsets[leg]
                )
                last_hip = last_body + rotation_zyx_numpy(last_rpy) @ hip_offsets[leg]
                held_footholds[leg] = 0.5 * (first_hip + last_hip)
                height = max(float(body_plan_world[2, node]), 0.0)
                centrifugal = (
                    height
                    / 9.81
                    * np.cross(
                        body_plan_world[6:9, node],
                        references_world[9:12, node],
                    )
                )
                velocity_tracking = np.sqrt(height / 9.81) * (
                    body_plan_world[6:9, node]
                    - references_world[6:9, node]
                )
                held_footholds[leg] += centrifugal + velocity_tracking
                held_footholds[leg, 2] = 0.02
                if legs[leg] not in planned_touchdowns:
                    planned_touchdowns[legs[leg]] = held_footholds[leg].copy()

            # The public stack passes ``foot_positions_body_.row(i + 1)`` to
            # every interval, including rows at which a foot is in swing.  A
            # zero GRF makes most swing-foot levers dynamically irrelevant,
            # but the distinction is essential at a liftoff boundary: u_i is
            # still a stance force while row i+1 is the first swing node.  A
            # previous version zeroed that lever and made the MPC believe the
            # last stance force produced no body moment.  Preserve the last
            # foothold here; at liftoff it is exactly the source foot plan's
            # first swing position.
            grf_point = held_footholds[leg].copy()
            grf_point[2] -= 0.02
            # Like getFootPositionsBodyFrame() in the public stack, the lever
            # is formed around the previous predicted body plan, not around
            # the desired reference trajectory.
            lever = grf_point - body_plan_world[0:3, node]
            result[3 * leg : 3 * leg + 3, stage] = -lever
    return result, planned_touchdowns


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(
    config: PaperAcadosExperimentConfig,
    records: list[dict],
    status: str,
    *,
    plant_mass: float,
    disabled_obstacles: int,
) -> dict:
    command_speed_error = np.array([row["vx_error"] for row in records])
    mpc_reference_speed_error = np.array(
        [row["vx_mpc_reference_error"] for row in records]
    )
    height_error = np.array([row["z_error"] for row in records])
    position_error = np.array(
        [[row["x_error"], row["y_error"]] for row in records]
    )
    attitude = np.array(
        [[row["roll"], row["pitch"]] for row in records]
    )
    solve_ms = np.array([row["solve_ms"] for row in records])
    end_to_end_ms = np.array([row["mpc_end_to_end_ms"] for row in records])
    solver_internal_ms = np.array(
        [row.get("solver_internal_end_to_end_ms", row["solve_ms"]) for row in records]
    )
    t2s_feature_ms = np.array(
        [row.get("t2s_feature_ms", 0.0) for row in records]
    )
    t2s_linearization_ms = np.array(
        [row.get("t2s_linearization_ms", 0.0) for row in records]
    )
    body_sliding_friction = np.array(
        [row.get("friction_body_sliding", PAPER_HIGH_FRICTION_VECTOR[0]) for row in records]
    )
    minimum_foot_sliding_friction = np.array(
        [row.get("friction_foot_min_sliding", PAPER_HIGH_FRICTION_VECTOR[0]) for row in records]
    )
    deadline_ms = 1000.0 / PAPER_MPC_HZ
    fifty_hz_period_ms = 1000.0 / 50.0
    finite = bool(records)
    simulated_seconds = float(records[-1]["time"]) if finite else 0.0
    # The last MPC sample can precede the requested endpoint by one 200 Hz
    # control period.  Anything shorter, or any non-completed status, is a
    # truncated trajectory and must not be used as a formal tracking result.
    completed_full_window = (
        finite
        and status == "completed"
        and simulated_seconds >= config.duration - 1.0 / PAPER_MPC_HZ - 1.0e-9
    )
    return {
        **asdict(config),
        "status": status,
        "simulated_seconds": simulated_seconds,
        "requested_seconds": config.duration,
        "valid_for_comparison": completed_full_window,
        "metrics_scope": (
            "full_requested_trial"
            if completed_full_window
            else "truncated_diagnostic_only"
        ),
        "failure_time_seconds": (
            None if completed_full_window else simulated_seconds
        ),
        "samples": len(records),
        # Retain the old key as an explicit alias for the error to the final
        # experiment command.  This is not the same as tracking the filtered
        # reference actually supplied to MPC during startup.
        "speed_rmse_mps": float(np.sqrt(np.mean(command_speed_error**2))) if finite else None,
        "speed_command_rmse_mps": float(np.sqrt(np.mean(command_speed_error**2))) if finite else None,
        "speed_mpc_reference_rmse_mps": (
            float(np.sqrt(np.mean(mpc_reference_speed_error**2)))
            if finite
            else None
        ),
        "height_rmse_m": float(np.sqrt(np.mean(height_error**2))) if finite else None,
        "position_x_rmse_m": float(np.sqrt(np.mean(position_error[:, 0] ** 2))) if finite else None,
        "position_y_rmse_m": float(np.sqrt(np.mean(position_error[:, 1] ** 2))) if finite else None,
        "position_xy_rmse_m": float(np.sqrt(np.mean(np.sum(position_error**2, axis=1)))) if finite else None,
        "roll_pitch_rmse_rad": float(np.sqrt(np.mean(attitude**2))) if finite else None,
        "final_x_error_m": float(position_error[-1, 0]) if finite else None,
        "final_y_error_m": float(position_error[-1, 1]) if finite else None,
        "mean_solve_ms": float(np.mean(solve_ms)) if finite else None,
        "p95_solve_ms": float(np.percentile(solve_ms, 95)) if finite else None,
        "max_solve_ms": float(np.max(solve_ms)) if finite else None,
        "mean_solver_internal_end_to_end_ms": (
            float(np.mean(solver_internal_ms)) if finite else None
        ),
        "mean_t2s_feature_ms": (
            float(np.mean(t2s_feature_ms)) if finite else None
        ),
        "mean_t2s_linearization_ms": (
            float(np.mean(t2s_linearization_ms)) if finite else None
        ),
        "mean_mpc_end_to_end_ms": float(np.mean(end_to_end_ms)) if finite else None,
        "p95_mpc_end_to_end_ms": float(np.percentile(end_to_end_ms, 95)) if finite else None,
        "max_mpc_end_to_end_ms": float(np.max(end_to_end_ms)) if finite else None,
        "mean_effective_compute_hz": (
            float(1000.0 / np.mean(end_to_end_ms))
            if finite and np.mean(end_to_end_ms) > 0.0
            else None
        ),
        "mpc_deadline_ms": deadline_ms,
        "mpc_deadline_miss_count": int(np.count_nonzero(end_to_end_ms > deadline_ms)) if finite else 0,
        "mpc_deadline_miss_rate": float(np.mean(end_to_end_ms > deadline_ms)) if finite else None,
        "fifty_hz_period_ms": fifty_hz_period_ms,
        "over_fifty_hz_period_count": (
            int(np.count_nonzero(end_to_end_ms > fifty_hz_period_ms))
            if finite
            else 0
        ),
        "over_fifty_hz_period_rate": (
            float(np.mean(end_to_end_ms > fifty_hz_period_ms))
            if finite
            else None
        ),
        "mean_qp_iterations": float(np.mean([row["qp_iterations"] for row in records])) if finite else None,
        "solver_retry_count": int(sum(row["solver_retry_count"] for row in records)),
        "ssi_updates": int(sum(row.get("ssi_updated", 0) for row in records)),
        "ssi_skipped_contact_change": int(
            sum(row.get("ssi_skipped_contact_change", 0) for row in records)
        ),
        "final_alpha_l2": (
            float(records[-1].get("alpha_l2", 0.0)) if finite else 0.0
        ),
        "t2s_learning_samples": int(
            sum(row.get("t2s_sample_added", 0) for row in records)
        ),
        "t2s_jobs_submitted": int(
            sum(row.get("t2s_job_submitted", 0) for row in records)
        ),
        "t2s_final_active_model_version": (
            int(records[-1].get("t2s_model_version", 0)) if finite else 0
        ),
        "friction_body_sliding_min_encountered": (
            float(np.min(body_sliding_friction)) if finite else None
        ),
        "friction_body_sliding_max_encountered": (
            float(np.max(body_sliding_friction)) if finite else None
        ),
        "friction_body_sliding_mean_encountered": (
            float(np.mean(body_sliding_friction)) if finite else None
        ),
        "friction_foot_sliding_min_encountered": (
            float(np.min(minimum_foot_sliding_friction)) if finite else None
        ),
        "paper_protocol": {
            "deliberate_difference": (
                f"acados_{config.acados_nlp_solver}_instead_of_CasADi_IPOPT"
            ),
            "physics_engine": "MuJoCo",
            "payload_kg": config.payload_mass_kg,
            "payload": (
                {
                    "type": "fixed_rigid_trunk_payload",
                    "mass_kg": config.payload_mass_kg,
                    "known_to_controller": False,
                    "mount_position_body_m": [0.0, 0.0, 0.13],
                    "inertia_diagonal_kg_m2": (
                        _rigid_payload_inertia_diagonal(
                            _paper_rigid_payload_config(
                                config.payload_mass_kg
                            )
                        ).tolist()
                    ),
                }
                if config.payload_mass_kg > 0.0
                else None
            ),
            "terrain": (
                "flat_alternating_random_red_blue_friction"
                if config.terrain == "alternating_random_friction"
                else "flat_constant_paper_high_friction"
            ),
            "plant_friction": (
                None
                if config.terrain == "alternating_random_friction"
                else PAPER_HIGH_FRICTION_VECTOR.tolist()
            ),
            "friction_field": (
                {
                    "known_to_controller": False,
                    "process": "alternating_random_red_blue_discrete_patches",
                    "seed": config.seed + 300_000,
                    "global_sliding_range": [0.05, 0.50],
                    "red_sliding_range": [
                        config.friction_red_sliding_min,
                        config.friction_red_sliding_max,
                    ],
                    "blue_sliding_range": [
                        config.friction_blue_sliding_min,
                        config.friction_blue_sliding_max,
                    ],
                    "red_length_range_m": [
                        config.friction_red_length_min,
                        config.friction_red_length_max,
                    ],
                    "blue_length_range_m": [
                        config.friction_blue_length_min,
                        config.friction_blue_length_max,
                    ],
                    "warmup_end_x_m": config.friction_warmup_end_x,
                    "spatial_resolution_m": config.friction_tile_length,
                    "friction_vector_mapping": "[mu_s,mu_s,0.02*mu_s]",
                    "boundary_transition": "discontinuous",
                }
                if config.terrain == "alternating_random_friction"
                else None
            ),
            "target_forward_speed_mps": config.target_speed,
            "target_body_height_m": TARGET_HEIGHT,
            "mpc_hz": PAPER_MPC_HZ,
            "low_level_hz": PAPER_LOW_LEVEL_HZ,
            "shooting_intervals": PAPER_ACADOS_INTERVALS,
            "node_dt_seconds": PAPER_NODE_DT,
            "maximum_horizon_seconds": PAPER_ACADOS_HORIZON_SECONDS,
            "discretization": "paper_forward_euler",
            "rti_iterations_per_cycle": config.rti_iterations_per_cycle,
            "state_dimension": 12,
            "control_dimension": 12,
            "friction_cone_mu": PAPER_FRICTION_COEFFICIENT,
            "minimum_stance_fz_N": PAPER_MIN_STANCE_FZ,
            "maximum_stance_fz_N": PAPER_MAX_STANCE_FZ,
            "model_mass_kg": PAPER_MODEL_MASS,
            "plant_mass_kg": plant_mass,
            "dry_plant_mass_kg": plant_mass - config.payload_mass_kg,
            "grf_reference_mass_kg": PAPER_FORCE_REFERENCE_MASS,
            "published_model_mass_kg": PAPER_MODEL_MASS,
            "published_grf_reference_mass_kg": PAPER_FORCE_REFERENCE_MASS,
            "Q_diagonal": PAPER_STATE_COST_DIAGONAL.tolist(),
            "R_diagonal": PAPER_INPUT_COST_DIAGONAL.tolist(),
            "gait_period_seconds": AUTHOR_GAIT_PERIOD,
            "gait_duty": AUTHOR_GAIT_DUTY,
            "gait_phase_offsets": [0.0, 0.5, 0.5, 0.0],
            "low_level_controller": (
                "source_derived_QuadSDK_inverse_dynamics"
                if config.low_level == "quad_sdk"
                else "MuJoCo_operational_space_GRF_adapter"
            ),
            "actuator": "ideal_joint_torque_no_added_noise_or_lag",
            "initial_condition": "symmetric_static_pose_at_target_body_height",
            "pretrial_stand_seconds": PRETRIAL_STAND_SECONDS,
            "outer_path_follower": {
                "publish_rate_hz": PAPER_COMMAND_PUBLISH_HZ,
                "forward_speed_mps": config.target_speed,
                "lateral_position_gain": PAPER_LATERAL_POSITION_GAIN,
                "heading_gain": PAPER_HEADING_GAIN,
            },
            "command_velocity_filter_new_weight": PAPER_COMMAND_FILTER_NEW_WEIGHT,
            "disabled_upstream_obstacles": disabled_obstacles,
        },
        "ssi_protocol": {
            "input_dimension": 15,
            "input_layout": "rpy3_world_velocity3_body_omega3_JT_u_over_100_6",
            "output_dimension": 6,
            "output_layout": "residual_generalized_force3_body_torque3",
            "random_feature_count": PAPER_RFF_COUNT,
            "kernel_std": PAPER_RFF_KERNEL_STD,
            "learning_rate": PAPER_RFF_LEARNING_RATE,
            "initial_alpha": "zero",
            "update_rate_hz": PAPER_MPC_HZ,
            "contact_transition_updates": "skipped_as_public_source_code",
        },
        "t2s_protocol": {
            "physical_input_dimension": 15,
            "physical_input_layout": (
                "rpy3_world_velocity3_body_omega3_JT_u_over_100_6"
            ),
            "time_embedding_dimension": PAPER_T2S_TIME_EMBEDDING_DIM,
            "network_input_dimension": PAPER_T2S_INPUT_DIM,
            "output_dimension": 6,
            "output_layout": "residual_generalized_force3_body_torque3",
            "hidden_dimension": PAPER_T2S_HIDDEN_DIM,
            "fast_update_every_mpc_cycles": config.t2s_fast_update_every,
            "slow_update_every_mpc_cycles": config.t2s_slow_update_every,
            "fast_batch_size": config.t2s_fast_batch_size,
            "slow_batch_size": config.t2s_slow_batch_size,
            "fast_epochs": config.t2s_fast_epochs,
            "slow_epochs": config.t2s_slow_epochs,
            "fast_learning_rate": config.t2s_fast_learning_rate,
            "slow_learning_rate": config.t2s_slow_learning_rate,
            "training_execution": "asynchronous_worker",
            "network_in_mpc": "stagewise_first_order_realtime_l4casadi",
            "initial_residual": "zero",
            "contact_transition_updates": "skipped_same_as_ssi",
        },
    }


def run_paper_acados_experiment(
    config: PaperAcadosExperimentConfig,
) -> tuple[dict, Path]:
    if config.controller not in {"nominal", "ssi", "t2s"}:
        raise ValueError("controller must be nominal, ssi, or t2s")
    if config.duration <= 0.0 or config.target_speed <= 0.0:
        raise ValueError("duration and target_speed must be positive")
    if config.low_level not in {"quad_sdk", "operational_space"}:
        raise ValueError("low_level must be quad_sdk or operational_space")
    if config.acados_nlp_solver not in {"SQP_RTI", "SQP"}:
        raise ValueError("acados_nlp_solver must be SQP_RTI or SQP")
    if config.rti_iterations_per_cycle <= 0:
        raise ValueError("rti_iterations_per_cycle must be positive")
    if config.execution_gait not in {"author", "adapter"}:
        raise ValueError("execution_gait must be author or adapter")
    if config.terrain not in {
        "constant_paper_friction",
        "alternating_random_friction",
    }:
        raise ValueError(
            "terrain must be constant_paper_friction or "
            "alternating_random_friction"
        )
    if config.payload_mass_kg < 0.0:
        raise ValueError("payload_mass_kg cannot be negative")
    if (
        config.payload_mass_kg > 0.0
        and config.terrain != "alternating_random_friction"
    ):
        raise ValueError(
            "The rigid payload extension currently requires "
            "alternating_random_friction terrain"
        )
    t2s_positive_integers = (
        config.t2s_fast_update_every,
        config.t2s_slow_update_every,
        config.t2s_fast_batch_size,
        config.t2s_slow_batch_size,
        config.t2s_fast_epochs,
        config.t2s_slow_epochs,
        config.t2s_torch_threads,
    )
    if any(value <= 0 for value in t2s_positive_integers):
        raise ValueError("T2S schedules, batch sizes, epochs, and threads must be positive")
    if min(
        config.t2s_fast_learning_rate, config.t2s_slow_learning_rate
    ) <= 0.0:
        raise ValueError("T2S learning rates must be positive")

    np.random.seed(config.seed)
    project_root = Path(__file__).resolve().parents[1]
    go2 = PinGo2Model()
    low_level = PaperQuadSdkLowLevel(
        update_period=1.0 / PAPER_LOW_LEVEL_HZ
    )
    operational_space_low_level = LegController()
    q_initial = _initialize_paper_standing_pose(go2, low_level)
    if config.terrain == "alternating_random_friction":
        simulation = MuJoCoRandomFrictionGo2Model(
            _alternating_friction_config(config),
            rigid_payload=(
                _paper_rigid_payload_config(config.payload_mass_kg)
                if config.payload_mass_kg > 0.0
                else None
            ),
        )
        disabled_obstacles = simulation.disabled_upstream_geometries
    else:
        simulation = MuJoCo_GO2_Model()
        disabled_obstacles = _disable_nonflat_world_geometry(simulation)
        _set_constant_paper_friction(simulation)
    simulation.update_with_q_pin(q_initial)
    simulation.model.opt.timestep = SIM_DT
    simulation.data.qvel[:] = 0.0
    simulation.update_pin_with_mujoco(go2)
    plant_mass = float(np.sum(simulation.model.body_mass))
    if config.payload_mass_kg > 0.0:
        payload_body_id = mj.mj_name2id(
            simulation.model, mj.mjtObj.mjOBJ_BODY, "payload_fixed"
        )
        if payload_body_id < 0 or not np.isclose(
            simulation.model.body_mass[payload_body_id],
            config.payload_mass_kg,
            atol=1.0e-12,
        ):
            raise RuntimeError("MuJoCo payload body mass is missing or incorrect")
    contact_reader = Go2FootContactReader(simulation.model)
    if not np.isclose(
        plant_mass,
        contact_reader.robot_mass + config.payload_mass_kg,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "MuJoCo total mass does not match its dry robot plus payload"
        )
    contact_accumulator = ContactWrenchIntervalAccumulator()

    if config.execution_gait == "author":
        gait = _AuthorDiscreteTrottingGait(
            # Our pretrial uses a separate stand controller and does not run
            # the paper local planner.  Its first-plan timestamp therefore
            # corresponds to the start of the scored walking interval.
            PRETRIAL_STAND_SECONDS,
            config.target_speed,
        )
    else:
        gait = Gait(3.0, 0.60)
    standing_gait = _StandingGait()
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
    residual_model = None
    realtime_model = None
    trainer = None
    if config.controller == "t2s":
        torch.manual_seed(config.seed)
        residual_model = BoundedTwoSpeedMLP(
            PAPER_T2S_INPUT_DIM,
            hidden_dim=PAPER_T2S_HIDDEN_DIM,
            output_dim=6,
        )
        _zero_t2s_output_layer(residual_model)
        realtime_model = RealTimeL4CasADi(
            residual_model,
            approximation_order=1,
            name=f"rt_go2_paper_t2s_seed{config.seed}",
        )
    solver = PaperAcadosMPC(
        project_root=project_root,
        random_features=random_features,
        realtime_residual=realtime_model,
        time_embedding_dim=PAPER_T2S_TIME_EMBEDDING_DIM,
        name_suffix=(
            f"source_mass_{config.acados_nlp_solver.lower()}_single_seed_"
            f"{config.seed}"
        ),
        model_mass=PAPER_MODEL_MASS,
        nlp_solver_type=config.acados_nlp_solver,
        rti_iterations_per_cycle=config.rti_iterations_per_cycle,
    )
    residual_output_scale = solver.dynamics.residual_output_scale
    if residual_model is not None:
        trainer = AsyncRealtimeT2STrainer(
            residual_model,
            input_dim=PAPER_T2S_INPUT_DIM,
            hidden_dim=PAPER_T2S_HIDDEN_DIM,
            output_dim=6,
            fast_learning_rate=config.t2s_fast_learning_rate,
            slow_learning_rate=config.t2s_slow_learning_rate,
            fast_epochs=config.t2s_fast_epochs,
            slow_epochs=config.t2s_slow_epochs,
            torch_threads=config.t2s_torch_threads,
            bounded_output=True,
        )

    force_hold = np.zeros(12)
    torque_hold = np.zeros(12)
    low_level_result = None
    previous_state = None
    previous_force = None
    previous_negative_feet = None
    previous_contact = None
    previous_body_plan = None
    previous_plan_index: int | None = None
    body_plan_hold = None
    body_plan_start_time = 0.0
    body_plan_first_duration = PAPER_NODE_DT
    initial_x = None
    initial_y = None
    initial_yaw = None
    records: list[dict] = []
    status = "completed"
    walking_started = False
    filtered_body_command = np.zeros(3)
    replay_rng = np.random.default_rng(config.seed + 700_000)
    replay_inputs: list[np.ndarray] = []
    replay_targets: list[np.ndarray] = []
    training_records: list[dict] = []
    previous_feature = None
    active_model_version = 0
    next_model_version = 1
    mpc_index = 0

    total_seconds = PRETRIAL_STAND_SECONDS + config.duration
    for sim_step in range(int(round(total_seconds * SIM_HZ))):
        time_now = float(simulation.data.time)
        in_warmup = time_now + 0.5 * SIM_DT < PRETRIAL_STAND_SECONDS
        experiment_time = max(0.0, time_now - PRETRIAL_STAND_SECONDS)
        active_gait = standing_gait if in_warmup else gait

        if not in_warmup and not walking_started:
            previous_state = None
            previous_force = None
            previous_negative_feet = None
            previous_contact = None
            previous_feature = None
            previous_body_plan = None
            previous_plan_index = None
            contact_accumulator.average_and_reset()
            walking_started = True

        if sim_step % SIM_STEPS_PER_MPC == 0:
            cycle_start = time.perf_counter()
            if trainer is not None:
                completed_training = trainer.poll()
                if completed_training is not None:
                    if (
                        completed_training.error is not None
                        or completed_training.state_dict is None
                    ):
                        raise RuntimeError(
                            "Go2 paper-style T2S training failed:\n"
                            + str(completed_training.error)
                        )
                    load_numpy_state_dict(
                        residual_model, completed_training.state_dict
                    )
                    active_model_version = completed_training.version
                    training_records.append(
                        _t2s_training_record(
                            completed_training,
                            ready_step=mpc_index,
                            activated=True,
                        )
                    )
            actual_contact = contact_accumulator.average_and_reset()
            simulation.update_pin_with_mujoco(go2)
            world_state = _paper_world_state(go2)
            state = world_state_to_paper_state(world_state)
            if initial_x is None and not in_warmup:
                initial_x = float(state[0])
                initial_y = float(state[1])
                initial_yaw = float(state[5])

            # The released experiment's path_following.py publishes at
            # 100 Hz. cmdVelCallback applies the 0.1 exponential update only
            # when one of those messages arrives; applying it at the 200 Hz
            # MPC rate doubles the command ramp and is not source-equivalent.
            command_publish_stride = SIM_HZ // PAPER_COMMAND_PUBLISH_HZ
            if sim_step % command_publish_stride == 0:
                if in_warmup:
                    # The source cmdVel callback keeps receiving the requested
                    # walking command while the mode remains STAND.
                    raw_body_command = np.array(
                        [config.target_speed, 0.0, 0.0], dtype=float
                    )
                else:
                    raw_body_command = path_following_body_command(
                        state,
                        target_speed=config.target_speed,
                        lateral_position_reference=initial_y,
                        heading_reference=initial_yaw,
                        lateral_gain=PAPER_LATERAL_POSITION_GAIN,
                        heading_gain=PAPER_HEADING_GAIN,
                    )
                filtered_body_command = (
                    PAPER_COMMAND_FILTER_NEW_WEIGHT * raw_body_command
                    + (1.0 - PAPER_COMMAND_FILTER_NEW_WEIGHT)
                    * filtered_body_command
                )
            if isinstance(active_gait, _AuthorDiscreteTrottingGait):
                active_gait.target_speed = filtered_body_command[0]
                planning_grid_elapsed = max(
                    0.0, time_now - active_gait.start_time
                )
            else:
                planning_grid_elapsed = experiment_time
            first_element_duration = paper_first_element_duration(
                planning_grid_elapsed
            )
            plan_index = int(
                np.floor(planning_grid_elapsed / PAPER_NODE_DT + 1.0e-9)
            )
            if previous_body_plan is not None and previous_plan_index is not None:
                previous_body_plan = _shift_body_plan_for_new_grid(
                    previous_body_plan,
                    plan_index - previous_plan_index,
                )
            previous_plan_index = plan_index
            current_negative_feet = _negative_world_foot_levers(go2)
            current_contact = np.asarray(
                active_gait.compute_current_mask(time_now), dtype=bool
            ).reshape(4)

            update_target = np.full(6, np.nan)
            prediction_error = np.full(6, np.nan)
            ssi_updated = False
            skipped_contact = False
            t2s_sample_added = False
            if (
                (learner is not None or residual_model is not None)
                and not in_warmup
                and previous_state is not None
            ):
                if not np.array_equal(current_contact, previous_contact):
                    skipped_contact = True
                else:
                    if learner is not None:
                        update_target, prediction_error = learner.update(
                            previous_state,
                            state,
                            previous_force,
                            previous_negative_feet,
                            1.0 / PAPER_MPC_HZ,
                        )
                        ssi_updated = True
                    else:
                        update_target = paper_residual_wrench_target_numpy(
                            previous_state,
                            state,
                            previous_force,
                            previous_negative_feet,
                            1.0 / PAPER_MPC_HZ,
                        )
                        previous_prediction = (
                            exact_model_output(
                                residual_model, previous_feature[None, :]
                            )[0]
                            * residual_output_scale
                        )
                        prediction_error = update_target - previous_prediction
                        replay_inputs.append(
                            previous_feature.astype(np.float32)
                        )
                        replay_targets.append(
                            (update_target / residual_output_scale).astype(
                                np.float32
                            )
                        )
                        if len(replay_inputs) > PAPER_T2S_REPLAY_CAPACITY:
                            replay_inputs.pop(0)
                            replay_targets.pop(0)
                        t2s_sample_added = True

            references_world = author_twist_reference(
                world_state,
                target_speed=filtered_body_command[0],
                lateral_speed=filtered_body_command[1],
                yaw_rate=filtered_body_command[2],
                first_element_duration=first_element_duration,
            )
            references_world = references_world[
                :, : PAPER_ACADOS_INTERVALS + 1
            ]
            contact_nodes = active_gait.compute_contact_table(
                time_now, PAPER_NODE_DT, PAPER_ACADOS_INTERVALS + 1
            ).astype(bool)
            contact_horizon = contact_nodes[:, :PAPER_ACADOS_INTERVALS]
            negative_feet_horizon, planned_touchdowns = _horizon_negative_feet(
                go2,
                references_world,
                contact_nodes,
                body_plan_world=previous_body_plan,
            )
            if isinstance(active_gait, _AuthorDiscreteTrottingGait):
                active_gait.set_planned_touchdowns(planned_touchdowns)
            references_paper = references_world.copy()
            time_embeddings = None
            realtime_parameters = None
            t2s_feature_ms = 0.0
            t2s_linearization_ms = 0.0
            if residual_model is not None:
                time_embeddings = _t2s_time_embeddings(
                    experiment_time, first_element_duration
                )
                t2s_feature_start = time.perf_counter()
                expansion_features = _paper_t2s_horizon_features(
                    solver,
                    state,
                    references_paper,
                    negative_feet_horizon,
                    contact_horizon,
                    time_embeddings,
                )
                t2s_feature_ms = 1000.0 * (
                    time.perf_counter() - t2s_feature_start
                )
                t2s_linearization_start = time.perf_counter()
                realtime_parameters, _, _ = first_order_parameters(
                    residual_model, expansion_features
                )
                t2s_linearization_ms = 1000.0 * (
                    time.perf_counter() - t2s_linearization_start
                )
            result = solver.solve(
                state,
                references_paper,
                negative_feet_horizon,
                contact_horizon,
                first_element_duration=first_element_duration,
                alpha=(learner.alpha if learner is not None else None),
                time_embeddings=time_embeddings,
                realtime_parameters=realtime_parameters,
            )
            if result.status != 0 or not np.all(np.isfinite(result.force)):
                status = f"solver_failed_status_{result.status}"
                break
            force_hold = result.force
            previous_body_plan = result.states.copy()
            body_plan_hold = result.states.copy()
            body_plan_start_time = time_now
            body_plan_first_duration = first_element_duration
            embedding_now = time_embedding_np(
                experiment_time / PAPER_T2S_TIME_SCALE_SECONDS,
                d=PAPER_T2S_TIME_EMBEDDING_DIM,
            )
            transition_feature = (
                _paper_t2s_feature(
                    state,
                    force_hold,
                    current_negative_feet,
                    embedding_now,
                )
                if residual_model is not None
                else None
            )
            if learner is not None:
                predicted_residual = learner.predict(
                    state, force_hold, current_negative_feet
                )
            elif residual_model is not None:
                predicted_residual = (
                    exact_model_output(
                        residual_model, transition_feature[None, :]
                    )[0]
                    * residual_output_scale
                )
            else:
                predicted_residual = np.zeros(6)

            t2s_job_submitted = False
            if residual_model is not None and not in_warmup:
                fast_due = (
                    mpc_index > 0
                    and mpc_index % config.t2s_fast_update_every == 0
                    and len(replay_inputs) >= config.t2s_fast_batch_size
                )
                slow_due = (
                    mpc_index > 0
                    and mpc_index % config.t2s_slow_update_every == 0
                    and len(replay_inputs) >= config.t2s_slow_batch_size
                )
                if fast_due or slow_due:
                    fast_inputs = fast_targets = None
                    slow_inputs = slow_targets = None
                    if fast_due:
                        fast_inputs = np.asarray(
                            replay_inputs[-config.t2s_fast_batch_size :],
                            dtype=np.float32,
                        ).copy()
                        fast_targets = np.asarray(
                            replay_targets[-config.t2s_fast_batch_size :],
                            dtype=np.float32,
                        ).copy()
                    if slow_due:
                        slow_inputs, slow_targets = (
                            sample_replay_without_replacement(
                                replay_rng,
                                replay_inputs,
                                replay_targets,
                                config.t2s_slow_batch_size,
                            )
                        )
                    t2s_job_submitted = trainer.submit(
                        RealtimeT2SJob(
                            version=next_model_version,
                            trigger_step=mpc_index,
                            trigger_wall_time=time.perf_counter(),
                            fast_inputs=fast_inputs,
                            fast_targets=fast_targets,
                            slow_inputs=slow_inputs,
                            slow_targets=slow_targets,
                        )
                    )
                    if t2s_job_submitted:
                        next_model_version += 1

            if not in_warmup:
                x_reference = initial_x + config.target_speed * experiment_time
                measured_foot_positions = np.stack(
                    go2.get_foot_placement_in_world()
                )
                if config.terrain == "alternating_random_friction":
                    body_friction = simulation.friction_at(state[0], state[1])
                    foot_sliding_friction = np.array(
                        [
                            simulation.friction_at(position[0], position[1])[0]
                            for position in measured_foot_positions
                        ],
                        dtype=float,
                    )
                else:
                    body_friction = PAPER_HIGH_FRICTION_VECTOR
                    foot_sliding_friction = np.full(4, body_friction[0])
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
                        "vx_reference_filtered": references_world[6, 0],
                        "vy_reference_filtered": references_world[7, 0],
                        "vx_body_command_filtered": filtered_body_command[0],
                        "vy_body_command_filtered": filtered_body_command[1],
                        "yaw_rate_command_filtered": filtered_body_command[2],
                        "z_reference": TARGET_HEIGHT,
                        "x_reference_diagnostic": x_reference,
                        "y_reference_diagnostic": initial_y,
                        "vx_error": state[6] - config.target_speed,
                        "vx_mpc_reference_error": (
                            state[6] - references_world[6, 0]
                        ),
                        "z_error": state[2] - TARGET_HEIGHT,
                        "x_error": state[0] - x_reference,
                        "y_error": state[1] - initial_y,
                        "friction_body_sliding": float(body_friction[0]),
                        "friction_body_torsional": float(body_friction[1]),
                        "friction_body_rolling": float(body_friction[2]),
                        "friction_foot_min_sliding": float(
                            np.min(foot_sliding_friction)
                        ),
                        "friction_foot_max_sliding": float(
                            np.max(foot_sliding_friction)
                        ),
                        "net_commanded_fx": float(np.sum(force_hold[0::3])),
                        "net_commanded_fy": float(np.sum(force_hold[1::3])),
                        "net_commanded_fz": float(np.sum(force_hold[2::3])),
                        "net_actual_contact_fx": float(actual_contact.net_force_world[0]),
                        "net_actual_contact_fy": float(actual_contact.net_force_world[1]),
                        "net_actual_contact_fz": float(actual_contact.net_force_world[2]),
                        "net_actual_contact_tx": float(actual_contact.net_moment_world[0]),
                        "net_actual_contact_ty": float(actual_contact.net_moment_world[1]),
                        "net_actual_contact_tz": float(actual_contact.net_moment_world[2]),
                        "actual_contact_samples": int(actual_contact.simulation_samples),
                        "mean_actual_contact_count": float(actual_contact.mean_contact_count),
                        "max_abs_joint_torque": float(np.max(np.abs(torque_hold))),
                        "saturated_joint_count": int(
                            np.count_nonzero(
                                np.isclose(
                                    np.abs(torque_hold),
                                    np.tile(np.array([23.7, 23.7, 45.0]), 4),
                                    rtol=0.0,
                                    atol=1.0e-8,
                                )
                            )
                        ),
                        **{
                            f"commanded_{leg.lower()}_{axis}": float(
                                force_hold[3 * leg_index + axis_index]
                            )
                            for leg_index, leg in enumerate(("FL", "FR", "RL", "RR"))
                            for axis_index, axis in enumerate(("fx", "fy", "fz"))
                        },
                        **{
                            f"actual_{leg.lower()}_{axis}": float(
                                actual_contact.foot_forces_world[
                                    leg_index, axis_index
                                ]
                            )
                            for leg_index, leg in enumerate(("FL", "FR", "RL", "RR"))
                            for axis_index, axis in enumerate(("fx", "fy", "fz"))
                        },
                        **{
                            f"measured_foot_{leg.lower()}_{axis}": float(
                                measured_foot_positions[
                                    leg_index, axis_index
                                ]
                            )
                            for leg_index, leg in enumerate(("FL", "FR", "RL", "RR"))
                            for axis_index, axis in enumerate(("x", "y", "z"))
                        },
                        **{
                            f"desired_foot_{leg.lower()}_{axis}": (
                                float(
                                    low_level_result.desired_foot_position[
                                        3 * leg_index + axis_index
                                    ]
                                )
                                if low_level_result is not None
                                else float("nan")
                            )
                            for leg_index, leg in enumerate(("FL", "FR", "RL", "RR"))
                            for axis_index, axis in enumerate(("x", "y", "z"))
                        },
                        **{
                            f"planned_contact_{leg.lower()}": int(
                                current_contact[leg_index]
                            )
                            for leg_index, leg in enumerate(("FL", "FR", "RL", "RR"))
                        },
                        **{
                            f"planned_touchdown_{leg.lower()}_{axis}": float(
                                planned_touchdowns.get(
                                    leg, np.full(3, np.nan)
                                )[axis_index]
                            )
                            for leg in ("FL", "FR", "RL", "RR")
                            for axis_index, axis in enumerate(("x", "y", "z"))
                        },
                        "predicted_body_vx_first": float(result.states[6, 1]),
                        "predicted_body_vx_last": float(result.states[6, -1]),
                        "predicted_body_z_last": float(result.states[2, -1]),
                        "predicted_body_roll_first": float(result.states[3, 1]),
                        "predicted_body_roll_last": float(result.states[3, -1]),
                        "predicted_body_pitch_first": float(result.states[4, 1]),
                        "predicted_body_pitch_last": float(result.states[4, -1]),
                        "predicted_body_omega_x_first": float(result.states[9, 1]),
                        "predicted_body_omega_x_last": float(result.states[9, -1]),
                        "predicted_body_omega_y_first": float(result.states[10, 1]),
                        "predicted_body_omega_y_last": float(result.states[10, -1]),
                        **{
                            f"mpc_stage0_negative_lever_{leg.lower()}_{axis}": float(
                                negative_feet_horizon[
                                    3 * leg_index + axis_index, 0
                                ]
                            )
                            for leg_index, leg in enumerate(LEGS)
                            for axis_index, axis in enumerate("xyz")
                        },
                        "max_negative_foot_lever": float(
                            np.max(np.abs(negative_feet_horizon))
                        ),
                        "solve_ms": result.solve_ms,
                        "solver_internal_end_to_end_ms": result.end_to_end_ms,
                        "t2s_feature_ms": t2s_feature_ms,
                        "t2s_linearization_ms": t2s_linearization_ms,
                        "mpc_end_to_end_ms": 1000.0 * (time.perf_counter() - cycle_start),
                        "solver_status": result.status,
                        "solver_retry_count": result.retry_count,
                        "sqp_iterations": result.sqp_iterations,
                        "qp_iterations": result.qp_iterations,
                        "ssi_updated": int(ssi_updated),
                        "ssi_skipped_contact_change": int(skipped_contact),
                        "alpha_l2": float(np.linalg.norm(learner.alpha)) if learner is not None else 0.0,
                        "t2s_sample_added": int(t2s_sample_added),
                        "t2s_job_submitted": int(t2s_job_submitted),
                        "t2s_model_version": active_model_version,
                        "t2s_model_l2": (
                            float(
                                np.sqrt(
                                    sum(
                                        torch.sum(parameter.detach() ** 2).item()
                                        for parameter in residual_model.parameters()
                                    )
                                )
                            )
                            if residual_model is not None
                            else 0.0
                        ),
                        "prediction_error_l2": float(np.linalg.norm(prediction_error)),
                        **{
                            f"target_residual_{axis}": update_target[index]
                            for index, axis in enumerate(("fx", "fy", "fz", "tx", "ty", "tz"))
                        },
                        **{
                            f"pred_residual_{axis}": predicted_residual[index]
                            for index, axis in enumerate(("fx", "fy", "fz", "tx", "ty", "tz"))
                        },
                    }
                )
                previous_state = state.copy()
                previous_force = force_hold.copy()
                previous_negative_feet = current_negative_feet.copy()
                previous_contact = current_contact.copy()
                previous_feature = (
                    transition_feature.copy()
                    if transition_feature is not None
                    else None
                )
                mpc_index += 1

            if state[2] < 0.12 or max(abs(state[3]), abs(state[4])) > 1.0:
                status = "fell"
                break

        if sim_step % SIM_STEPS_PER_LOW_LEVEL == 0:
            simulation.update_pin_with_mujoco(go2)
            command_yaw = float(
                go2.current_config.compute_euler_angle_world()[2]
            )
            command_cosine = np.cos(command_yaw)
            command_sine = np.sin(command_yaw)
            go2.x_vel_des_world = (
                command_cosine * filtered_body_command[0]
                - command_sine * filtered_body_command[1]
            )
            go2.y_vel_des_world = (
                command_sine * filtered_body_command[0]
                + command_cosine * filtered_body_command[1]
            )
            go2.x_pos_des_world = float(go2.pos_com_world[0])
            go2.y_pos_des_world = float(go2.pos_com_world[1])
            go2.yaw_rate_des_world = filtered_body_command[2]
            # The released robot stack uses its dedicated stand controller
            # before entering inverse-dynamics walking control.  Use the
            # stable all-feet operational-space stand during the unscored
            # pretrial instead of incorrectly running the walking WBC there.
            if config.low_level == "quad_sdk" and not in_warmup:
                desired_body_state = (
                    _interpolate_body_plan(
                        body_plan_hold,
                        time_now - body_plan_start_time,
                        body_plan_first_duration,
                    )
                    if body_plan_hold is not None
                    else None
                )
                low_level_result = low_level.compute(
                    go2,
                    active_gait,
                    force_hold,
                    time_now,
                    desired_body_state=desired_body_state,
                )
                torque_hold = low_level_result.torque
            else:
                raw_torque = np.zeros(12, dtype=float)
                for leg_index, leg in enumerate(("FL", "FR", "RL", "RR")):
                    leg_slice = slice(3 * leg_index, 3 * leg_index + 3)
                    leg_output = operational_space_low_level.compute_leg_torque(
                        leg,
                        go2,
                        active_gait,
                        force_hold[leg_slice],
                        time_now,
                    )
                    raw_torque[leg_slice] = leg_output.tau
                torque_hold = np.clip(
                    raw_torque,
                    -np.tile(np.array([23.7, 23.7, 45.0]), 4),
                    np.tile(np.array([23.7, 23.7, 45.0]), 4),
                )

        simulation.set_joint_torque(torque_hold)
        if config.terrain == "alternating_random_friction":
            # Contact points become available after step1.  Override each
            # foot/ground contact with the hidden coefficient at that foot's
            # world position before MuJoCo solves constraints in step2.
            mj.mj_step1(simulation.model, simulation.data)
            simulation.apply_spatial_contact_friction()
            mj.mj_step2(simulation.model, simulation.data)
        else:
            mj.mj_step(simulation.model, simulation.data)
        contact_accumulator.add(contact_reader.read(simulation.data))

    if trainer is not None:
        for completed_training in trainer.finish():
            if (
                completed_training.error is not None
                or completed_training.state_dict is None
            ):
                raise RuntimeError(
                    "Pending Go2 paper-style T2S training failed:\n"
                    + str(completed_training.error)
                )
            training_records.append(
                _t2s_training_record(
                    completed_training,
                    ready_step=np.nan,
                    activated=False,
                )
            )

    output = (
        Path(config.output_dir).expanduser().resolve()
        / (
            f"{config.controller}_{config.duration:g}s_"
            + (
                "alternating_random_friction_"
                if config.terrain == "alternating_random_friction"
                else ""
            )
            + (
                f"payload{config.payload_mass_kg:g}kg_"
                if config.payload_mass_kg > 0.0
                else ""
            )
            + f"seed{config.seed}"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(
        config,
        records,
        status,
        plant_mass=plant_mass,
        disabled_obstacles=disabled_obstacles,
    )
    metrics["t2s_protocol"]["residual_output_scale"] = (
        residual_output_scale.tolist()
    )
    metrics["t2s_training_results"] = len(training_records)
    metrics["t2s_fast_training_results"] = int(
        sum(bool(row["contains_fast"]) for row in training_records)
    )
    metrics["t2s_slow_training_results"] = int(
        sum(bool(row["contains_slow"]) for row in training_records)
    )
    metrics["mean_t2s_training_ms"] = (
        float(np.mean([row["train_ms"] for row in training_records]))
        if training_records
        else None
    )
    _write_csv(output / "trajectory.csv", records)
    _write_csv(output / "t2s_training.csv", training_records)
    with (output / "metrics.json").open("w") as stream:
        json.dump(metrics, stream, indent=2)
    return metrics, output


def save_comparison_plot(outputs: dict[str, Path], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_output = next(iter(outputs.values()))
    first_data = np.genfromtxt(
        first_output / "trajectory.csv", delimiter=",", names=True
    )
    has_friction = "friction_body_sliding" in first_data.dtype.names
    row_count = 5 if has_friction else 4
    figure, axes = plt.subplots(
        row_count, 1, figsize=(9, 10 if has_friction else 9), sharex=True
    )
    for method, output in outputs.items():
        data = np.genfromtxt(output / "trajectory.csv", delimiter=",", names=True)
        label = {
            "nominal": "Nominal MPC",
            "ssi": "SSI-MPC",
            "t2s": "T2S-MPC",
        }.get(method, method)
        axes[0].plot(data["time"], data["vx"], label=label)
        axes[1].plot(data["time"], data["z"], label=label)
        axes[2].plot(data["time"], data["x_error"], label=label)
        axes[3].plot(data["time"], data["y_error"], label=label)
        if has_friction:
            axes[4].plot(
                data["time"], data["friction_body_sliding"], label=label
            )
    axes[0].axhline(0.5, color="black", linestyle="--", label="reference")
    axes[1].axhline(TARGET_HEIGHT, color="black", linestyle="--")
    axes[2].axhline(0.0, color="black", linestyle="--")
    axes[3].axhline(0.0, color="black", linestyle="--")
    axes[0].set_ylabel("$v_x$ [m/s]")
    axes[1].set_ylabel("height [m]")
    axes[2].set_ylabel("pace $x$ error [m]")
    axes[3].set_ylabel("$y$ error [m]")
    if has_friction:
        axes[4].set_ylabel("body $\\mu_s$")
    axes[-1].set_xlabel("time [s]")
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
