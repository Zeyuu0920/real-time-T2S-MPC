"""Author-code-compatible Go2 Nominal and random-feature SSI NMPC.

This module is intentionally separate from ``go2_realtime_acados``.  It ports
the public implementation in UM-iRaL/Adaptive-Legged-Locomotion rather than
changing the shared T2S comparison stack.  In particular it follows the
repository's 20-node, 0.03 s collocation grid, IPOPT solve, Go2 rigid-body
parameters, force bounds, friction pyramid, and random-feature update.

The public repository and the paper differ in two implementation details.  The
repository uses backward Euler and ``1/sqrt(M)`` random-feature normalization;
those source-code choices are used here because this is the external-baseline
reproduction path.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import casadi as cs
import numpy as np


STATE_DIM = 12
CONTROL_DIM = 12
FOOT_COUNT = 4
PAPER_NODES = 20
PAPER_INTERVALS = PAPER_NODES - 1
PAPER_NODE_DT = 0.03
PAPER_HORIZON_SECONDS = PAPER_NODES * PAPER_NODE_DT
PAPER_MPC_HZ = 200
PAPER_LOW_LEVEL_HZ = 500
PAPER_FRICTION_COEFFICIENT = 0.3
PAPER_MIN_STANCE_FZ = 10.0
PAPER_MAX_STANCE_FZ = 150.0
PAPER_MODEL_MASS = 16.086
# The author's quad_nlp.h uses 13.3 kg only for the nominal GRF cost target,
# while its generated Go2 dynamics use 16.086 kg.  Preserve that distinction.
PAPER_FORCE_REFERENCE_MASS = 13.3
GRAVITY = 9.81

PAPER_STATE_COST_DIAGONAL = np.array(
    [12.5, 12.5, 12.5, 0.5, 0.5, 2.5, 0.2, 0.2, 0.4, 0.1, 0.1, 0.4],
    dtype=float,
)
PAPER_INPUT_COST_DIAGONAL = np.full(CONTROL_DIM, 5e-5, dtype=float)

PAPER_RFF_INPUT_DIM = 15
PAPER_RFF_OUTPUT_DIM = 6
PAPER_RFF_COUNT = 50
PAPER_RFF_KERNEL_STD = 0.01
PAPER_RFF_LEARNING_RATE = 0.003
PAPER_WRENCH_NORMALIZATION = 100.0


def paper_first_element_duration(elapsed_seconds: float) -> float:
    """Return time from a 200 Hz solve to the next 30 ms grid node."""

    elapsed = max(0.0, float(elapsed_seconds))
    duration = PAPER_NODE_DT - elapsed % PAPER_NODE_DT
    if duration < 1e-9:
        return PAPER_NODE_DT
    return duration


def rotation_zyx_numpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float).reshape(3)
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


def world_state_to_paper_state(state: np.ndarray) -> np.ndarray:
    """Convert ``[p,rpy,v,omega_world]`` to the author's body-omega state."""

    result = np.asarray(state, dtype=float).reshape(STATE_DIM).copy()
    result[9:12] = rotation_zyx_numpy(result[3:6]).T @ result[9:12]
    return result


def _paper_terms_symbolic(
    state,
    negative_foot_positions_body,
    *,
    model_mass: float = PAPER_MODEL_MASS,
):
    """Port ``computeLegDynamicGo2`` from the authors' public repository."""

    symbolic_type = cs.SX if isinstance(state, cs.SX) else cs.MX
    roll, pitch, yaw = state[3], state[4], state[5]
    omega1, omega2, omega3 = state[9], state[10], state[11]
    t2, t3, t4 = cs.cos(roll), cs.cos(pitch), cs.cos(yaw)
    t5, t6, t7 = cs.sin(roll), cs.sin(pitch), cs.sin(yaw)
    w1_sq, w2_sq, w3_sq = omega1**2, omega2**2, omega3**2

    mass_matrix = symbolic_type.zeros(6, 6)
    model_mass = float(model_mass)
    if model_mass <= 0.0:
        raise ValueError("model_mass must be positive")
    mass_matrix[0:3, 0:3] = model_mass * symbolic_type.eye(3)
    mass_matrix[3, 3:6] = cs.horzcat(4.4294859e-2, 1.2166e-4, 1.4849e-3)
    mass_matrix[4, 3:6] = cs.horzcat(
        t2 * 1.2166e-4 - t5 * 1.4849e-3,
        t2 * 4.4084322384e-1 + t5 * 3.12e-5,
        t2 * (-3.12e-5) - t5 * 4.6958108284e-1,
    )
    mass_matrix[5, 3:6] = cs.horzcat(
        t6 * (-4.4294859e-2) + t2 * t3 * 1.4849e-3 + t3 * t5 * 1.2166e-4,
        t6 * (-1.2166e-4) - t2 * t3 * 3.12e-5 + t3 * t5 * 4.4084322384e-1,
        t6 * (-1.4849e-3) + t2 * t3 * 4.6958108284e-1 - t3 * t5 * 3.12e-5,
    )

    nonlinear = symbolic_type.zeros(6, 1)
    nonlinear[2] = model_mass * GRAVITY
    nonlinear[3] = (
        omega2 * (omega1 * 1.4849e-3 - omega2 * 3.12e-5 + omega3 * 4.6958108284e-1)
        - omega3 * (omega1 * 1.2166e-4 + omega2 * 4.4084322384e-1 - omega3 * 3.12e-5)
    )
    nonlinear[4] = (
        t2 * w1_sq * (-1.4849e-3)
        + t2 * w3_sq * 1.4849e-3
        - t5 * w1_sq * 1.2166e-4
        + t5 * w2_sq * 1.2166e-4
        + omega1 * omega2 * t2 * 3.12e-5
        + omega1 * omega3 * t2 * (-4.2528622384e-1)
        + omega2 * omega3 * t2 * 1.2166e-4
        - omega1 * omega2 * t5 * 3.9654836484e-1
        + omega1 * omega3 * t5 * 3.12e-5
        + omega2 * omega3 * t5 * 1.4849e-3
    )
    nonlinear[5] = (
        t6 * w2_sq * 3.12e-5
        - t6 * w3_sq * 3.12e-5
        - omega1 * omega2 * t6 * 1.4849e-3
        + omega1 * omega3 * t6 * 1.2166e-4
        - omega2 * omega3 * t6 * 2.8737859e-2
        + t2 * t3 * w1_sq * 1.2166e-4
        - t2 * t3 * w2_sq * 1.2166e-4
        - t3 * t5 * w1_sq * 1.4849e-3
        + t3 * t5 * w3_sq * 1.4849e-3
        + omega1 * omega2 * t2 * t3 * 3.9654836484e-1
        - omega1 * omega3 * t2 * t3 * 3.12e-5
        - omega2 * omega3 * t2 * t3 * 1.4849e-3
        + omega1 * omega2 * t3 * t5 * 3.12e-5
        + omega1 * omega3 * t3 * t5 * (-4.2528622384e-1)
        + omega2 * omega3 * t3 * t5 * 1.2166e-4
    )

    contact_jacobian = symbolic_type.zeros(6, CONTROL_DIM)
    for leg in range(FOOT_COUNT):
        base = 3 * leg
        foot_x = negative_foot_positions_body[base]
        foot_y = negative_foot_positions_body[base + 1]
        foot_z = negative_foot_positions_body[base + 2]
        contact_jacobian[0:3, base : base + 3] = symbolic_type.eye(3)
        contact_jacobian[3, base : base + 3] = cs.horzcat(
            -foot_y * t6 - foot_z * t3 * t7,
            foot_x * t6 + foot_z * t3 * t4,
            -t3 * (foot_y * t4 - foot_x * t7),
        )
        contact_jacobian[4, base : base + 3] = cs.horzcat(
            -foot_z * t4,
            -foot_z * t7,
            foot_x * t4 + foot_y * t7,
        )
        contact_jacobian[5, base : base + 3] = cs.horzcat(
            foot_y, -foot_x, 0.0
        )

    q_dot = cs.vertcat(
        state[6:9],
        (omega1 * t3 + omega3 * t2 * t6 + omega2 * t5 * t6) / (t3 + 1e-9),
        omega2 * t2 - omega3 * t5,
        (omega3 * t2 + omega2 * t5) / (t3 + 1e-9),
    )
    return mass_matrix, nonlinear, contact_jacobian, q_dot


_TERMS_STATE = cs.MX.sym("paper_terms_state", STATE_DIM)
_TERMS_FEET = cs.MX.sym("paper_terms_feet", CONTROL_DIM)
_TERMS = _paper_terms_symbolic(_TERMS_STATE, _TERMS_FEET)
PAPER_TERMS_FUNCTION = cs.Function(
    "paper_go2_terms", [_TERMS_STATE, _TERMS_FEET], list(_TERMS)
)


def paper_terms_numpy(
    state: np.ndarray, negative_foot_positions_body: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    outputs = PAPER_TERMS_FUNCTION(
        np.asarray(state, dtype=float).reshape(STATE_DIM),
        np.asarray(negative_foot_positions_body, dtype=float).reshape(CONTROL_DIM),
    )
    return tuple(np.asarray(value.full(), dtype=float) for value in outputs)


def author_code_features_numpy(
    state: np.ndarray,
    force: np.ndarray,
    negative_foot_positions_body: np.ndarray,
) -> np.ndarray:
    """Return the exact source-code feature order ``[theta,v,omega,JTu/100]``."""

    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    force = np.asarray(force, dtype=float).reshape(CONTROL_DIM)
    normalized_wrench = paper_contact_wrench_numpy(
        state, force, negative_foot_positions_body
    ) / PAPER_WRENCH_NORMALIZATION
    return np.concatenate((state[3:12], normalized_wrench.reshape(6)))


def paper_contact_wrench_numpy(
    state: np.ndarray,
    force: np.ndarray,
    negative_foot_positions_body: np.ndarray,
) -> np.ndarray:
    """Evaluate the published ``J(q).T @ u`` without CasADi call overhead."""

    state = np.asarray(state, dtype=float).reshape(STATE_DIM)
    forces = np.asarray(force, dtype=float).reshape(FOOT_COUNT, 3)
    feet = np.asarray(negative_foot_positions_body, dtype=float).reshape(
        FOOT_COUNT, 3
    )
    pitch, yaw = state[4], state[5]
    cos_pitch, cos_yaw = np.cos(pitch), np.cos(yaw)
    sin_pitch, sin_yaw = np.sin(pitch), np.sin(yaw)
    result = np.zeros(6, dtype=float)
    result[:3] = np.sum(forces, axis=0)
    for (foot_x, foot_y, foot_z), (fx, fy, fz) in zip(feet, forces):
        result[3] += (
            (-foot_y * sin_pitch - foot_z * cos_pitch * sin_yaw) * fx
            + (foot_x * sin_pitch + foot_z * cos_pitch * cos_yaw) * fy
            - cos_pitch * (foot_y * cos_yaw - foot_x * sin_yaw) * fz
        )
        result[4] += (
            -foot_z * cos_yaw * fx
            - foot_z * sin_yaw * fy
            + (foot_x * cos_yaw + foot_y * sin_yaw) * fz
        )
        result[5] += foot_y * fx - foot_x * fy
    return result


def paper_residual_wrench_target_numpy(
    state_before: np.ndarray,
    state_after: np.ndarray,
    force: np.ndarray,
    negative_foot_positions_body: np.ndarray,
    interval_seconds: float,
) -> np.ndarray:
    """Reconstruct the paper's six-axis generalized-force residual label.

    Both SSI and T2S use this same physical target.  It is the generalized
    force/torque needed to make the nominal paper dynamics explain the
    measured velocity transition under the commanded four-foot GRFs.
    """

    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive")
    state_before = np.asarray(state_before, dtype=float).reshape(STATE_DIM)
    state_after = np.asarray(state_after, dtype=float).reshape(STATE_DIM)
    force = np.asarray(force, dtype=float).reshape(CONTROL_DIM)
    mass_matrix, nonlinear, contact_jacobian, _ = paper_terms_numpy(
        state_before, negative_foot_positions_body
    )
    return (
        mass_matrix
        @ ((state_after[6:12] - state_before[6:12]) / interval_seconds)
        + nonlinear.reshape(6)
        - contact_jacobian @ force
    )


@dataclass(frozen=True)
class AuthorCodeRandomFeatures:
    omega: np.ndarray
    phase: np.ndarray

    @classmethod
    def sample(cls, seed: int) -> "AuthorCodeRandomFeatures":
        rng = np.random.default_rng(seed)
        return cls(
            omega=rng.normal(
                0.0,
                PAPER_RFF_KERNEL_STD,
                size=(PAPER_RFF_COUNT, PAPER_RFF_INPUT_DIM),
            ),
            phase=rng.uniform(0.0, 2.0 * np.pi, size=PAPER_RFF_COUNT),
        )

    def evaluate_input(self, feature_input: np.ndarray) -> np.ndarray:
        feature_input = np.asarray(feature_input, dtype=float).reshape(
            PAPER_RFF_INPUT_DIM
        )
        return np.cos(self.omega @ feature_input + self.phase) / np.sqrt(
            PAPER_RFF_COUNT
        )

    def numpy(
        self,
        state: np.ndarray,
        force: np.ndarray,
        negative_foot_positions_body: np.ndarray,
    ) -> np.ndarray:
        return self.evaluate_input(
            author_code_features_numpy(
                state, force, negative_foot_positions_body
            )
        )


class AuthorCodeSSIOnlineLearner:
    def __init__(self, features: AuthorCodeRandomFeatures) -> None:
        self.features = features
        self.alpha = np.zeros(
            (PAPER_RFF_OUTPUT_DIM, PAPER_RFF_COUNT), dtype=float
        )

    def predict(
        self,
        state: np.ndarray,
        force: np.ndarray,
        negative_foot_positions_body: np.ndarray,
    ) -> np.ndarray:
        return self.alpha @ self.features.numpy(
            state, force, negative_foot_positions_body
        )

    def update(
        self,
        state_before: np.ndarray,
        state_after: np.ndarray,
        force: np.ndarray,
        negative_foot_positions_body: np.ndarray,
        interval_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the authors' unprojected per-cycle OGD update."""

        target = paper_residual_wrench_target_numpy(
            state_before,
            state_after,
            force,
            negative_foot_positions_body,
            interval_seconds,
        )
        phi = self.features.numpy(
            state_before, force, negative_foot_positions_body
        )
        prediction_before = self.alpha @ phi
        prediction_error = target - prediction_before
        self.alpha += 2.0 * PAPER_RFF_LEARNING_RATE * np.outer(
            prediction_error, phi
        )
        return target, prediction_error


def paper_force_reference(
    contact_mask: np.ndarray,
    *,
    mass: float = PAPER_FORCE_REFERENCE_MASS,
) -> np.ndarray:
    contact_mask = np.asarray(contact_mask, dtype=bool).reshape(FOOT_COUNT)
    mass = float(mass)
    if mass <= 0.0:
        raise ValueError("mass must be positive")
    result = np.zeros(CONTROL_DIM, dtype=float)
    count = int(np.sum(contact_mask))
    if count:
        for leg in np.flatnonzero(contact_mask):
            result[3 * leg + 2] = (
                mass * GRAVITY / count
            )
    return result


@dataclass(frozen=True)
class PaperNMPCResult:
    force: np.ndarray
    states: np.ndarray
    forces: np.ndarray
    solve_ms: float
    end_to_end_ms: float
    status: str
    iterations: int


class PaperGo2NMPC:
    """Persistent CasADi/IPOPT transcription of the public Go2 controller."""

    def __init__(
        self,
        *,
        random_features: AuthorCodeRandomFeatures | None = None,
    ) -> None:
        self.random_features = random_features
        self._build_solver()
        self._guess: np.ndarray | None = None
        self._last_solution: np.ndarray | None = None
        self._last_contact_table: np.ndarray | None = None

    @property
    def method(self) -> str:
        return "ssi" if self.random_features is not None else "nominal"

    def _build_solver(self) -> None:
        nodes = PAPER_NODES
        intervals = PAPER_INTERVALS
        states = cs.SX.sym("paper_states", STATE_DIM, nodes)
        forces = cs.SX.sym("paper_forces", CONTROL_DIM, intervals)
        initial_state = cs.SX.sym("paper_initial_state", STATE_DIM)
        references = cs.SX.sym("paper_references", STATE_DIM, nodes)
        negative_feet = cs.SX.sym(
            "paper_negative_feet_body", CONTROL_DIM, intervals
        )
        force_references = cs.SX.sym(
            "paper_force_references", CONTROL_DIM, intervals
        )
        first_element_duration = cs.SX.sym("paper_first_element_duration")
        parameters = [
            initial_state,
            cs.vec(references),
            cs.vec(negative_feet),
            cs.vec(force_references),
            first_element_duration,
        ]
        alpha = None
        if self.random_features is not None:
            alpha_parameter = cs.SX.sym(
                "paper_ssi_alpha", PAPER_RFF_OUTPUT_DIM * PAPER_RFF_COUNT
            )
            alpha = cs.reshape(
                alpha_parameter, PAPER_RFF_OUTPUT_DIM, PAPER_RFF_COUNT
            )
            parameters.append(alpha_parameter)

        objective = 0
        constraints = [states[:, 0] - initial_state]
        lower_constraints = [np.zeros(STATE_DIM)]
        upper_constraints = [np.zeros(STATE_DIM)]
        q_weight = cs.DM(PAPER_STATE_COST_DIAGONAL)
        r_weight = cs.DM(PAPER_INPUT_COST_DIAGONAL)

        for stage in range(intervals):
            stage_dt = (
                first_element_duration if stage == 0 else PAPER_NODE_DT
            )
            state_before = states[:, stage]
            state_after = states[:, stage + 1]
            force = forces[:, stage]
            feet = negative_feet[:, stage]
            mass_matrix, nonlinear, contact_jacobian, q_dot = (
                _paper_terms_symbolic(state_after, feet)
            )
            residual = cs.SX.zeros(PAPER_RFF_OUTPUT_DIM, 1)
            if alpha is not None:
                # The public RF dynamics evaluate z at x_k, while M,h,J and
                # q_dot use x_{k+1} in the backward-Euler constraint.
                _, _, feature_jacobian, _ = _paper_terms_symbolic(
                    state_before, feet
                )
                feature_input = cs.vertcat(
                    state_before[3:12],
                    feature_jacobian
                    @ (force / PAPER_WRENCH_NORMALIZATION),
                )
                phi = cs.cos(
                    cs.DM(self.random_features.omega) @ feature_input
                    + cs.DM(self.random_features.phase)
                ) / np.sqrt(PAPER_RFF_COUNT)
                residual = alpha @ phi

            position_error = (
                state_after[0:6] - state_before[0:6]
                - stage_dt * q_dot
            )
            velocity_error = (
                mass_matrix @ (state_after[6:12] - state_before[6:12])
                + stage_dt
                * (nonlinear - contact_jacobian @ force - residual)
            )
            constraints.append(cs.vertcat(position_error, velocity_error))
            lower_constraints.append(np.zeros(STATE_DIM))
            upper_constraints.append(np.zeros(STATE_DIM))

            for leg in range(FOOT_COUNT):
                offset = 3 * leg
                fx, fy, fz = (
                    force[offset],
                    force[offset + 1],
                    force[offset + 2],
                )
                constraints.append(
                    cs.vertcat(
                        fx - PAPER_FRICTION_COEFFICIENT * fz,
                        -fx - PAPER_FRICTION_COEFFICIENT * fz,
                        fy - PAPER_FRICTION_COEFFICIENT * fz,
                        -fy - PAPER_FRICTION_COEFFICIENT * fz,
                    )
                )
                lower_constraints.append(np.full(4, -np.inf))
                upper_constraints.append(np.zeros(4))

            state_error = state_after - references[:, stage + 1]
            force_error = force - force_references[:, stage]
            cost_scale = (
                first_element_duration / PAPER_NODE_DT
                if stage == 0
                else 1.0
            )
            objective += cost_scale * 0.5 * cs.dot(
                q_weight * state_error, state_error
            ) + cost_scale * 0.5 * cs.dot(
                r_weight * force_error, force_error
            )

        decision = cs.vertcat(cs.vec(states), cs.vec(forces))
        parameter = cs.vertcat(*parameters)
        constraint = cs.vertcat(*constraints)
        problem = {"x": decision, "p": parameter, "f": objective, "g": constraint}
        options = {
            "error_on_fail": False,
            "print_time": False,
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-3,
            "ipopt.dual_inf_tol": 1e10,
            "ipopt.constr_viol_tol": 1e-2,
            "ipopt.compl_inf_tol": 1e-2,
            "ipopt.warm_start_bound_push": 1e-6,
            "ipopt.warm_start_slack_bound_push": 1e-6,
            "ipopt.warm_start_mult_bound_push": 1e-6,
            # The public controller does not override IPOPT's 3000-iteration
            # default.  Keeping 100 here falsely rejected contact switches.
            "ipopt.max_iter": 3000,
            "ipopt.sb": "yes",
        }
        self.solver = cs.nlpsol(
            f"paper_go2_{self.method}_ipopt", "ipopt", problem, options
        )
        self._state_size = STATE_DIM * nodes
        self._decision_size = int(decision.shape[0])
        self._lbg = np.concatenate(lower_constraints)
        self._ubg = np.concatenate(upper_constraints)

    def _decision_bounds(self, contact_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        contact_table = np.asarray(contact_table, dtype=bool).reshape(
            FOOT_COUNT, PAPER_INTERVALS
        )
        lower = np.full(self._decision_size, -np.inf, dtype=float)
        upper = np.full(self._decision_size, np.inf, dtype=float)
        # State order is column-major in cs.vec.  Match the repository's body
        # bounds, including nonnegative height.
        for node in range(PAPER_NODES):
            state_offset = STATE_DIM * node
            lower[state_offset + 2] = 0.0
            lower[state_offset + 3 : state_offset + 5] = -np.pi
            upper[state_offset + 3 : state_offset + 5] = np.pi
            lower[state_offset + 5] = -10.0
            upper[state_offset + 5] = 10.0
        for stage in range(PAPER_INTERVALS):
            force_offset = self._state_size + CONTROL_DIM * stage
            for leg in range(FOOT_COUNT):
                leg_slice = slice(
                    force_offset + 3 * leg, force_offset + 3 * leg + 3
                )
                if contact_table[leg, stage]:
                    lower[force_offset + 3 * leg + 2] = PAPER_MIN_STANCE_FZ
                    upper[force_offset + 3 * leg + 2] = PAPER_MAX_STANCE_FZ
                else:
                    lower[leg_slice] = 0.0
                    upper[leg_slice] = 0.0
        return lower, upper

    def _cold_guess(
        self, references: np.ndarray, force_references: np.ndarray
    ) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(references, dtype=float).reshape(-1, order="F"),
                np.asarray(force_references, dtype=float).reshape(-1, order="F"),
            )
        )

    def _shift_guess(
        self,
        initial_state: np.ndarray,
        references: np.ndarray,
        force_references: np.ndarray,
    ) -> np.ndarray:
        if self._last_solution is None:
            return self._cold_guess(references, force_references)
        states = self._last_solution[: self._state_size].reshape(
            STATE_DIM, PAPER_NODES, order="F"
        )
        forces = self._last_solution[self._state_size :].reshape(
            CONTROL_DIM, PAPER_INTERVALS, order="F"
        )
        shifted_states = np.column_stack((states[:, 1:], states[:, -1]))
        shifted_states[:, 0] = initial_state
        shifted_forces = np.column_stack((forces[:, 1:], forces[:, -1]))
        return np.concatenate(
            (
                shifted_states.reshape(-1, order="F"),
                shifted_forces.reshape(-1, order="F"),
            )
        )

    def solve(
        self,
        initial_state: np.ndarray,
        references: np.ndarray,
        negative_foot_positions_body: np.ndarray,
        contact_table: np.ndarray,
        *,
        first_element_duration: float = PAPER_NODE_DT,
        alpha: np.ndarray | None = None,
    ) -> PaperNMPCResult:
        cycle_start = time.perf_counter()
        initial_state = np.asarray(initial_state, dtype=float).reshape(STATE_DIM)
        references = np.asarray(references, dtype=float).reshape(
            STATE_DIM, PAPER_NODES
        )
        negative_feet = np.asarray(
            negative_foot_positions_body, dtype=float
        ).reshape(CONTROL_DIM, PAPER_INTERVALS)
        contact_table = np.asarray(contact_table, dtype=bool).reshape(
            FOOT_COUNT, PAPER_INTERVALS
        )
        first_element_duration = float(first_element_duration)
        if not 0.0 < first_element_duration <= PAPER_NODE_DT + 1e-9:
            raise ValueError(
                "first_element_duration must lie in (0, PAPER_NODE_DT]"
            )
        force_references = np.column_stack(
            [paper_force_reference(contact_table[:, k]) for k in range(PAPER_INTERVALS)]
        )
        parameter_parts = [
            initial_state,
            references.reshape(-1, order="F"),
            negative_feet.reshape(-1, order="F"),
            force_references.reshape(-1, order="F"),
            np.array([first_element_duration]),
        ]
        if self.random_features is None:
            if alpha is not None:
                raise ValueError("nominal solver does not accept SSI alpha")
        else:
            if alpha is None:
                raise ValueError("SSI solver requires alpha")
            parameter_parts.append(
                np.asarray(alpha, dtype=float)
                .reshape(PAPER_RFF_OUTPUT_DIM, PAPER_RFF_COUNT)
                .reshape(-1, order="F")
            )
        lower, upper = self._decision_bounds(contact_table)
        contact_changed = False
        if self._last_contact_table is not None:
            same_grid = np.array_equal(
                contact_table, self._last_contact_table
            )
            expected_one_node_shift = np.array_equal(
                contact_table[:, :-1], self._last_contact_table[:, 1:]
            )
            contact_changed = not (same_grid or expected_one_node_shift)
        guess = (
            self._cold_guess(references, force_references)
            if contact_changed
            else self._shift_guess(
                initial_state, references, force_references
            )
        )
        # The source disables its warm start and reinitializes GRFs whenever
        # the contact sequence changes unexpectedly.
        guess[:STATE_DIM] = initial_state
        solve_start = time.perf_counter()
        solution = self.solver(
            x0=guess,
            p=np.concatenate(parameter_parts),
            lbx=lower,
            ubx=upper,
            lbg=self._lbg,
            ubg=self._ubg,
        )
        solve_ms = 1000.0 * (time.perf_counter() - solve_start)
        stats = self.solver.stats()
        status = str(stats.get("return_status", "unknown"))
        decision = np.asarray(solution["x"].full(), dtype=float).reshape(-1)
        self._last_solution = decision
        self._last_contact_table = contact_table.copy()
        states = decision[: self._state_size].reshape(
            STATE_DIM, PAPER_NODES, order="F"
        )
        forces = decision[self._state_size :].reshape(
            CONTROL_DIM, PAPER_INTERVALS, order="F"
        )
        return PaperNMPCResult(
            force=forces[:, 0].copy(),
            states=states,
            forces=forces,
            solve_ms=solve_ms,
            end_to_end_ms=1000.0 * (time.perf_counter() - cycle_start),
            status=status,
            iterations=int(stats.get("iter_count", -1)),
        )
