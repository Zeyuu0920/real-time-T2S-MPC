"""MuJoCo adapter for the paper's Quad-SDK inverse-dynamics leg controller.

The paper does not release its MuJoCo bridge.  This module ports the public
Quad-SDK controller semantics to the Pinocchio Go2 model used by our MuJoCo
plant: planned GRFs for stance, planned swing-foot position/velocity/
acceleration, full-body inverse dynamics for swing feedforward, and the
published Go2 joint gains and torque limits at 500 Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from convex_mpc.go2_robot_data import PinGo2Model


LEGS = ("FL", "FR", "RL", "RR")
LEG_INDEX = {leg: index for index, leg in enumerate(LEGS)}
LEG_SLICES = {
    leg: slice(3 * index, 3 * index + 3)
    for index, leg in enumerate(LEGS)
}
JOINT_KP = np.tile(np.array([10.0, 10.0, 10.0]), 4)
JOINT_KD = np.tile(np.array([1.0, 1.0, 1.0]), 4)
TORQUE_LIMIT = np.tile(np.array([23.7, 23.7, 45.0]), 4)


@dataclass(frozen=True)
class QuadSdkLowLevelResult:
    torque: np.ndarray
    feedforward_torque: np.ndarray
    desired_joint_position: np.ndarray
    desired_joint_velocity: np.ndarray
    desired_foot_position: np.ndarray
    desired_foot_velocity: np.ndarray
    desired_foot_acceleration: np.ndarray
    contact_mask: np.ndarray


class PaperQuadSdkLowLevel:
    """Source-derived Quad-SDK low-level controller for the MuJoCo Go2."""

    def __init__(self, *, update_period: float = 0.002) -> None:
        if update_period <= 0.0:
            raise ValueError("update_period must be positive")
        self.update_period = float(update_period)
        self._last_contact: np.ndarray | None = None
        self._swing_start_time = np.zeros(4)
        self._swing_trajectory: list[object | None] = [None] * 4
        self._swing_liftoff_position: list[np.ndarray | None] = [None] * 4
        self._stance_position: list[np.ndarray | None] = [None] * 4
        self._last_desired_joint_position: np.ndarray | None = None

    @staticmethod
    def _all_foot_jacobians(go2: PinGo2Model) -> np.ndarray:
        return np.vstack(
            [go2.compute_full_foot_Jacobian_world(leg) for leg in LEGS]
        )

    @staticmethod
    def _all_jdot_v(go2: PinGo2Model) -> np.ndarray:
        return np.concatenate(
            [go2.compute_Jdot_dq_world(leg) for leg in LEGS]
        )

    @staticmethod
    def _sdls_inverse(jacobian: np.ndarray) -> np.ndarray:
        """Selective-damping pseudoinverse used by Quad-SDK ``math_utils``."""

        left, singular_values, right_transpose = np.linalg.svd(
            np.asarray(jacobian, dtype=float), full_matrices=False
        )
        maximum = float(np.max(singular_values))
        if maximum <= np.finfo(float).eps:
            return np.zeros((jacobian.shape[1], jacobian.shape[0]))
        inverse = np.minimum(
            1.0 / np.maximum(singular_values, np.finfo(float).eps),
            1.0 / (0.1 * maximum),
        )
        return right_transpose.T @ np.diag(inverse) @ left.T

    @staticmethod
    def _leg_joint_indices(go2: PinGo2Model, leg: str) -> tuple[list[int], list[int]]:
        joint_names = (
            f"{leg}_hip_joint",
            f"{leg}_thigh_joint",
            f"{leg}_calf_joint",
        )
        q_indices = []
        v_indices = []
        for name in joint_names:
            joint_id = go2.model.getJointId(name)
            q_indices.append(int(go2.model.joints[joint_id].idx_q))
            v_indices.append(int(go2.model.joints[joint_id].idx_v))
        return q_indices, v_indices

    def _inverse_kinematics(
        self,
        go2: PinGo2Model,
        desired_foot_positions: np.ndarray,
        desired_body_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Independent damped IK, equivalent to Quad-SDK's ikRobotState call."""

        q_des = np.asarray(go2.current_config.get_q(), dtype=float).copy()
        if desired_body_state is not None:
            desired_body_state = np.asarray(
                desired_body_state, dtype=float
            ).reshape(12)
            q_des[0:3] = desired_body_state[0:3]
            q_des[3:7] = np.asarray(
                pin.Quaternion(
                    pin.rpy.rpyToMatrix(desired_body_state[3:6])
                ).coeffs(),
                dtype=float,
            )
        desired = np.asarray(desired_foot_positions, dtype=float).reshape(4, 3)
        data = go2.model.createData()
        for leg_index, leg in enumerate(LEGS):
            q_indices, v_indices = self._leg_joint_indices(go2, leg)
            foot_id = getattr(go2, f"{leg}_foot_id")
            for _ in range(12):
                pin.forwardKinematics(go2.model, data, q_des)
                pin.updateFramePlacements(go2.model, data)
                pin.computeJointJacobians(go2.model, data, q_des)
                foot_position = np.asarray(data.oMf[foot_id].translation)
                error = desired[leg_index] - foot_position
                if np.linalg.norm(error) < 1.0e-5:
                    break
                jacobian = pin.getFrameJacobian(
                    go2.model,
                    data,
                    foot_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )[:3, v_indices]
                damping = 1.0e-5
                increment = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + damping * np.eye(3), error
                )
                increment = np.clip(increment, -0.12, 0.12)
                q_des[q_indices] += increment
        return q_des[7:19]

    def _desired_joint_velocity(
        self,
        go2: PinGo2Model,
        desired_joint_position: np.ndarray,
        desired_foot_velocity: np.ndarray,
        desired_body_state: np.ndarray,
    ) -> np.ndarray:
        """Solve the source's velocity IK from the planned full-body state."""

        desired_body_state = np.asarray(
            desired_body_state, dtype=float
        ).reshape(12)
        q_des = np.asarray(go2.current_config.get_q(), dtype=float).copy()
        q_des[0:3] = desired_body_state[0:3]
        rotation = pin.rpy.rpyToMatrix(desired_body_state[3:6])
        q_des[3:7] = np.asarray(
            pin.Quaternion(rotation).coeffs(), dtype=float
        )
        q_des[7:19] = np.asarray(
            desired_joint_position, dtype=float
        ).reshape(12)
        # Pinocchio's free-flyer tangent stores the translational velocity in
        # the local body frame and the angular velocity in the body frame.
        base_velocity = np.concatenate(
            (
                rotation.T @ desired_body_state[6:9],
                desired_body_state[9:12],
            )
        )
        desired_foot_velocity = np.asarray(
            desired_foot_velocity, dtype=float
        ).reshape(4, 3)
        data = go2.model.createData()
        pin.forwardKinematics(go2.model, data, q_des)
        pin.computeJointJacobians(go2.model, data, q_des)
        pin.updateFramePlacements(go2.model, data)
        joint_velocity = np.zeros(12)
        for leg_index, leg in enumerate(LEGS):
            _, velocity_indices = self._leg_joint_indices(go2, leg)
            foot_id = getattr(go2, f"{leg}_foot_id")
            jacobian = pin.getFrameJacobian(
                go2.model,
                data,
                foot_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:3]
            rhs = (
                desired_foot_velocity[leg_index]
                - jacobian[:, :6] @ base_velocity
            )
            joint_block = jacobian[:, velocity_indices]
            joint_velocity[3 * leg_index : 3 * leg_index + 3] = (
                self._sdls_inverse(joint_block) @ rhs
            )
        return joint_velocity

    def _desired_foot_plan(
        self,
        go2: PinGo2Model,
        gait,
        current_time: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        contact = np.asarray(
            gait.compute_current_mask(current_time), dtype=bool
        ).reshape(4)
        positions = np.zeros((4, 3))
        velocities = np.zeros((4, 3))
        accelerations = np.zeros((4, 3))
        measured_positions = []
        for leg in LEGS:
            position, _ = go2.get_single_foot_state_in_world(leg)
            measured_positions.append(position)

        if self._last_contact is None:
            self._last_contact = contact.copy()
            for index in range(4):
                self._stance_position[index] = measured_positions[index].copy()
                if not contact[index]:
                    # A first STEP plan can start with a diagonal pair already
                    # in swing.  In the public foot planner their remembered
                    # footholds are the measured feet at plan index zero.
                    self._swing_liftoff_position[index] = measured_positions[
                        index
                    ].copy()
                    self._swing_start_time[index] = current_time

        for index, leg in enumerate(LEGS):
            took_off = self._last_contact[index] and not contact[index]
            touched_down = not self._last_contact[index] and contact[index]
            if took_off:
                trajectory, _ = gait.compute_swing_traj_and_touchdown(go2, leg)
                self._swing_trajectory[index] = trajectory
                self._swing_liftoff_position[index] = measured_positions[
                    index
                ].copy()
                self._swing_start_time[index] = current_time
            if touched_down:
                # The public continuous foot plan holds the planned touchdown
                # position after the schedule changes to stance.  Freezing the
                # measured toe position here is incorrect when MuJoCo contact
                # is a few milliseconds late: an airborne toe would become
                # the stance target and never be driven the remaining distance
                # to the ground.
                trajectory = self._swing_trajectory[index]
                if trajectory is None:
                    self._stance_position[index] = measured_positions[
                        index
                    ].copy()
                else:
                    touchdown, _, _ = trajectory(gait.swing_time)
                    self._stance_position[index] = np.asarray(
                        touchdown, dtype=float
                    ).reshape(3)

            if contact[index]:
                if self._stance_position[index] is None:
                    self._stance_position[index] = measured_positions[index].copy()
                positions[index] = self._stance_position[index]
            else:
                trajectory = self._swing_trajectory[index]
                if trajectory is None:
                    trajectory, _ = gait.compute_swing_traj_and_touchdown(go2, leg)
                    self._swing_trajectory[index] = trajectory
                    self._swing_liftoff_position[index] = measured_positions[
                        index
                    ].copy()
                    self._swing_start_time[index] = current_time
                # The public local planner recomputes its entire continuous
                # foot plan after every 200 Hz MPC solve.  Its swing curve
                # retains the original liftoff state but uses the newest
                # planned touchdown.  Freezing our trajectory at takeoff was
                # therefore not source-equivalent and allowed an increasingly
                # stale endpoint to miss contact.  Recreate that online
                # endpoint update when using the ported author gait.
                planned_touchdowns = getattr(
                    gait, "_planned_touchdowns", {}
                )
                make_author_trajectory = getattr(
                    gait, "_make_author_swing_trajectory", None
                )
                liftoff_position = self._swing_liftoff_position[index]
                if (
                    leg in planned_touchdowns
                    and make_author_trajectory is not None
                    and liftoff_position is not None
                ):
                    trajectory = make_author_trajectory(
                        liftoff_position,
                        planned_touchdowns[leg],
                        gait.swing_time,
                        apex_height=0.07,
                    )
                    self._swing_trajectory[index] = trajectory
                position, velocity, acceleration = trajectory(
                    current_time - self._swing_start_time[index]
                )
                positions[index] = position
                velocities[index] = velocity
                accelerations[index] = acceleration
        self._last_contact = contact.copy()
        return (
            contact,
            positions.reshape(12),
            velocities.reshape(12),
            accelerations.reshape(12),
        )

    def _inverse_dynamics_feedforward(
        self,
        go2: PinGo2Model,
        desired_foot_acceleration: np.ndarray,
        commanded_grf: np.ndarray,
        contact: np.ndarray,
    ) -> np.ndarray:
        """Port Quad-SDK's constrained floating-base inverse dynamics."""

        mass_matrix_upper = np.asarray(go2.data.M, dtype=float)
        mass_matrix = np.triu(mass_matrix_upper) + np.triu(
            mass_matrix_upper, 1
        ).T
        nonlinear = np.asarray(
            pin.nonLinearEffects(
                go2.model,
                go2.data,
                go2.current_config.get_q(),
                go2.current_config.get_dq(),
            ),
            dtype=float,
        ).reshape(18)
        jacobian = self._all_foot_jacobians(go2)
        jdot_v = self._all_jdot_v(go2)
        desired_acceleration = np.asarray(
            desired_foot_acceleration, dtype=float
        ).reshape(12)
        commanded_grf = np.asarray(commanded_grf, dtype=float).reshape(12)

        tau_stance = -jacobian.T @ commanded_grf
        foot_acceleration_qdd = desired_acceleration - jdot_v
        joint_jacobian = jacobian[:, 6:]
        joint_jacobian_inverse = self._sdls_inverse(joint_jacobian)

        contact_legs = np.flatnonzero(contact)
        stance_rows = (
            np.concatenate(
                [np.arange(3 * leg, 3 * leg + 3) for leg in contact_legs]
            )
            if len(contact_legs)
            else np.empty(0, dtype=int)
        )
        constraint_jacobian = jacobian[stance_rows]
        constraint_jdot_v = jdot_v[stance_rows]
        constraint_count = len(stance_rows)

        block_matrix = np.zeros(
            (18 + constraint_count, 18 + constraint_count), dtype=float
        )
        jb = jacobian[:, :6]
        block_matrix[0:6, 0:6] = (
            -mass_matrix[0:6, 0:6]
            + mass_matrix[0:6, 6:18] @ joint_jacobian_inverse @ jb
        )
        block_matrix[6:18, 0:6] = (
            -mass_matrix[6:18, 0:6]
            + mass_matrix[6:18, 6:18] @ joint_jacobian_inverse @ jb
        )
        for leg, is_contact in enumerate(contact):
            if not is_contact:
                leg_slice = slice(6 + 3 * leg, 6 + 3 * leg + 3)
                block_matrix[leg_slice, leg_slice] = np.eye(3)
        if constraint_count:
            block_matrix[:18, 18:] = -constraint_jacobian.T
            constraint_base = constraint_jacobian[:, :6]
            constraint_joint = constraint_jacobian[:, 6:18]
            block_matrix[18:, :6] = (
                -constraint_base
                + constraint_joint @ joint_jacobian_inverse @ jb
            )

        block_rhs = np.zeros(18 + constraint_count, dtype=float)
        block_rhs[0:6] = (
            nonlinear[0:6]
            + mass_matrix[0:6, 6:18]
            @ joint_jacobian_inverse
            @ foot_acceleration_qdd
        )
        block_rhs[6:18] = (
            nonlinear[6:18]
            + mass_matrix[6:18, 6:18]
            @ joint_jacobian_inverse
            @ foot_acceleration_qdd
            - tau_stance[6:18]
        )
        if constraint_count:
            block_rhs[18:] = (
                constraint_jdot_v
                # Match QuadKD::computeInverseDynamics literally.  The
                # released implementation uses A.leftCols(12) here rather
                # than the kinematically more usual A.rightCols(12).
                + constraint_jacobian[:, :12]
                @ joint_jacobian_inverse
                @ foot_acceleration_qdd
            )

        solution = np.linalg.lstsq(block_matrix, block_rhs, rcond=None)[0]
        swing_torque = solution[6:18]
        feedforward = swing_torque.copy()
        for leg, is_contact in enumerate(contact):
            if is_contact:
                leg_slice = slice(3 * leg, 3 * leg + 3)
                feedforward[leg_slice] = tau_stance[6:18][leg_slice]
        if not np.all(np.isfinite(feedforward)):
            feedforward[:] = 0.0
        return feedforward

    def compute(
        self,
        go2: PinGo2Model,
        gait,
        commanded_grf: np.ndarray,
        current_time: float,
        desired_body_state: np.ndarray | None = None,
    ) -> QuadSdkLowLevelResult:
        contact, foot_position, foot_velocity, foot_acceleration = (
            self._desired_foot_plan(go2, gait, current_time)
        )
        desired_joint_position = self._inverse_kinematics(
            go2, foot_position, desired_body_state
        )
        if desired_body_state is not None:
            desired_joint_velocity = self._desired_joint_velocity(
                go2,
                desired_joint_position,
                foot_velocity,
                desired_body_state,
            )
        elif self._last_desired_joint_position is None:
            desired_joint_velocity = np.zeros(12)
        else:
            desired_joint_velocity = (
                desired_joint_position - self._last_desired_joint_position
            ) / self.update_period
        desired_joint_velocity = np.clip(desired_joint_velocity, -25.0, 25.0)
        self._last_desired_joint_position = desired_joint_position.copy()

        feedforward = self._inverse_dynamics_feedforward(
            go2, foot_acceleration, commanded_grf, contact
        )
        measured_joint_position = go2.current_config.get_q()[7:19]
        measured_joint_velocity = go2.current_config.get_dq()[6:18]
        torque = (
            feedforward
            + JOINT_KP * (desired_joint_position - measured_joint_position)
            + JOINT_KD * (desired_joint_velocity - measured_joint_velocity)
        )
        torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)
        return QuadSdkLowLevelResult(
            torque=torque,
            feedforward_torque=feedforward,
            desired_joint_position=desired_joint_position,
            desired_joint_velocity=desired_joint_velocity,
            desired_foot_position=foot_position,
            desired_foot_velocity=foot_velocity,
            desired_foot_acceleration=foot_acceleration,
            contact_mask=contact,
        )
