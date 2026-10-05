"""Author-style SSI RK4 prediction-error update with aligned command history.

Six outputs and intra-period command switching are task adaptations. This is
the author's outer-product update, NOT differentiation through the RK4 solver.
For switching commands, average the start-state RFF values over their actual
durations (not the commands before applying the nonlinear feature map).
"""
import casadi as cs
import numpy as np

from src.ssi_mpc import SSIOnlineLearner


class AlignedRK4SSIOnlineLearner(SSIOnlineLearner):
    def __init__(self, *args, physics_dt, control_dt, **kwargs):
        super().__init__(*args, **kwargs)
        self.physics_dt = float(physics_dt)
        self.control_dt = float(control_dt)
        self.substeps = int(round(self.control_dt / self.physics_dt))
        if self.substeps < 1 or not np.isclose(
            self.substeps * self.physics_dt, self.control_dt
        ):
            raise ValueError("control interval must contain whole physics substeps")
        nx = self._predict.size1_in(0)
        state = cs.MX.sym("aligned_ssi_state", nx)
        commands = cs.MX.sym("aligned_ssi_commands", 4, self.substeps)
        alpha = cs.MX.sym("aligned_ssi_alpha", self.alpha.size)
        predicted = state
        feature_sum = cs.MX.zeros(self.features.count, 1)
        for i in range(self.substeps):
            predicted = self._predict(predicted, commands[:, i], self.physics_dt, alpha)
            predicted = cs.vertcat(
                predicted[:6], cs.atan2(cs.sin(predicted[6:9]), cs.cos(predicted[6:9])),
                predicted[9:],
            )
            feature_sum += self.features.symbolic(state, commands[:, i])
        self._schedule_predict = cs.Function(
            "aligned_ssi_rk4_history", [state, commands, alpha],
            [predicted, feature_sum / self.substeps],
        )
        self.last_update = {}

    def update_transition(self, transition):
        schedule = np.asarray(transition.command_schedule, dtype=float)
        if schedule.shape != (self.substeps, 4) or not np.all(np.isfinite(schedule)):
            raise ValueError("invalid command history")
        if not np.isclose(transition.end_time - transition.source_time, self.control_dt):
            raise ValueError("transition timestamps do not span the control interval")
        predicted, phi = self._schedule_predict(
            transition.state, schedule.T, self.parameter_vector
        )
        predicted = np.asarray(predicted).reshape(-1)
        phi = np.asarray(phi).reshape(-1)
        observed = np.asarray(transition.next_state)[self.derivative_indices]
        error = (predicted[self.derivative_indices] - observed) / self.control_dt
        self.alpha -= 2.0 * self.learning_rate * np.outer(error, phi)
        self.last_update = {
            "source_step": transition.source_step,
            "source_time": transition.source_time,
            "end_time": transition.end_time,
            **{f"rk4_predicted_{i}": float(v) for i, v in enumerate(predicted[self.derivative_indices])},
            **{f"rk4_error_{i}": float(v) for i, v in enumerate(error)},
            **{f"update_feature_{i}": float(v) for i, v in enumerate(phi)},
        }
        return error
