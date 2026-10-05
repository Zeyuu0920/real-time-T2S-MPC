"""Versioned measurement and joint-actuator interfaces, no privileged learners."""
from src.go2_sim2real_config import RealismOptions
import numpy as np
from scipy.spatial.transform import Rotation


def same_contact_mask(a, b):
    """Upstream gait returns (4,1); WBC uses (4,), same four contacts."""
    return np.array_equal(np.asarray(a,dtype=bool).reshape(4),
                          np.asarray(b,dtype=bool).reshape(4))


class BoundarySensor:
    """One coherent noisy free-flyer configuration/velocity at each MPC tick.

    q is Pinocchio [xyz, quaternion xyzw, joints]; dq is body linear/body
    angular/joint velocity. Joint encoders are unchanged, from the SAME tick.
    FK and the centroid state are subsequently both derived from this packet.
    No noise is drawn at physics ticks or depending on a solver's outcome.
    """
    def __init__(self, options, seed, dt):
        self.options = options
        self.rng = np.random.default_rng(seed + 810000)
        self.rho = np.exp(-dt/options.noise_correlation_s)
        self.std = np.repeat([options.position_std, options.attitude_std,
                              options.velocity_std, options.body_rate_std], 3)
        self.noise = None

    def observe(self, q, dq):
        q, dq = q.copy(), dq.copy()
        draw = self.rng.standard_normal(12)
        self.noise = (self.std*draw if self.noise is None else
                      self.rho*self.noise + np.sqrt(1-self.rho**2)*self.std*draw)
        old_rotation = Rotation.from_quat(q[3:7])
        world_velocity = old_rotation.apply(dq[:3])
        # Small rotation-vector perturbation in the world frame, not an
        # unnormalized quaternion perturbation.
        rotation = Rotation.from_rotvec(self.noise[3:6]) * old_rotation
        q[:3] += self.noise[:3]
        q[3:7] = rotation.as_quat()
        dq[:3] = rotation.inv().apply(world_velocity + self.noise[6:9])
        dq[3:6] += self.noise[9:12]
        return q, dq


class JointResponse:
    """Synthetic first-order torque response after the unchanged WBC.

    This is NOT a calibrated Go2 motor transfer function. Gain is sampled
    once per joint/trial. No extra random torque noise or command delay.
    """
    def __init__(self, options, seed, dt, initial_torque):
        self.gain = np.random.default_rng(seed+820000).uniform(
            1-options.actuator_gain_range, 1+options.actuator_gain_range, 12)
        self.alpha = 1. if options.actuator_tau_s == 0 else -np.expm1(-dt/options.actuator_tau_s)
        self.torque = np.asarray(initial_torque).copy()
        self.limit = np.array([23.7,23.7,45.]*4)

    def step(self, requested):
        target = np.clip(self.gain*np.asarray(requested), -self.limit, self.limit)
        self.torque += self.alpha*(target-self.torque)
        self.torque = np.clip(self.torque, -self.limit, self.limit)
        return self.torque.copy()


def aligned_transition(packets, schedules, eligible, source_index):
    """FE wrench label input from a complete, delayed boundary transition.

    Since M,h,J are frozen at the start in the existing FE label, averaging
    its known dispatched GRFs is exactly equivalent to averaging J @ u.
    This is an interval-average training approximation for the nonlinear
    residual, NOT a claim that the learned residual commutes with averaging.
    """
    if source_index < 1 or not eligible[source_index-1]:
        return None
    before, after = packets[source_index-1], packets[source_index]
    schedule = schedules[source_index-1]
    if schedule.shape != (20,12):
        raise ValueError("Expected twenty 1-ms command-history entries")
    return (before['state'].copy(), after['state'].copy(), schedule.mean(axis=0),
            before['feet'].copy(), before['time'])
