from types import SimpleNamespace

import casadi as cs
import numpy as np
import pytest

from src.ssi_mpc import SSIOnlineLearner, SSIRandomFeatures
from src.ssi_rk4_aligned import AlignedRK4SSIOnlineLearner


def learners(substeps=10):
    features = SSIRandomFeatures.sample(
        42, count=50, kernel_std=.7, thrust_feature_mode="normalized_command",
        command_thrust_scale=np.full(4, .15),
    )
    x, u, a = cs.MX.sym("x", 16), cs.MX.sym("u", 4), cs.MX.sym("a", 300)
    residual = cs.reshape(a, 6, 50) @ features.symbolic(x, u)
    dx = cs.MX.zeros(16, 1)
    for j, i in enumerate([1, 3, 5, 9, 10, 11]):
        dx[i] = -.2 * x[i] ** 2 + cs.sum1(x[12:16] ** 2) + residual[j]
    for p, v in [(0, 1), (2, 3), (4, 5)]:
        dx[p] = x[v]
    dx[12:16] = (cs.sqrt(u) - x[12:16]) / .025
    dynamics = cs.Function("test_ssi_dynamics", [x, u, a], [dx])
    old = SSIOnlineLearner(features, dynamics, residual_dimension=6)
    new = AlignedRK4SSIOnlineLearner(
        features, dynamics, residual_dimension=6, physics_dt=.02 / substeps, control_dt=.02,
    )
    initial_alpha = np.random.default_rng(17).normal(0, .1, (6, 50))
    old.alpha[:] = new.alpha[:] = initial_alpha
    return old, new


def transition(schedule):
    state = np.linspace(.01, .16, 16)
    return SimpleNamespace(
        state=state, next_state=state + .001, command_schedule=schedule,
        source_step=11, source_time=.22, end_time=.24,
        target=np.full(6, 1e9),  # MUST NOT be consumed by the RK4 updater.
    )


def test_one_segment_matches_author_existing_update():
    old, new = learners(1)
    t = transition(np.full((1, 4), .1))
    expected = old.update(t.state, t.command_schedule[0], t.next_state, .02)
    actual = new.update_transition(t)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(new.alpha, old.alpha, atol=1e-12)


def test_switched_command_rollout_and_outer_product():
    old, new = learners()
    schedule = np.vstack([np.full((3, 4), .03), np.full((7, 4), .14)])
    t = transition(schedule)
    initial = old.alpha.copy()
    predicted = t.state.copy()
    for command in schedule:
        predicted = np.asarray(old._predict(predicted, command, .002, old.parameter_vector)).ravel()
        predicted[6:9] = np.arctan2(np.sin(predicted[6:9]), np.cos(predicted[6:9]))
    phi = np.mean([old.features.numpy(t.state, command) for command in schedule], axis=0)
    error = (predicted[old.derivative_indices] - t.next_state[old.derivative_indices]) / .02
    actual = new.update_transition(t)
    np.testing.assert_allclose(actual, error, atol=1e-11)
    np.testing.assert_allclose(new.alpha, initial - .5 * np.outer(error, phi), atol=1e-11)
    assert np.linalg.norm(phi - old.features.numpy(t.state, schedule.mean(axis=0))) > 1e-4


def test_command_order_changes_prediction():
    _, a = learners()
    _, b = learners()
    schedule = np.vstack([np.full((5, 4), .03), np.full((5, 4), .14)])
    assert np.linalg.norm(a.update_transition(transition(schedule)) -
                          b.update_transition(transition(schedule[::-1]))) > 1e-5


def test_constant_command_feature_is_author_start_feature():
    old, new = learners()
    t = transition(np.full((10, 4), .1))
    _, phi = new._schedule_predict(t.state, t.command_schedule.T, new.parameter_vector)
    np.testing.assert_allclose(np.asarray(phi).ravel(), old.features.numpy(t.state, t.command_schedule[0]))


def test_bad_time_pair_rejected():
    _, new = learners()
    t = transition(np.full((10, 4), .1))
    t.end_time = .26
    with pytest.raises(ValueError, match="timestamps"):
        new.update_transition(t)
