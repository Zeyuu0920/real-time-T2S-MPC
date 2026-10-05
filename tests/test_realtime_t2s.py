import time

import casadi as cs
import numpy as np
import torch
from l4casadi.realtime import RealTimeL4CasADi

from src.models import TwoSpeedMLP, time_embedding_np
from src.realtime_dynamics import Quadrotor2DRealTimeT2SDynamics
from src.realtime_neural import first_order_parameters, relu_activation_patterns
from src.realtime_t2s import (
    AsyncRealtimeT2STrainer,
    RealtimeT2SJob,
    sample_replay_without_replacement,
)


def _state(model):
    return {
        name: value.detach().numpy().copy()
        for name, value in model.state_dict().items()
    }


def test_replay_sampling_returns_distinct_aligned_entries():
    rng = np.random.default_rng(12)
    inputs = [np.array([index], dtype=np.float32) for index in range(100)]
    targets = [np.array([-index], dtype=np.float32) for index in range(100)]
    sampled_inputs, sampled_targets = sample_replay_without_replacement(
        rng, inputs, targets, batch_size=64
    )

    assert sampled_inputs.shape == (64, 1)
    assert len(np.unique(sampled_inputs[:, 0])) == 64
    np.testing.assert_array_equal(sampled_targets[:, 0], -sampled_inputs[:, 0])


def test_replay_sampling_rejects_batch_larger_than_buffer():
    rng = np.random.default_rng(2)
    with np.testing.assert_raises_regex(ValueError, "distinct samples"):
        sample_replay_without_replacement(
            rng,
            [np.array([0.0])],
            [np.array([0.0])],
            batch_size=2,
        )


def test_two_speed_relu_patterns_cover_both_hidden_layers():
    model = TwoSpeedMLP(input_dim=1, hidden_dim=1, output_dim=1)
    with torch.no_grad():
        model.layer1.weight[:] = 1.0
        model.layer1.bias[:] = 0.0
        model.layer2.weight[:] = 1.0
        model.layer2.bias[:] = -0.5
    patterns = relu_activation_patterns(
        model, np.array([[-1.0], [0.25], [1.0]], dtype=np.float32)
    )
    assert patterns.tolist() == [
        [False, False],
        [True, False],
        [True, True],
    ]


def test_fast_worker_updates_only_output_layer():
    torch.manual_seed(4)
    model = TwoSpeedMLP(input_dim=5, hidden_dim=4, output_dim=3)
    initial = _state(model)
    trainer = AsyncRealtimeT2STrainer(
        model,
        input_dim=5,
        hidden_dim=4,
        fast_epochs=2,
        slow_epochs=2,
        torch_threads=1,
    )
    job = RealtimeT2SJob(
        version=1,
        trigger_step=3,
        trigger_wall_time=time.perf_counter(),
        fast_inputs=np.ones((4, 5), dtype=np.float32),
        fast_targets=np.zeros((4, 3), dtype=np.float32),
    )
    assert trainer.submit(job)
    result = trainer.finish()[0]
    assert result.error is None
    for name in initial:
        changed = not np.array_equal(initial[name], result.state_dict[name])
        assert changed == name.startswith("layer3")


def test_slow_worker_keeps_output_layer_fixed():
    torch.manual_seed(5)
    model = TwoSpeedMLP(input_dim=5, hidden_dim=4, output_dim=3)
    initial = _state(model)
    trainer = AsyncRealtimeT2STrainer(
        model,
        input_dim=5,
        hidden_dim=4,
        fast_epochs=2,
        slow_epochs=2,
        torch_threads=1,
    )
    job = RealtimeT2SJob(
        version=1,
        trigger_step=10,
        trigger_wall_time=time.perf_counter(),
        slow_inputs=np.ones((32, 5), dtype=np.float32),
        slow_targets=np.zeros((32, 3), dtype=np.float32),
    )
    assert trainer.submit(job)
    result = trainer.finish()[0]
    assert result.error is None
    assert np.array_equal(initial["layer3.weight"], result.state_dict["layer3.weight"])
    assert np.array_equal(initial["layer3.bias"], result.state_dict["layer3.bias"])
    assert any(
        not np.array_equal(initial[name], result.state_dict[name])
        for name in initial
        if name.startswith("layer1") or name.startswith("layer2")
    )


def test_realtime_t2s_dynamics_prepends_embedding_to_taylor_parameters():
    class Env:
        MASS = 0.027
        J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
        GRAVITY_ACC = 9.8
        L = 0.0397
        KF = 3.16e-10
        PWM2RPM_SCALE = 0.2685
        PWM2RPM_CONST = 4070.3
        MIN_PWM = 20000.0
        MAX_PWM = 65535.0

    torch.manual_seed(6)
    model = TwoSpeedMLP(input_dim=40, hidden_dim=4, output_dim=3)
    realtime = RealTimeL4CasADi(
        model, approximation_order=1, name="test_rt_t2s_dynamics"
    )
    dynamics = Quadrotor2DRealTimeT2SDynamics(Env(), realtime).model()
    assert int(dynamics.p.shape[0]) == 32 + 40 + 3 + 3 * 40

    state = np.array([0.1, 0.2, 0.9, -0.1, 0.05, 0.0])
    control = np.array([0.13, 0.14])
    tau = 0.25
    point = np.concatenate([state, control, time_embedding_np(tau)])
    packed, exact, _ = first_order_parameters(model, point[None, :])
    function = cs.Function(
        "test_rt_t2s_residual_function",
        [dynamics.x, dynamics.u, dynamics.p],
        [dynamics.f_residual],
    )
    residual = function(
        state, control, np.concatenate((time_embedding_np(tau), packed[0]))
    ).full().ravel()[[1, 3, 5]]
    np.testing.assert_allclose(residual, exact[0], atol=1e-6)


def test_analytic_t2s_jacobian_matches_autograd():
    from src.models import BoundedTwoSpeedMLP

    torch.manual_seed(17)
    model = BoundedTwoSpeedMLP(input_dim=7, hidden_dim=9, output_dim=4)
    points = torch.randn(5, 7)
    _, analytic_values, analytic_jacobians = first_order_parameters(
        model, points.numpy()
    )
    with torch.no_grad():
        expected_values = model(points).numpy()
    expected_jacobians = (
        torch.func.vmap(torch.func.jacrev(model))(points).detach().numpy()
    )
    np.testing.assert_allclose(analytic_values, expected_values, atol=1e-7)
    np.testing.assert_allclose(
        analytic_jacobians, expected_jacobians, atol=1e-6
    )
