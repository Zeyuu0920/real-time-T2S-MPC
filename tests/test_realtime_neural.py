import numpy as np
import torch
import time
import casadi as cs
from l4casadi.realtime import RealTimeL4CasADi

from src.realtime_neural import (
    AsyncRealtimeTrainer,
    RealtimeTrainingJob,
    evaluate_first_order,
    first_order_parameters,
    relu_activation_patterns,
)


def test_first_order_parameter_order_matches_casadi_column_major():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    point = np.array([[5.0, 6.0]], dtype=np.float32)
    parameters, values, jacobians = first_order_parameters(model, point)

    np.testing.assert_allclose(values, [[17.0, 39.0]])
    np.testing.assert_allclose(jacobians, [[[1.0, 2.0], [3.0, 4.0]]])
    # CasADi's vec_F([[1, 2], [3, 4]]) is [1, 3, 2, 4].
    np.testing.assert_allclose(parameters[0, 4:], [1.0, 3.0, 2.0, 4.0])


def test_packed_parameters_drive_realtime_l4casadi_expression_correctly():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    realtime = RealTimeL4CasADi(model, approximation_order=1, name="test_pack")
    symbolic_input = cs.MX.sym("input", 2, 1)
    symbolic_output = realtime(symbolic_input)
    packed_symbol = realtime.get_sym_params()
    packed_input = cs.MX.sym("packed", int(packed_symbol.shape[0]))
    packed_output = cs.substitute(symbolic_output, packed_symbol, packed_input)
    function = cs.Function("packed_realtime_model", [symbolic_input, packed_input], [packed_output])

    expansion = np.array([[5.0, 6.0]], dtype=np.float32)
    parameters, _, _ = first_order_parameters(model, expansion)
    result = function(np.array([6.0, 7.0]), parameters[0]).full().ravel()
    np.testing.assert_allclose(result, [20.0, 46.0], atol=1e-8)


def test_first_order_is_exact_for_affine_model():
    model = torch.nn.Linear(2, 2)
    expansion = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float32)
    query = np.array([[3.0, 4.0], [-2.0, 5.0]], dtype=np.float32)
    _, values, jacobians = first_order_parameters(model, expansion)
    approximation = evaluate_first_order(
        expansion, values, jacobians, query
    )
    with torch.no_grad():
        exact = model(torch.tensor(query)).numpy()
    np.testing.assert_allclose(approximation, exact, atol=1e-6)


def test_relu_pattern_reports_boundary_crossing():
    class SmallMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(1, 1, bias=False),
                torch.nn.ReLU(),
                torch.nn.Linear(1, 1, bias=False),
            )
            with torch.no_grad():
                self.net[0].weight[:] = 1.0
                self.net[2].weight[:] = 1.0

        def forward(self, value):
            return self.net(value)

    model = SmallMLP()
    patterns = relu_activation_patterns(
        model, np.array([[-1.0], [1.0]], dtype=np.float32)
    )
    assert patterns.tolist() == [[False], [True]]


def test_training_runs_in_persistent_process_and_returns_weights():
    torch.manual_seed(8)
    # The process reconstructs the repository MLP, so use the matching model.
    from src.models import MLP

    model = MLP(input_dim=8, output_dim=3, hidden_dim=4, num_layers=1)
    initial = {
        name: value.detach().numpy().copy()
        for name, value in model.state_dict().items()
    }
    trainer = AsyncRealtimeTrainer(
        model,
        hidden_dim=4,
        num_layers=1,
        epochs=2,
        torch_threads=1,
    )
    job = RealtimeTrainingJob(
        version=1,
        trigger_step=50,
        trigger_wall_time=time.perf_counter(),
        inputs=np.ones((32, 8), dtype=np.float32),
        targets=np.zeros((32, 3), dtype=np.float32),
    )
    assert trainer.submit(job)
    assert not trainer.submit(job)
    pending = trainer.finish()
    assert len(pending) == 1
    result = pending[0]
    assert result.error is None
    assert result.state_dict is not None
    assert any(
        not np.array_equal(initial[name], result.state_dict[name])
        for name in initial
    )
