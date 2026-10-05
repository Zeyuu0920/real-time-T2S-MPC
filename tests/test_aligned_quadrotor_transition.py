import numpy as np

from src.aligned_quadrotor_transition import CausalAlignedTransitionBuffer
from src.realtime_dynamics_3d import RESIDUAL_DERIVATIVE_INDICES
from src.state_measurement import StateEstimatePacket


def _packet(source_step, arrival_step, state, dt=0.02):
    return StateEstimatePacket(
        source_step=source_step,
        source_time=source_step * dt,
        arrival_step=arrival_step,
        arrival_time=arrival_step * dt,
        state=np.asarray(state, dtype=float),
    )


def test_delayed_transition_uses_all_actual_command_substeps():
    def derivative(state, control):
        result = np.zeros(12)
        result[1] = control[0]
        return result

    transition_buffer = CausalAlignedTransitionBuffer(
        nominal_derivative=derivative,
        state_dimension=12,
        derivative_indices=RESIDUAL_DERIVATIVE_INDICES,
        control_dt=0.02,
        physics_dt=0.002,
    )
    state0 = np.zeros(12)
    transition_buffer.reset(_packet(0, 0, state0))
    schedule = np.vstack(
        (np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
         np.tile([3.0, 0.0, 0.0, 0.0], (7, 1)))
    )
    transition_buffer.record_command_interval(0, schedule)

    state1 = state0.copy()
    state1[1] = 0.002 * (3.0 * 1.0 + 7.0 * 3.0)
    transition_buffer.add_measurement(_packet(1, 2, state1))
    ready = transition_buffer.pop_ready()

    assert len(ready) == 1
    transition = ready[0]
    np.testing.assert_allclose(transition.target, np.zeros(6), atol=1e-12)
    np.testing.assert_allclose(
        transition.equivalent_control, [2.4, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(transition.command_schedule, schedule)
    assert transition.label_delay_steps == 1
    assert np.isclose(transition.label_delay_seconds, 0.02)
    assert transition_buffer.pop_ready() == []


def test_controller_motor_state_uses_source_time_not_current_plant_time():
    def derivative(state, control):
        result = np.zeros(16)
        result[12:16] = (np.sqrt(control) - state[12:16]) / 0.025
        return result

    transition_buffer = CausalAlignedTransitionBuffer(
        nominal_derivative=derivative,
        state_dimension=16,
        derivative_indices=RESIDUAL_DERIVATIVE_INDICES,
        control_dt=0.02,
        physics_dt=0.002,
        motor_time_constant=0.025,
    )
    physical0 = np.zeros(12)
    initial_motor = np.ones(4)
    transition_buffer.reset(
        _packet(0, 0, physical0), initial_motor_amplitude=initial_motor
    )
    schedule = np.full((10, 4), 4.0)
    transition_buffer.record_command_interval(0, schedule)

    # At arrival step 1, a one-step-delayed state still has source step zero.
    delayed_source_zero = _packet(0, 1, physical0)
    np.testing.assert_allclose(
        transition_buffer.state_for_packet(delayed_source_zero)[12:16],
        initial_motor,
    )

    source_one = _packet(1, 2, physical0)
    expected = 2.0 + (initial_motor - 2.0) * np.exp(-0.02 / 0.025)
    np.testing.assert_allclose(
        transition_buffer.state_for_packet(source_one)[12:16], expected
    )

