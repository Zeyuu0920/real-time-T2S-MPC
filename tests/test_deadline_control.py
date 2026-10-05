import numpy as np

from src.deadline_control import ComputationAwareControlRelease


def test_on_time_candidate_is_released_after_computation():
    release = ComputationAwareControlRelease(np.array([1.0, 1.0]))

    first = release.release(
        np.array([2.0, 3.0]),
        compute_ms=8.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=0,
    )
    second = release.release(
        np.array([4.0, 5.0]),
        compute_ms=9.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=1,
    )

    np.testing.assert_allclose(
        first.command_schedule[:4], np.tile([1.0, 1.0], (4, 1))
    )
    np.testing.assert_allclose(
        first.command_schedule[4:], np.tile([2.0, 3.0], (6, 1))
    )
    np.testing.assert_allclose(first.applied_control, [1.6, 2.2])
    assert first.candidate_accepted
    np.testing.assert_allclose(
        second.command_schedule[:5], np.tile([2.0, 3.0], (5, 1))
    )
    np.testing.assert_allclose(
        second.command_schedule[5:], np.tile([4.0, 5.0], (5, 1))
    )
    assert second.applied_source_step == 0
    assert second.applied_age_steps == 1


def test_late_candidate_is_discarded_and_previous_control_is_held():
    release = ComputationAwareControlRelease(np.array([1.0, 1.0]))
    release.release(
        np.array([2.0, 3.0]),
        compute_ms=20.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=0,
    )

    late = release.release(
        np.array([8.0, 9.0]),
        compute_ms=20.001,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=1,
    )
    after_late = release.release(
        np.array([4.0, 5.0]),
        compute_ms=7.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=2,
    )

    assert late.candidate_discarded
    np.testing.assert_allclose(
        late.command_schedule, np.tile([2.0, 3.0], (10, 1))
    )
    np.testing.assert_allclose(
        after_late.command_schedule[:4], np.tile([2.0, 3.0], (4, 1))
    )
    np.testing.assert_allclose(
        after_late.command_schedule[4:], np.tile([4.0, 5.0], (6, 1))
    )
    assert after_late.applied_age_steps == 2


def test_solver_failure_also_holds_previous_control():
    release = ComputationAwareControlRelease(np.array([1.0]))
    failed = release.release(
        np.array([2.0]),
        compute_ms=4.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=0,
        solver_success=False,
    )
    assert failed.candidate_discarded
    np.testing.assert_allclose(failed.applied_control, [1.0])
    assert failed.held_substeps == 10


def test_period_boundary_mode_defers_on_time_candidate_to_next_period():
    release = ComputationAwareControlRelease(
        np.array([1.0, 1.0]), mode="period_boundary"
    )

    first = release.release(
        np.array([2.0, 3.0]),
        compute_ms=4.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=0,
    )
    second = release.release(
        np.array([4.0, 5.0]),
        compute_ms=6.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=1,
    )

    assert first.candidate_accepted
    assert first.held_substeps == 10
    assert first.candidate_substeps == 0
    np.testing.assert_allclose(
        first.command_schedule, np.tile([1.0, 1.0], (10, 1))
    )
    np.testing.assert_allclose(
        second.command_schedule, np.tile([2.0, 3.0], (10, 1))
    )
    assert second.applied_source_step == 0
    assert second.applied_age_steps == 1


def test_period_boundary_mode_does_not_publish_late_candidate():
    release = ComputationAwareControlRelease(
        np.array([1.0]), mode="period_boundary"
    )
    release.release(
        np.array([2.0]),
        compute_ms=4.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=0,
    )
    late = release.release(
        np.array([9.0]),
        compute_ms=21.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=1,
    )
    after_late = release.release(
        np.array([3.0]),
        compute_ms=4.0,
        deadline_ms=20.0,
        substep_ms=2.0,
        substep_count=10,
        candidate_step=2,
    )

    assert late.candidate_discarded
    np.testing.assert_allclose(late.command_schedule, np.tile([2.0], (10, 1)))
    np.testing.assert_allclose(
        after_late.command_schedule, np.tile([2.0], (10, 1))
    )
    assert after_late.applied_source_step == 0
    assert after_late.applied_age_steps == 2


def test_unknown_release_mode_is_rejected():
    try:
        ComputationAwareControlRelease(np.array([1.0]), mode="unknown")
    except ValueError as error:
        assert "mode must be one of" in str(error)
    else:
        raise AssertionError("unknown release mode was accepted")
