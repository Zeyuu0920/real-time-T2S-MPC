"""Computation-aware control release inside a fixed sampling period."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlReleaseDecision:
    """Control applied now and disposition of the newly computed candidate."""

    applied_control: np.ndarray
    candidate_control: np.ndarray
    candidate_accepted: bool
    candidate_discarded: bool
    applied_source_step: int
    applied_age_steps: int
    command_schedule: np.ndarray
    held_substeps: int
    candidate_substeps: int


class ComputationAwareControlRelease:
    """Zero-order-hold publication with physics-rate latency emulation.

    The previously active command is applied while the controller computes.
    An on-time candidate is released for the remaining physics substeps of
    the same control interval.  A late or invalid candidate is discarded and
    the previous command is held for the complete interval.
    """

    VALID_MODES = ("intra_period", "period_boundary")

    def __init__(self, initial_control, mode="intra_period"):
        initial = np.asarray(initial_control, dtype=float)
        if initial.ndim != 1 or not np.all(np.isfinite(initial)):
            raise ValueError("initial control must be a finite vector")
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {self.VALID_MODES}, got {mode!r}"
            )
        self._active_control = initial.copy()
        self._active_source_step = -1
        self._mode = mode

    def release(
        self,
        candidate_control,
        *,
        compute_ms,
        deadline_ms,
        substep_ms,
        substep_count,
        candidate_step,
        solver_success=True,
    ):
        candidate = np.asarray(candidate_control, dtype=float)
        if candidate.shape != self._active_control.shape:
            raise ValueError("candidate and active controls must have equal shape")
        if not np.all(np.isfinite(candidate)):
            solver_success = False
        if not np.isfinite(compute_ms) or compute_ms < 0.0:
            raise ValueError("compute_ms must be finite and nonnegative")
        if not np.isfinite(deadline_ms) or deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be finite and positive")
        if not np.isfinite(substep_ms) or substep_ms <= 0.0:
            raise ValueError("substep_ms must be finite and positive")
        if int(substep_count) != substep_count or substep_count <= 0:
            raise ValueError("substep_count must be a positive integer")

        previous = self._active_control.copy()
        source_step = self._active_source_step
        accepted = bool(solver_success and compute_ms <= deadline_ms)
        held_substeps = int(
            np.clip(np.ceil(compute_ms / substep_ms), 0, substep_count)
        )
        if accepted and self._mode == "intra_period":
            candidate_substeps = int(substep_count) - held_substeps
            command_schedule = np.repeat(
                previous[None, :], held_substeps, axis=0
            )
            if candidate_substeps:
                command_schedule = np.concatenate(
                    (
                        command_schedule,
                        np.repeat(
                            candidate[None, :], candidate_substeps, axis=0
                        ),
                    ),
                    axis=0,
                )
            self._active_control = candidate.copy()
            self._active_source_step = int(candidate_step)
        elif accepted:
            # Synchronous sampled-data ablation: a control computed during
            # interval k is published only at the k+1 sampling boundary.
            # Every physics substep in interval k therefore uses the command
            # that was active when the interval began.
            held_substeps = int(substep_count)
            candidate_substeps = 0
            command_schedule = np.repeat(
                previous[None, :], int(substep_count), axis=0
            )
            self._active_control = candidate.copy()
            self._active_source_step = int(candidate_step)
        else:
            held_substeps = int(substep_count)
            candidate_substeps = 0
            command_schedule = np.repeat(
                previous[None, :], int(substep_count), axis=0
            )

        applied = command_schedule.mean(axis=0)

        return ControlReleaseDecision(
            applied_control=applied,
            candidate_control=candidate.copy(),
            candidate_accepted=accepted,
            candidate_discarded=not accepted,
            applied_source_step=source_step,
            applied_age_steps=int(candidate_step) - source_step,
            command_schedule=command_schedule,
            held_substeps=held_substeps,
            candidate_substeps=candidate_substeps,
        )
