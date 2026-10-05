"""Integer-tick scheduling; do not drive hybrid events from accumulated floats."""
import math


def tick_time(tick: int, frequency: int) -> float:
    if not isinstance(tick, int) or tick < 0:
        raise ValueError("tick must be a nonnegative integer")
    if not isinstance(frequency, int) or frequency <= 0:
        raise ValueError("frequency must be a positive integer")
    return tick / frequency


def resume_wall_boundary(planned: float, observed: float, period: float):
    """Continue after a host stall without aborting or running catch-up cycles.

    Returns (effective current boundary, wall shift). If a whole wall period
    was lost, future releases are re-anchored at the actual observation time.
    Simulation time and all previously submitted jobs' deadlines are unchanged.
    Sub-period jitter retains the existing wall schedule and deadline budget.
    """
    if not all(math.isfinite(v) for v in (planned, observed, period)) or period <= 0:
        raise ValueError("Wall times must be finite and the period positive")
    lateness = max(0., observed-planned)
    shift = lateness if lateness >= period else 0.
    return planned+shift, shift
