"""Measured-latency scheduling on simulation time, independent of plant cost.

This is software-in-the-loop latency replay, not wall-paced/hard-real-time
execution. Never release a solution or a trained snapshot before its measured
completion time on the logical clock. A slow plant must neither shorten a
controller's budget nor give the asynchronous learner free logical time.
"""
import math
import time


def completion_cycle(source_cycle, elapsed_seconds, period):
    if source_cycle < 0 or not all(math.isfinite(v) for v in (elapsed_seconds, period)):
        raise ValueError("Invalid latency or source cycle")
    if elapsed_seconds < 0 or period <= 0:
        raise ValueError("Latency must be nonnegative and period positive")
    return source_cycle + max(1, math.ceil(elapsed_seconds / period))


def stamp_latency_response(response, observation_wall, period):
    elapsed = response["received_wall"] - observation_wall
    response.update(
        observation_wall=observation_wall,
        deadline_wall=observation_wall + period,
        received_e2e_ms=1000 * elapsed,
        compute_e2e_ms=1000 * (response["ready_wall"] - observation_wall),
        deadline_missed=elapsed > period,
        available_cycle=completion_cycle(response["cycle"], elapsed, period),
        ready_sim_time=response["cycle"] * period + elapsed,
        deadline_sim_time=(response["cycle"] + 1) * period,
    )
    return response


class LatencyReplayTrainer:
    """Gate a real asynchronous trainer's snapshots on measured logical time.

    materialize() may wait for a pending computation outside the timed MPC
    section. Looking at its completed timestamp is only scheduling metadata:
    poll_at() withholds the weights and submit() stays busy until logical time
    reaches that timestamp. This preserves one atomic training job in flight.
    The underlying trainer, samples, epochs and optimizers are unchanged.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.pending = None
        self.result = None
        self.last_release = None

    def submit(self, job, logical_submit):
        if self.pending is not None:
            return False
        if not self.trainer.submit(job):
            return False
        self.pending = (logical_submit, job.trigger_wall_time)
        return True

    def materialize(self, timeout=10.):
        start = time.perf_counter()
        if self.pending is not None and self.result is None:
            while self.result is None:
                self.result = self.trainer.poll()
                if self.result is not None:
                    break
                if time.perf_counter() - start > timeout:
                    raise TimeoutError("Training result materialization timed out")
                time.sleep(.0005)
        return time.perf_counter() - start

    def poll_at(self, logical_time):
        if self.result is None:
            return None
        logical_submit, wall_submit = self.pending
        latency = self.result.ready_wall_time - wall_submit
        if not math.isfinite(latency) or latency < 0:
            raise ValueError("Invalid measured training latency")
        ready_sim = logical_submit + latency
        if ready_sim > logical_time:
            return None
        self.last_release = dict(submitted_sim_time=logical_submit,
                                 ready_sim_time=ready_sim,
                                 training_latency_ms=1000 * latency)
        result = self.result
        self.pending = self.result = None
        return result

    def finish(self, timeout=10.):
        results = [self.result] if self.result is not None else []
        results.extend(self.trainer.finish(timeout=timeout))
        self.pending = self.result = None
        return results
