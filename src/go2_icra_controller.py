"""Isolated controller process for boundary-sampled Go2 experiments."""
from collections import deque
from pathlib import Path
import os
import gc
import time
import traceback
import numpy as np
import torch
from l4casadi.realtime import RealTimeL4CasADi
from convex_mpc.go2_robot_data import PinGo2Model
from src.go2_icra_protocol import due_updates
from src.go2_paper_acados import PaperAcadosMPC, PAPER_ACADOS_INTERVALS
from src.go2_paper_acados_experiment import (
    _zero_t2s_output_layer, _paper_t2s_horizon_features, _horizon_negative_feet,
    _negative_world_foot_levers, _paper_world_state,
    _shift_body_plan_for_new_grid,
)
from src.go2_paper_baseline import (
    AuthorCodeRandomFeatures, AuthorCodeSSIOnlineLearner, paper_terms_numpy,
    author_code_features_numpy, paper_residual_wrench_target_numpy,
    world_state_to_paper_state, paper_first_element_duration, rotation_zyx_numpy,
    paper_force_reference,
)
from src.go2_paper_experiment import _AuthorDiscreteTrottingGait, author_twist_reference
from src.models import TwoSpeedMLP, time_embedding_np
from src.realtime_neural import first_order_parameters, exact_model_output, load_numpy_state_dict
from src.realtime_t2s import AsyncRealtimeT2STrainer, RealtimeT2SJob, sample_replay_without_replacement
from src.go2_icra_latency import LatencyReplayTrainer


def to_world(state):
    result = state.copy()
    result[9:12] = rotation_zyx_numpy(state[3:6]) @ state[9:12]
    return result


def horizon_time_codes(timestamp, first_dt, dimension=16):
    """Physical seconds, matching manuscript omega_i=pi/i and quad TIME_SCALE=1."""
    times = np.r_[timestamp, timestamp+first_dt+.03*np.arange(PAPER_ACADOS_INTERVALS-1)]
    return np.stack([time_embedding_np(t,d=dimension) for t in times])


def t2s_input_features(state, force, feet, embedding, mode="paper15"):
    """Same physical feature definition for labels, queries and shooting nodes."""
    if mode == "full_state_control24":
        physical = np.r_[np.asarray(state).reshape(12), np.asarray(force).reshape(12)]
    elif mode == "paper15":
        physical = author_code_features_numpy(state, force, feet)
    else:
        raise ValueError("Unknown T2S feature mode")
    embedding = np.asarray(embedding).reshape(-1)
    if len(embedding) == 0 or len(embedding) % 2:
        raise ValueError("Time embedding must have a positive even dimension")
    return np.r_[physical, embedding]


def t2s_horizon_features(solver, state, references, feet, contacts, embeddings, mode="paper15"):
    if mode == "paper15":
        return _paper_t2s_horizon_features(solver, state, references, feet, contacts, embeddings)
    if mode != "full_state_control24":
        raise ValueError("Unknown T2S feature mode")
    expansion = np.empty((PAPER_ACADOS_INTERVALS, 24 + embeddings.shape[1]))
    for k in range(PAPER_ACADOS_INTERVALS):
        x = solver.solver.get(k, "x") if solver.initialized else references[:, k]
        u = solver.solver.get(k, "u") if solver.initialized else paper_force_reference(contacts[:, k])
        if k == 0:
            x = state
        expansion[k] = t2s_input_features(x, u, feet[:, k], embeddings[k], mode)
    return expansion


def delayed_initial_state(state, force, feet, dt, residual):
    """First-order predictor for the *known held-command* actuation interval.

    Same nominal equation, residual snapshot and delay for every controller.
    There is no hidden contact force, liquid mass or friction measurement here.
    """
    mass, nonlinear, jac, qdot = paper_terms_numpy(state, feet)
    derivative = np.r_[qdot.ravel(), np.linalg.solve(
        mass, jac @ force + residual - nonlinear.ravel())]
    return state + dt * derivative


def predict_command_delay(state, force, feet, timestamp, delay, gait, warmup, predictor):
    """Predict the WBC-dispatched GRFs, including a scheduled liftoff.

    A held MPC vector does not push through a swing leg. In particular, at a
    diagonal exchange the old solution can temporarily give *zero* usable GRF.
    Treating that interval as full support creates fictitious support impulses.
    Only known gait timing is used; no future plant measurement is queried.
    """
    predicted = state.copy()
    elapsed = 0.
    previous_mask = None
    residual = np.zeros(6)
    while elapsed < delay - 1e-12:
        step = min(.001, delay-elapsed)
        mask = gait.compute_current_mask(warmup+timestamp+elapsed).astype(bool).reshape(4)
        applied = force.reshape(4,3) * mask[:,None]
        levers = (feet.reshape(4,3) + predicted[:3]-state[:3]).ravel()
        if previous_mask is None or not np.array_equal(mask, previous_mask):
            residual = predictor(predicted, applied.ravel(), levers, timestamp+elapsed)
            previous_mask = mask
        predicted = delayed_initial_state(predicted, applied.ravel(), levers, step, residual)
        elapsed += step
    return predicted


class Go2ICRAController:
    def __init__(self, config, training_core=None):
        self.config = config
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        self.go2 = PinGo2Model()
        self.gait = _AuthorDiscreteTrottingGait(config.warmup, config.speed)
        self.random_features = (AuthorCodeRandomFeatures.sample(config.seed)
                                if config.method == "ssi" else None)
        self.ssi = AuthorCodeSSIOnlineLearner(self.random_features) if self.random_features else None
        self.model = self.trainer = self.gp = None
        self.latency_trainer = None
        self.feature_mode = getattr(config, "t2s_feature_mode", "paper15")
        self.time_feat_dim = getattr(config, "time_feat_dim", 16)
        input_dim = (24 if self.feature_mode == "full_state_control24" else 15) + self.time_feat_dim
        realtime = None
        if config.method in ("t2s", "stgp"):
            self.model = TwoSpeedMLP(input_dim, hidden_dim=64, output_dim=6)
            _zero_t2s_output_layer(self.model)
            realtime = RealTimeL4CasADi(self.model, approximation_order=1,
                                      name=f"icra_{config.method}_feature{input_dim}")
        scale_option = dict(output_normalization=np.asarray(config.wrench_scales)) if realtime is not None else {}
        if self.feature_mode == "full_state_control24":
            scale_option["realtime_feature_mode"] = self.feature_mode
        suffix = "_xu24" if self.feature_mode == "full_state_control24" else ""
        if self.time_feat_dim != 16:
            if config.method != "t2s" or self.feature_mode != "full_state_control24":
                raise ValueError("Variable time dimension is restricted to full-state T2S")
            suffix += f"_te{self.time_feat_dim}"
            scale_option["time_embedding_dim"] = self.time_feat_dim
        self.solver = PaperAcadosMPC(
            project_root=Path(__file__).resolve().parents[1],
            random_features=self.random_features, realtime_residual=realtime,
            name_suffix=f"icra_v2_wrench_{config.method}_seed{config.seed}{suffix}",
            **scale_option,
        )
        self.scale = self.solver.dynamics.residual_output_scale
        if config.method == "t2s":
            self.trainer = AsyncRealtimeT2STrainer(
                self.model, input_dim=input_dim, hidden_dim=64, output_dim=6,
                fast_learning_rate=config.fast_lr, slow_learning_rate=config.slow_lr,
                fast_epochs=config.fast_epochs, slow_epochs=config.slow_epochs,
                torch_threads=1, bounded_output=False, cpu_core=training_core)
            if config.timing == "latency_replay_pipeline":
                self.latency_trainer = LatencyReplayTrainer(self.trainer)
        elif config.method == "stgp":
            from src.go2_stgp import Go2SpatioTemporalGPLearner
            self.gp = Go2SpatioTemporalGPLearner(
                control_dt=config.dt, inducing_point_count=config.gp_inducing,
                spatial_lengthscale=config.gp_spatial_lengthscale,
                temporal_lengthscale=config.gp_temporal_lengthscale,
                observation_noise_variance=config.gp_noise)
        self.inputs = deque(maxlen=config.replay_capacity)
        self.targets = deque(maxlen=config.replay_capacity)
        self.hybrid_replay = None
        if getattr(config, "t2s_replay_mode", "fifo") == "fifo_reservoir":
            if config.method != "t2s":
                raise ValueError("Hybrid replay is a T2S-only option")
            from src.go2_t2s_hybrid_replay import Go2T2SHybridReplay
            self.hybrid_replay = Go2T2SHybridReplay(config)
        self.rng = np.random.default_rng(config.seed + 700000)
        self.samples = self.version = 0
        self.next_version = 1
        self.training_log = []
        self.previous_plan = None
        self.previous_plan_index = None

    def predict(self, state, force, feet, timestamp):
        if self.config.method == "nominal":
            return np.zeros(6)
        physical = author_code_features_numpy(state, force, feet)
        if self.ssi is not None:
            return self.ssi.alpha @ self.random_features.evaluate_input(physical)
        if self.gp is not None:
            return self.gp.value_and_jacobian(physical[None], timestamp)[0][0] * self.scale
        if self.trainer is not None:
            features = t2s_input_features(state, force, feet,
                time_embedding_np(timestamp, d=self.time_feat_dim), self.feature_mode)
            return exact_model_output(self.model, features[None])[0] * self.scale
        return np.zeros(6)

    def solve(self, packet):
        start = time.perf_counter()
        cfg = self.config
        training_events = []
        if self.trainer is not None:
            completed = (self.latency_trainer.poll_at(packet["time"])
                         if self.latency_trainer else self.trainer.poll())
            if completed is not None:
                if completed.error:
                    raise RuntimeError(completed.error)
                load_numpy_state_dict(self.model, completed.state_dict)
                self.version = completed.version
                event = dict(version=completed.version, trigger_cycle=completed.trigger_step,
                             completed_wall=completed.ready_wall_time,
                             activated_cycle=packet["cycle"], train_ms=1000*completed.train_seconds)
                if self.latency_trainer:
                    event.update(self.latency_trainer.last_release)
                training_events.append(event)
                self.training_log.append(event)
        prediction_errors = []
        for sample in packet["samples"]:
            if cfg.method == "nominal":
                continue
            # These are completed, constant-command boundary transitions.
            before, after, force, feet, timestamp = sample
            target = paper_residual_wrench_target_numpy(before, after, force, feet, cfg.dt)
            physical = author_code_features_numpy(before, force, feet)
            if self.ssi is not None:
                phi = self.random_features.evaluate_input(physical)
                error = target - self.ssi.alpha @ phi
                self.ssi.alpha += 2 * cfg.ssi_learning_rate * np.outer(error, phi)
            elif self.gp is not None:
                error = -self.gp.update(physical, target / self.scale, timestamp) * self.scale
            elif self.trainer is not None:
                features = t2s_input_features(before, force, feet,
                    time_embedding_np(timestamp, d=self.time_feat_dim), self.feature_mode)
                error = target - exact_model_output(self.model, features[None])[0] * self.scale
                if self.hybrid_replay is None:
                    self.inputs.append(features.astype(np.float32))
                    self.targets.append((target / self.scale).astype(np.float32))
                else:
                    self.hybrid_replay.append(features, target / self.scale, timestamp)
            else:
                error = target
            prediction_errors.append(error)
            self.samples += 1
        sample_ms = 1000 * (time.perf_counter() - start)
        state = packet["state"].copy()
        self.go2.update_model(packet["q"], packet["dq"])
        feet_now = _negative_world_foot_levers(self.go2)
        delay = cfg.dt if cfg.timing != "ideal" else 0.
        residual_now = self.predict(state, packet["force"], feet_now, packet["time"])
        if cfg.delay_predictor == "model":
            predicted = predict_command_delay(state, packet["force"], feet_now,
                packet["time"], delay, self.gait, cfg.warmup, self.predict)
        else:
            predicted = state.copy()
            if cfg.delay_predictor == "kinematic":
                predicted[:6] += delay * paper_terms_numpy(state, feet_now)[3].ravel()
        plan_time = packet["time"] + delay
        absolute_time = cfg.warmup + plan_time
        first_dt = paper_first_element_duration(plan_time)
        plan_index = int(np.floor(plan_time / .03 + 1e-9))
        # The outer command filter publishes at 100 Hz. Extrapolate only its
        # known command, never the measured plant state, to the release time.
        command = packet["command"].copy()
        self.gait.target_speed = command[0]
        refs = author_twist_reference(to_world(predicted), target_speed=command[0],
            lateral_speed=command[1], yaw_rate=command[2], first_element_duration=first_dt)
        refs = refs[:, :PAPER_ACADOS_INTERVALS + 1]
        refs_paper = np.column_stack([world_state_to_paper_state(refs[:, i])
                                     for i in range(refs.shape[1])])
        nodes = self.gait.compute_contact_table(absolute_time, .03, PAPER_ACADOS_INTERVALS + 1).astype(bool)
        if self.previous_plan is not None:
            self.previous_plan = _shift_body_plan_for_new_grid(
                self.previous_plan, plan_index - self.previous_plan_index)
        levers, touchdowns = _horizon_negative_feet(
            self.go2, refs, nodes, body_plan_world=self.previous_plan)
        # The residual and RTI must expand around the SAME shifted/repaired
        # trajectory. Preparing inside solve after computing J_res is too late.
        self.solver.prepare_warm_start(predicted,refs_paper,nodes[:,:-1],plan_index=plan_index)
        embeddings = parameters = None
        linearization_ms = 0.
        def refresh_parameters():
            nonlocal linearization_ms
            linearization_start = time.perf_counter()
            expansion = t2s_horizon_features(
                self.solver, predicted, refs_paper, levers, nodes[:, :-1], embeddings,
                self.feature_mode)
            if self.gp is not None:
                output = self.gp.horizon_parameters(expansion, plan_time, first_dt)
            else:
                output = first_order_parameters(self.model, expansion)[0]
            linearization_ms += 1000*(time.perf_counter()-linearization_start)
            return output
        if self.model is not None:
            embeddings = horizon_time_codes(plan_time, first_dt, self.time_feat_dim)
            parameters = refresh_parameters()
        result = self.solver.solve(predicted, refs_paper, levers, nodes[:, :-1],
            first_element_duration=first_dt, plan_index=plan_index,
            alpha=self.ssi.alpha if self.ssi is not None else None,
            time_embeddings=embeddings, realtime_parameters=parameters,
            warm_start_prepared=True,
            realtime_parameter_refresh=refresh_parameters if self.model is not None else None)
        # Foothold planning consumes WORLD angular velocity. The WBC bridge
        # separately converts it back to BODY omega for velocity IK.
        body_plan = np.column_stack([to_world(result.states[:, i]) for i in range(result.states.shape[1])])
        self.previous_plan = body_plan.copy()
        self.previous_plan_index = plan_index
        submitted = False
        fast_due = slow_due = False
        replay_audit = None
        if self.trainer is not None:
            fast_due, slow_due = (due_updates(packet["cycle"], len(self.inputs), cfg)
                if self.hybrid_replay is None else self.hybrid_replay.due(packet["cycle"]))
            if fast_due or slow_due:
                if self.hybrid_replay is None:
                    fast_x = np.array(list(self.inputs)[-cfg.fast_batch:]) if fast_due else None
                    fast_y = np.array(list(self.targets)[-cfg.fast_batch:]) if fast_due else None
                    slow_x = slow_y = None
                    if slow_due:
                        slow_x, slow_y = sample_replay_without_replacement(
                            self.rng, list(self.inputs), list(self.targets), cfg.slow_batch)
                else:
                    fast_x, fast_y, slow_x, slow_y, replay_audit = self.hybrid_replay.batches(
                        self.rng, fast_due, slow_due)
                job = RealtimeT2SJob(
                    version=self.next_version, trigger_step=packet["cycle"],
                    trigger_wall_time=time.perf_counter(), fast_inputs=fast_x,
                    fast_targets=fast_y, slow_inputs=slow_x, slow_targets=slow_y)
                if self.latency_trainer:
                    logical_submit = packet["time"] + (job.trigger_wall_time - packet["observation_wall"])
                    submitted = self.latency_trainer.submit(job, logical_submit)
                else:
                    submitted = self.trainer.submit(job)
                if submitted:
                    self.next_version += 1
        ready = time.perf_counter()
        return dict(cycle=packet["cycle"], force=result.force, plan=body_plan,
                    first_dt=first_dt, plan_time=plan_time, touchdowns=touchdowns,
                    success=result.status == 0 and np.all(np.isfinite(result.force)),
                    solver_status=result.status, solve_ms=result.solve_ms,
                    worker_ms=1000*(ready-start), ready_wall=ready,
                    sample_ms=sample_ms, linearization_ms=linearization_ms,
                    cumulative_samples=self.samples, model_version=self.version,
                    residual=residual_now, prediction_error=(np.mean(prediction_errors, axis=0)
                      if prediction_errors else np.full(6, np.nan)),
                    training_events=training_events, training_submitted=submitted,
                    fast_due=fast_due, slow_due=slow_due,
                    retry_count=result.retry_count,
                    **({"replay_audit": replay_audit} if self.hybrid_replay is not None else {}))

    def close(self):
        if self.trainer is not None:
            trainer = self.latency_trainer or self.trainer
            for result in trainer.finish(timeout=10):
                self.training_log.append(dict(version=result.version, trigger_cycle=result.trigger_step,
                    completed_wall=result.ready_wall_time, activated_cycle=None,
                    train_ms=1000*result.train_seconds, error=result.error))
        return self.training_log


def controller_process(connection, config, cpu_core=None, training_core=None):
    controller = None
    try:
        if cpu_core is not None:
            os.sched_setaffinity(0, {cpu_core})
        controller = Go2ICRAController(config, training_core)
        gc.collect()
        gc.disable()
        connection.send({"initialized": True})
        while True:
            packet = connection.recv()
            if packet is None:
                break
            if packet.get("prepare_latency_cycle"):
                # Simulation bookkeeping only. Actual weight activation/copy
                # remains inside solve() and its measured controller budget.
                waited = (controller.latency_trainer.materialize()
                          if controller.latency_trainer else 0.)
                connection.send({"latency_cycle_prepared": True, "training_materialization_ms": 1000*waited})
                continue
            result = controller.solve(packet)
            connection.send(result)
        connection.send({"closed": True, "training_log": controller.close()})
        controller = None
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        gc.enable()
        if controller is not None:
            controller.close()
        connection.close()
