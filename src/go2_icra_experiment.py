"""Four-method, three-scenario Go2 benchmark with explicit control release.

paced_pipeline is a wall-paced co-simulation: a control is constant throughout
each 20 ms physical interval; MuJoCo/WBC can integrate that interval as a burst,
but its endpoint observation is not published before the next wall boundary.
An independent MPC process computes concurrently. A one-period input delay is
explicitly logged. The default uses the boundary measurement without an added
dynamics-based delay compensator. This is not a hard-RT guarantee.
Whole-period host stalls are logged and the wall schedule is re-anchored;
they do not abort the episode or advance simulation time during the pause.

latency_replay_pipeline isolates plant integration from timed controller calls.
Measured controller latency determines logical completion/release boundaries;
the one-period delay and rejection of late/stale results remain in force.
It is explicitly NOT a wall-paced or hard-real-time benchmark.
"""
from dataclasses import asdict
from pathlib import Path
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import threading
import time
import numpy as np
import mujoco as mj
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.leg_controller import LegController
from convex_mpc.mujoco_model import MuJoCo_GO2_Model
from src.go2_liquid import MuJoCoLiquidGo2Model, LiquidTankConfig
from src.go2_friction import MuJoCoRandomFrictionGo2Model
from src.go2_icra_protocol import Go2ICRAConfig, releasable
from src.go2_icra_clock import tick_time, resume_wall_boundary
from src.go2_icra_latency import stamp_latency_response
from src.go2_icra_controller import controller_process
from src.go2_paper_acados_experiment import (
    PaperAcadosExperimentConfig, _alternating_friction_config, _paper_rigid_payload_config,
    _initialize_paper_standing_pose, _disable_nonflat_world_geometry,
    _paper_world_state, _negative_world_foot_levers, _interpolate_body_plan,
)
from src.go2_paper_baseline import world_state_to_paper_state, paper_force_reference
from src.go2_icra_diagnostics import PlantDiagnostics, wbc_state_from_world_plan
from src.go2_paper_experiment import _AuthorDiscreteTrottingGait, _StandingGait
from src.go2_quad_sdk_low_level import PaperQuadSdkLowLevel
from src.go2_reference import path_following_body_command


def make_world(config):
    friction = _alternating_friction_config(PaperAcadosExperimentConfig(
        duration=config.duration, seed=config.seed, target_speed=config.speed)) if config.variable_friction else None
    if config.liquid:
        simulation = MuJoCoLiquidGo2Model(LiquidTankConfig(
            initial_fill_depth=.085, final_fill_depth=.035, container_mass=.6),
            protocol="liquid_drain", friction_config=friction)
    elif config.variable_friction:
        simulation = MuJoCoRandomFrictionGo2Model(friction,
            rigid_payload=_paper_rigid_payload_config(config.payload_mass))
    else:
        simulation = MuJoCo_GO2_Model()
    if not config.variable_friction:
        _disable_nonflat_world_geometry(simulation)
        # Inherit the old drain MJCF, including priority-one toe contacts.
    simulation.model.opt.timestep = 1. / config.physics_hz
    return simulation


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def dump_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def summarize(config, rows, jobs, status, elapsed, sample_count):
    def rmse(values):
        return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None
    timings = np.array([j["received_e2e_ms"] for j in jobs if "received_e2e_ms" in j])
    application_delays = np.array([j["application_e2e_ms"] for j in jobs
                                   if j.get("applied") and "application_e2e_ms" in j])
    metrics = dict(config=asdict(config), config_hash=config.fingerprint,
        status=status, full_duration=status == "completed", simulated_seconds=rows[-1]["time"] if rows else 0.,
        wall_seconds=elapsed, boundary_observations=len(rows), valid_transitions=sample_count,
        learned_samples=jobs[-1].get("cumulative_samples", 0) if jobs else 0,
        launched_jobs=len(jobs), applied_jobs=sum(j.get("applied", False) for j in jobs),
        solver_failures=sum(not j.get("success", False) for j in jobs),
        deadline_misses=sum(j.get("deadline_missed", False) for j in jobs),
        skipped_busy_cycles=sum(r["solver_busy"] for r in rows),
        maximum_boundary_lateness_ms=max((r["boundary_lateness_ms"] for r in rows), default=0.),
        platform_lateness_policy="record_and_rebase_no_abort",
        wall_clock_resynchronizations=sum(r.get("wall_resync_ms",0.) > 0 for r in rows),
        wall_clock_pause_total_ms=sum(r.get("wall_resync_ms",0.) for r in rows),
        strict_wall_clock_pacing=all(r["boundary_lateness_ms"] < 1000/config.mpc_hz for r in rows),
        deadline_basis="solution_received_before_original_job_deadline; actuation_at_next_simulation_boundary",
        gp_variant="official_approximate_STGP_non_cautious_shared_OCP",
        observations="MPC boundaries only; joint feedback at 500 Hz is WBC-only",
        command_release=f"one_period_pipeline_{config.delay_predictor}_estimate" if config.timing != "ideal" else "zero_delay_algorithmic_diagnostic",
    )
    if config.timing == "latency_replay_pipeline":
        metrics.update(platform_lateness_policy="plant_time_excluded_latency_replay",
            strict_wall_clock_pacing=False,
            deadline_basis="measured_observation_to_receipt_latency_le_period; release_on_logical_next_boundary",
            timing_claim="measured_latency_software_in_the_loop_not_wall_paced_or_hard_real_time",
            training_release="measured_worker_ready_latency_mapped_to_simulation_time; activation_copy_timed_in_MPC",
            plant_integration_ms_total=sum(r.get("plant_interval_ms", 0.) for r in rows[:-1]),
            training_materialization_ms_total=sum(r.get("training_materialization_ms", 0.) for r in rows),
            application_sim_delay_ms=1000*config.dt)
    # Do not let truncation produce a favorable formal RMSE. Prefix metrics
    # remain explicitly labeled diagnostics for failure investigation.
    evaluation = rows[:-1] if status == "completed" else rows
    values = dict(vx_rmse=rmse([r["vx_error"] for r in evaluation]),
                  height_rmse=rmse([r["z_error"] for r in evaluation]),
                  y_rmse=rmse([r["y_error"] for r in evaluation]),
                  roll_pitch_rmse=rmse([[r["roll"], r["pitch"]] for r in evaluation]),
                  pace_x_rmse=rmse([r["pace_x_error"] for r in evaluation]))
    metrics["tracking" if status == "completed" else "prefix_diagnostics"] = values
    metrics["steady_after_5s"] = ({
        "vx_rmse": rmse([r["vx_error"] for r in evaluation if r["time"] >= 5]),
        "height_rmse": rmse([r["z_error"] for r in evaluation if r["time"] >= 5]),
    } if status == "completed" else None)
    metrics["timing_ms"] = ({
        "mean": np.mean(timings), "p50": np.percentile(timings, 50),
        "p95": np.percentile(timings, 95), "p99": np.percentile(timings, 99),
        "max": np.max(timings), "over_5ms": np.count_nonzero(timings > 5),
        "over_20ms": np.count_nonzero(timings > 20),
    } if len(timings) else None)
    metrics["application_delay_ms"] = (dict(mean=float(np.mean(application_delays)),
        p95=float(np.percentile(application_delays,95)),max=float(np.max(application_delays)))
        if len(application_delays) else None)
    return metrics


def run_icra_experiment(config: Go2ICRAConfig, output_root):
    config.validate()
    output = Path(output_root).resolve() / f"{config.scenario}_{config.method}_seed{config.seed}_{config.timing}_{config.duration:g}s"
    # Refuse accidental result replacement, including incomplete prior runs.
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / "config.json", asdict(config))
    source_paths = [Path(__file__), Path(__file__).with_name("go2_icra_controller.py"),
        Path(__file__).with_name("go2_icra_protocol.py"), Path(__file__).with_name("go2_stgp.py"),
        Path(__file__).with_name("go2_icra_clock.py"),
        Path(__file__).with_name("go2_icra_latency.py"),
        Path(__file__).with_name("go2_icra_diagnostics.py"),
        Path(__file__).with_name("go2_paper_acados.py"), Path(__file__).with_name("go2_paper_baseline.py"),
        Path(__file__).with_name("go2_liquid.py"), Path(__file__).with_name("go2_friction.py"),
        Path(__file__).with_name("go2_quad_sdk_low_level.py"),
        Path(__file__).with_name("go2_paper_experiment.py"),
        Path(__file__).with_name("go2_paper_acados_experiment.py"),
        Path(__file__).with_name("realtime_t2s.py"), Path(__file__).with_name("realtime_neural.py"),
        Path(__file__).with_name("models.py"), Path(__file__).with_name("spatiotemporal_gp_mpc.py")]
    if getattr(config, "t2s_replay_mode", "fifo") == "fifo_reservoir":
        source_paths.extend([Path(__file__).with_name("go2_t2s_hybrid_replay.py"),
                             Path(__file__).with_name("hybrid_replay.py")])
    dump_json(output / "source_hashes.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    cpus = sorted(os.sched_getaffinity(0))
    execution_cores = cpus[-3:]
    dump_json(output / "hardware.json", dict(available_cpus=cpus, plant_core=execution_cores[0],
        controller_core=execution_cores[1] if len(execution_cores)>1 else None,
        trainer_core=execution_cores[2] if len(execution_cores)>2 else None,
        mujoco=mj.__version__, python=os.sys.version))
    go2 = PinGo2Model()
    low = PaperQuadSdkLowLevel(update_period=1./config.low_level_hz)
    stand_low = LegController()
    simulation = make_world(config)
    diagnostics = PlantDiagnostics(simulation, config)
    simulation.update_with_q_pin(_initialize_paper_standing_pose(go2, low))
    if config.liquid:
        simulation.data.qpos[-2:] = np.random.default_rng(config.seed + 400000).uniform(-.003, .003, 2)
    simulation.data.qvel[:] = 0.
    mj.mj_forward(simulation.model, simulation.data)
    dump_json(output / "world.json", dict(initial_mass=np.sum(simulation.model.body_mass),
        liquid=asdict(simulation.liquid_config) if config.liquid else None,
        friction=asdict(simulation.friction_config) if config.variable_friction else None,
        contact_geoms={n: dict(friction=simulation.model.geom_friction[mj.mj_name2id(simulation.model,mj.mjtObj.mjOBJ_GEOM,n)],
            priority=int(simulation.model.geom_priority[mj.mj_name2id(simulation.model,mj.mjtObj.mjOBJ_GEOM,n)]))
            for n in ("FL","FR","RL","RR")}))
    gait = _AuthorDiscreteTrottingGait(config.warmup, config.speed)
    stand = _StandingGait()
    command = np.zeros(3)
    torque = np.zeros(12)
    force = paper_force_reference(np.ones(4, dtype=bool))
    plan = None
    plan_time = 0.
    first_dt = .03
    boundary_state = np.zeros(12)
    initial_y = initial_yaw = 0.
    traces = []
    trace_times = []
    contacts_changed = False
    initial_mask = None
    boundary_dataset = []
    physics_tick = 0

    def advance_interval(warm=False):
        nonlocal command, torque, contacts_changed, physics_tick
        active_gait = stand if warm else gait
        for interval_step in range(round(config.dt*config.physics_hz)):
            # At 51.86 s accumulated float time crosses the gait tolerance.
            # Use one integer-derived clock for MPC/WBC hybrid events.
            step = physics_tick
            t = tick_time(step, config.physics_hz)
            if config.liquid:
                simulation.update_liquid_parameters(max(0., t - config.warmup))
            if step % (config.physics_hz // 100) == 0 and (warm or interval_step > 0):
                raw = np.array([config.speed, 0., 0.]) if warm else path_following_body_command(
                    boundary_state, target_speed=config.speed, lateral_position_reference=initial_y,
                    heading_reference=initial_yaw, lateral_gain=1., heading_gain=3.)
                command = .1*raw + .9*command
                gait.target_speed = command[0]
            if step % (config.physics_hz // config.low_level_hz) == 0:
                simulation.update_pin_with_mujoco(go2)
                yaw = go2.current_config.compute_euler_angle_world()[2]
                c, s = np.cos(yaw), np.sin(yaw)
                go2.x_vel_des_world = c*command[0]-s*command[1]
                go2.y_vel_des_world = s*command[0]+c*command[1]
                go2.x_pos_des_world = float(go2.pos_com_world[0])
                go2.y_pos_des_world = float(go2.pos_com_world[1])
                go2.yaw_rate_des_world = command[2]
                if warm:
                    torque = np.concatenate([stand_low.compute_leg_torque(leg, go2, stand,
                        force[3*i:3*i+3], t).tau for i, leg in enumerate(("FL","FR","RL","RR"))])
                    torque = np.clip(torque, -np.array([23.7,23.7,45.]*4), np.array([23.7,23.7,45.]*4))
                else:
                    desired = _interpolate_body_plan(plan, t-config.warmup-plan_time, first_dt) if plan is not None else None
                    # The foothold planner uses world omega, but velocity IK
                    # uses Pinocchio's body-frame free-flyer angular tangent.
                    desired = wbc_state_from_world_plan(desired)
                    torque = low.compute(go2, gait, force, t, desired_body_state=desired).torque
            mask = active_gait.compute_current_mask(t)
            if initial_mask is not None and not np.array_equal(mask, initial_mask):
                contacts_changed = True
            simulation.set_joint_torque(torque)
            if config.variable_friction:
                mj.mj_step1(simulation.model, simulation.data)
                simulation.apply_spatial_contact_friction()
                mj.mj_step2(simulation.model, simulation.data)
            else:
                mj.mj_step(simulation.model, simulation.data)
            physics_tick += 1
            if not warm:
                diagnostics.observe()
            if step % 40 == 0 and not warm:
                traces.append(simulation.data.qpos.copy())
                trace_times.append(tick_time(physics_tick,config.physics_hz)-config.warmup)

    advance_interval(warm=True)
    simulation.update_pin_with_mujoco(go2)
    boundary_state = world_state_to_paper_state(_paper_world_state(go2))
    initial_x, initial_y, initial_yaw = boundary_state[0], boundary_state[1], boundary_state[5]
    force = paper_force_reference(gait.compute_current_mask(config.warmup))
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    responses = queue.Queue()

    def receive_responses():
        try:
            os.sched_setaffinity(0, {execution_cores[0]})
            while True:
                result = parent.recv()
                result["received_wall"] = time.perf_counter()
                responses.put(result)
                if result.get("closed") or "error" in result:
                    break
        except (EOFError, OSError):
            pass
    worker = context.Process(target=controller_process,
        args=(child, config, execution_cores[1] if len(execution_cores)>1 else None,
              execution_cores[2] if len(execution_cores)>2 else None), daemon=False)
    worker.start()
    child.close()
    rows, jobs, pending_samples = [], [], []
    previous = None
    busy = False
    isolated = config.timing == "latency_replay_pipeline"
    pending_response = None
    launch = None
    sample_count = 0
    status = "completed"
    training_log = []
    started = time.perf_counter()
    gc_was_enabled = gc.isenabled()
    try:
        if not parent.poll(60):
            raise TimeoutError("controller initialization exceeded 60 s")
        init = parent.recv()
        if "error" in init:
            raise RuntimeError(init["error"])
        if not init.get("initialized"):
            raise RuntimeError("bad controller startup message")
        receiver = threading.Thread(target=receive_responses, daemon=True)
        receiver.start()
        os.sched_setaffinity(0, {execution_cores[0]})
        gc.collect()
        gc.disable()  # bounded episode; avoid cyclic-GC pauses in the timed loop
        epoch = time.perf_counter() + .05
        wall_shift = 0.
        last_applied_cycle = -1
        for cycle in range(round(config.duration * config.mpc_hz) + 1):
            materialization_ms = 0.
            if isolated:
                # Obtain training completion metadata BEFORE starting this
                # observation's MPC stopwatch. Future weights stay gated by
                # their measured logical ready time inside the controller.
                parent.send({"prepare_latency_cycle": True})
                prepared = responses.get(timeout=15)
                if not prepared.get("latency_cycle_prepared"):
                    raise RuntimeError(f"Bad latency preparation response: {prepared.get('error')}")
                materialization_ms = prepared["training_materialization_ms"]
            planned_wall_boundary = epoch + wall_shift + cycle * config.dt
            wall_boundary = planned_wall_boundary
            if config.timing == "paced_pipeline":
                remaining = wall_boundary - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            observation_wall = time.perf_counter()
            resync = 0.
            if config.timing == "paced_pipeline":
                wall_boundary, resync = resume_wall_boundary(
                    planned_wall_boundary, observation_wall, config.dt)
                wall_shift += resync
            simulation.update_pin_with_mujoco(go2)
            boundary_state = world_state_to_paper_state(_paper_world_state(go2))
            t = cycle * config.dt
            feet = _negative_world_foot_levers(go2)
            mask = gait.compute_current_mask(config.warmup+t)
            raw = path_following_body_command(boundary_state, target_speed=config.speed,
                lateral_position_reference=initial_y, heading_reference=initial_yaw,
                lateral_gain=1., heading_gain=3.)
            command = .1*raw + .9*command
            gait.target_speed = command[0]
            transition_valid = previous is not None and not contacts_changed and np.array_equal(mask, initial_mask)
            if transition_valid:
                pending_samples.append((previous[0], boundary_state.copy(), previous[1], previous[2], previous[3]))
                sample_count += 1
            applied = False
            if isolated and busy and pending_response["available_cycle"] <= cycle:
                response = pending_response
                applied = (response["success"] and not response["deadline_missed"]
                           and response["cycle"]+1 == cycle)
                response.update(applied=applied, applied_cycle=cycle if applied else None)
                jobs.append(response)
                if applied:
                    force, plan = response["force"], response["plan"]
                    response["applied_wall"] = time.perf_counter()
                    response["application_e2e_ms"] = 1000*(response["applied_wall"]-response["observation_wall"])
                    response["application_sim_delay_ms"] = 1000*config.dt
                    plan_time, first_dt = response["plan_time"], response["first_dt"]
                    gait.set_planned_touchdowns(response["touchdowns"])
                    last_applied_cycle = cycle
                busy = False
                pending_response = None
            if config.timing == "paced_pipeline" and busy and not responses.empty():
                response = responses.get_nowait()
                received_wall = response["received_wall"]
                if "error" in response:
                    raise RuntimeError(response["error"])
                response["received_e2e_ms"] = 1000*(received_wall-launch["observation_wall"])
                response["compute_e2e_ms"] = 1000*(response["ready_wall"]-launch["observation_wall"])
                response["observation_wall"] = launch["observation_wall"]
                response["deadline_wall"] = launch["deadline"]
                response["deadline_missed"] = received_wall > launch["deadline"]
                applied = releasable(response["cycle"], cycle, received_wall, launch["deadline"], response["success"])
                response["applied"] = applied
                response["applied_cycle"] = cycle if applied else None
                jobs.append(response)
                if applied:
                    force, plan = response["force"], response["plan"]
                    response["applied_wall"] = time.perf_counter()
                    response["application_e2e_ms"] = 1000*(response["applied_wall"]-launch["observation_wall"])
                    plan_time, first_dt = response["plan_time"], response["first_dt"]
                    gait.set_planned_touchdowns(response["touchdowns"])
                    last_applied_cycle = cycle
                busy = False
            row = dict(time=t, x=boundary_state[0], y=boundary_state[1], z=boundary_state[2],
                previous_transition_valid=transition_valid,
                roll=boundary_state[3], pitch=boundary_state[4], yaw=boundary_state[5],
                vx=boundary_state[6], vy=boundary_state[7], vz=boundary_state[8],
                omega_x=boundary_state[9], omega_y=boundary_state[10], omega_z=boundary_state[11],
                vx_error=boundary_state[6]-config.speed, z_error=boundary_state[2]-config.height,
                y_error=boundary_state[1]-initial_y, pace_x_error=boundary_state[0]-initial_x-config.speed*t,
                force_x=np.sum(force.reshape(4,3)[:,0]), force_z=np.sum(force.reshape(4,3)[:,2]),
                payload_mass=(simulation.liquid_properties.liquid_mass + simulation.liquid_config.container_mass
                              if config.liquid else config.payload_mass if config.variable_friction else 0.),
                liquid_mass=simulation.liquid_properties.liquid_mass if config.liquid else 0.,
                friction_at_body=(simulation.friction_at(boundary_state[0])[0] if config.variable_friction
                    else float(simulation.model.geom("FL").friction[0])),
                solver_busy=busy, fresh_control=applied, command_age_cycles=cycle-last_applied_cycle,
                boundary_lateness_ms=max(0.,1000*(observation_wall-planned_wall_boundary)) if config.timing == "paced_pipeline" else 0.,
                wall_resync_ms=1000*resync, wall_pause_cumulative_ms=1000*wall_shift)
            if isolated:
                row.update(training_materialization_ms=materialization_ms, plant_interval_ms=0.)
            rows.append(row)
            if boundary_state[2] < .12 or max(abs(boundary_state[3:5])) > 1.0 or not np.isfinite(boundary_state).all():
                status = "fell"
                break
            if cycle == round(config.duration * config.mpc_hz):
                break
            if not busy:
                packet = dict(cycle=cycle, time=t, state=boundary_state.copy(),
                    q=go2.current_config.get_q().copy(), dq=go2.current_config.get_dq().copy(),
                    force=force.copy(), command=(raw + (command-raw)*.9**2
                        if config.timing != "ideal" else command.copy()), samples=pending_samples)
                if isolated:
                    packet["observation_wall"] = observation_wall
                parent.send(packet)
                pending_samples = []
                launch = dict(observation_wall=observation_wall, deadline=wall_boundary+config.dt)
                busy = True
                if isolated:
                    # Waiting here changes wall runtime, never simulation time.
                    # Even a result already in memory cannot drive the plant
                    # until its measured logical completion boundary.
                    response = responses.get(timeout=15)
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    pending_response = stamp_latency_response(response, observation_wall, config.dt)
                    launch["deadline"] = pending_response["deadline_wall"]
                if config.timing == "ideal":
                    response = responses.get(timeout=10)
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    response["received_e2e_ms"] = 1000*(response["received_wall"]-observation_wall)
                    response["compute_e2e_ms"] = response["received_e2e_ms"]
                    response["deadline_missed"] = False
                    response["applied"] = response["success"]
                    response["applied_cycle"] = cycle if response["success"] else None
                    jobs.append(response)
                    if response["success"]:
                        force, plan = response["force"], response["plan"]
                        plan_time, first_dt = response["plan_time"], response["first_dt"]
                        gait.set_planned_touchdowns(response["touchdowns"])
                    busy = False
                    rows[-1]["fresh_control"] = response["success"]
                    if response["success"]:
                        last_applied_cycle = cycle
            rows[-1]["force_x"] = np.sum(force.reshape(4,3)[:,0])
            rows[-1]["force_z"] = np.sum(force.reshape(4,3)[:,2])
            rows[-1]["command_age_cycles"] = cycle-last_applied_cycle
            boundary_dataset.append((t, boundary_state.copy(), force.copy(), feet.copy(), transition_valid))
            previous = (boundary_state.copy(), force.copy(), feet.copy(), t)
            initial_mask = mask.copy()
            contacts_changed = False
            plant_start = time.perf_counter()
            advance_interval()
            if isolated:
                rows[-1]["plant_interval_ms"] = 1000*(time.perf_counter()-plant_start)
            if cycle % (5*config.mpc_hz) == 0:
                print(f"{config.scenario}/{config.method}/{config.seed} t={t:.1f}s vx={boundary_state[6]:.3f} z={boundary_state[2]:.3f}", flush=True)
            if (Path(output_root)/"STOP").exists():
                status = "stopped_by_request"
                break
            if time.perf_counter() - epoch > max(60., 4*config.duration):
                status = "wall_timeout"
                break
        if busy:
            trailing = pending_response if isolated else responses.get(timeout=10)
            if "error" in trailing:
                raise RuntimeError(trailing["error"])
            trailing.update(applied=False, applied_cycle=None, deadline_missed=trailing["received_wall"]>launch["deadline"],
                            observation_wall=launch["observation_wall"], deadline_wall=launch["deadline"],
                            received_e2e_ms=1000*(trailing["received_wall"]-launch["observation_wall"]))
            jobs.append(trailing)
        parent.send(None)
        closed = responses.get(timeout=15)
        if "error" in closed:
            raise RuntimeError(closed["error"])
        training_log = closed.get("training_log", [])
    except BaseException as error:
        status = "error"
        dump_json(output / "error.json", dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if gc_was_enabled:
            gc.enable()
        # Close only the process created for this trial. Never kill unrelated jobs.
        if worker.is_alive():
            try:
                parent.send(None)
            except (OSError, EOFError):
                pass
        worker.join(timeout=12)
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=2)
        parent.close()
        os.sched_setaffinity(0, set(cpus))
        # A stop/watchdog can fire just after a fully integrated interval.
        # Retain its endpoint measurement, so states = controls + one always.
        if len(boundary_dataset) == len(rows) and rows:
            simulation.update_pin_with_mujoco(go2)
            endpoint = world_state_to_paper_state(_paper_world_state(go2))
            terminal = rows[-1].copy()
            terminal["time"] = tick_time(physics_tick,config.physics_hz)-config.warmup
            for name, value in zip(("x","y","z","roll","pitch","yaw","vx","vy","vz","omega_x","omega_y","omega_z"), endpoint):
                terminal[name] = value
            terminal.update(vx_error=endpoint[6]-config.speed,z_error=endpoint[2]-config.height,
                y_error=endpoint[1]-initial_y,pace_x_error=endpoint[0]-initial_x-config.speed*terminal["time"],
                fresh_control=False, solver_busy=False, previous_transition_valid=False)
            rows.append(terminal)
        elapsed = time.perf_counter()-started
        with (output / "trajectory.csv").open("w", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        dump_json(output / "jobs.json", jobs)
        dump_json(output / "training.json", training_log)
        np.savez_compressed(output / "replay.npz", time=np.array(trace_times), qpos=np.array(traces))
        np.savez_compressed(output / "boundary_data.npz",
            time=np.array([r["time"] for r in rows]),
            state=np.array([[r[n] for n in ("x","y","z","roll","pitch","yaw","vx","vy","vz","omega_x","omega_y","omega_z")] for r in rows]),
            command=np.array([r[2] for r in boundary_dataset]),
            negative_feet=np.array([r[3] for r in boundary_dataset]),
            previous_transition_valid=np.array([r["previous_transition_valid"] for r in rows]))
        metrics = summarize(config, rows, jobs, status, elapsed, sample_count)
        metrics["plant_diagnostics"] = diagnostics.summary()
        applied_learning_jobs = [j for j in jobs if j.get("applied") and j.get("model_version", 0)>0]
        metrics["applied_neural_model_versions"] = sorted(set(j["model_version"] for j in applied_learning_jobs))
        metrics["controls_using_updated_neural_model"] = len(applied_learning_jobs)
        dump_json(output / "metrics.json", metrics)
    return metrics, output
