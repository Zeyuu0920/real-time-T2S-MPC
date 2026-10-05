"""New, isolated Go2 intra-period/delayed-observation experiment driver."""
from dataclasses import asdict, replace
from pathlib import Path
import csv
import gc
import multiprocessing as mp
import os
import time
import traceback
import numpy as np
import mujoco as mj
from convex_mpc.go2_robot_data import PinGo2Model
from convex_mpc.leg_controller import LegController
from src import go2_icra_experiment as old
from src.go2_paper_acados_experiment import (
    _initialize_paper_standing_pose, _paper_world_state,
    _negative_world_foot_levers, _interpolate_body_plan,
)
from src.go2_paper_baseline import world_state_to_paper_state, paper_force_reference
from src.go2_paper_experiment import _AuthorDiscreteTrottingGait, _StandingGait
from src.go2_quad_sdk_low_level import PaperQuadSdkLowLevel
from src.go2_icra_diagnostics import PlantDiagnostics, wbc_state_from_world_plan
from src.go2_reference import path_following_body_command
from src.go2_icra_clock import tick_time
from src.deadline_control import ComputationAwareControlRelease
from src.go2_realism_io import BoundarySensor, JointResponse, aligned_transition, same_contact_mask
from src.go2_sim2real_config import PROTOCOL


NAMES = ('x','y','z','roll','pitch','yaw','vx','vy','vz','omega_x','omega_y','omega_z')


def retain_completed_endpoint(rows, completed_intervals, capture_boundary):
    """Record a completed interval's missing endpoint without another control call.

    STOP/watchdog and interruption can occur after physics has advanced but
    before the next MPC boundary. Ordinary completion/falls already have their
    endpoint, while an incomplete physics interval has no full command history.
    """
    if rows and len(rows) == completed_intervals:
        row, _, _, _ = capture_boundary(completed_intervals, 0.)
        rows.append(row)


def run_trial(config, options, output, controller_target, run_metadata=None):
    config.validate()
    options.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    old.dump_json(output/'config.json', asdict(config))
    old.dump_json(output/'interfaces.json', asdict(options))
    if run_metadata is not None:
        old.dump_json(output/'run_protocol.json', run_metadata)
    cpus = sorted(os.sched_getaffinity(0))
    cores = cpus[-3:]
    if len(cores) < 3:
        raise RuntimeError('Need three separate cores for measured controller/trainer/plant')
    old.dump_json(output/'hardware.json', dict(cores=cores, available_cpus=cpus,
                                             mujoco=mj.__version__, python=os.sys.version))
    go2, sensor_model = PinGo2Model(), PinGo2Model()
    low = PaperQuadSdkLowLevel(update_period=1/config.low_level_hz)
    stand_low = LegController()
    # Smoke-test duration must not shorten/resample the frozen 60 s world.
    simulation = old.make_world(replace(config, duration=60.))
    diagnostics = PlantDiagnostics(simulation, config)
    simulation.update_with_q_pin(_initialize_paper_standing_pose(go2, low))
    if config.liquid:
        simulation.data.qpos[-2:] = np.random.default_rng(config.seed+400000).uniform(-.003,.003,2)
    simulation.data.qvel[:] = 0.
    mj.mj_forward(simulation.model, simulation.data)
    old.dump_json(output/'world.json', dict(initial_mass=np.sum(simulation.model.body_mass),
        liquid=asdict(simulation.liquid_config) if config.liquid else None,
        friction=asdict(simulation.friction_config) if config.variable_friction else None,
        contact_geoms={n:dict(friction=simulation.model.geom_friction[mj.mj_name2id(simulation.model,mj.mjtObj.mjOBJ_GEOM,n)],
          priority=int(simulation.model.geom_priority[mj.mj_name2id(simulation.model,mj.mjtObj.mjOBJ_GEOM,n)]))
          for n in ('FL','FR','RL','RR')}))
    gait, stand = _AuthorDiscreteTrottingGait(config.warmup, config.speed), _StandingGait()
    command = np.zeros(3)
    torque = np.zeros(12)
    force = paper_force_reference(np.ones(4,dtype=bool))
    plan, plan_time, first_dt = None, 0., .03
    boundary_state = np.zeros(12)
    initial_y = initial_yaw = 0.
    physics_tick = 0
    actuator = None
    traces, trace_times = [], []
    command_history, publication_history, torque_requests, torque_actual = [], [], [], []

    def install(response):
        nonlocal force, plan, plan_time, first_dt
        force, plan = response['force'].copy(), response['plan'].copy()
        plan_time, first_dt = response['plan_time'], response['first_dt']
        gait.set_planned_touchdowns(response['touchdowns'])

    def advance(warm=False, decision=None, response=None):
        nonlocal physics_tick, command, torque
        active_gait = stand if warm else gait
        mask_start = active_gait.compute_current_mask(tick_time(physics_tick,1000))
        changed = False
        consumed_force = force.copy()
        consumed, published, requested, actual = [], [], [], []
        for j in range(20):
            step = physics_tick
            t = tick_time(step,1000)
            if decision is not None and decision.candidate_accepted and j == decision.held_substeps:
                install(response)
            if config.liquid:
                simulation.update_liquid_parameters(max(0.,t-config.warmup))
            if step % 10 == 0 and (warm or j > 0):
                raw = np.array([config.speed,0.,0.]) if warm else path_following_body_command(
                    boundary_state,target_speed=config.speed,lateral_position_reference=initial_y,
                    heading_reference=initial_yaw,lateral_gain=1.,heading_gain=3.)
                command = .1*raw+.9*command
                gait.target_speed = command[0]
            mask = active_gait.compute_current_mask(t).astype(bool).reshape(4)
            changed |= not same_contact_mask(mask,mask_start)
            if step % 2 == 0:
                simulation.update_pin_with_mujoco(go2)
                yaw = go2.current_config.compute_euler_angle_world()[2]
                c,s = np.cos(yaw),np.sin(yaw)
                go2.x_vel_des_world = c*command[0]-s*command[1]
                go2.y_vel_des_world = s*command[0]+c*command[1]
                go2.x_pos_des_world = float(go2.pos_com_world[0])
                go2.y_pos_des_world = float(go2.pos_com_world[1])
                go2.yaw_rate_des_world = command[2]
                if warm:
                    torque = np.concatenate([stand_low.compute_leg_torque(leg,go2,stand,
                        force[3*i:3*i+3],t).tau for i,leg in enumerate(('FL','FR','RL','RR'))])
                    limit = np.array([23.7,23.7,45.]*4)
                    torque = np.clip(torque,-limit,limit)
                else:
                    desired = _interpolate_body_plan(plan,t-config.warmup-plan_time,first_dt) if plan is not None else None
                    torque = low.compute(go2,gait,force,t,
                        desired_body_state=wbc_state_from_world_plan(desired)).torque
                # Record exactly the 12-vector passed to the unchanged WBC;
                # do not redefine the old command-based residual with a mask.
                consumed_force = force.copy()
            applied_torque = torque if warm else actuator.step(torque)
            simulation.set_joint_torque(applied_torque)
            if config.variable_friction:
                mj.mj_step1(simulation.model,simulation.data)
                simulation.apply_spatial_contact_friction()
                mj.mj_step2(simulation.model,simulation.data)
            else:
                mj.mj_step(simulation.model,simulation.data)
            physics_tick += 1
            if not warm:
                diagnostics.observe()
                consumed.append(consumed_force.copy())
                published.append(force.copy())
                requested.append(torque.copy())
                actual.append(applied_torque.copy())
                if step % 40 == 0:
                    traces.append(simulation.data.qpos.copy())
                    trace_times.append(tick_time(physics_tick,1000)-config.warmup)
        changed |= not same_contact_mask(active_gait.compute_current_mask(tick_time(physics_tick,1000)),mask_start)
        # Exactly-deadline result has no effect inside the just-ended interval.
        if decision is not None and decision.candidate_accepted and decision.held_substeps == 20:
            install(response)
        if not warm:
            command_history.append(np.asarray(consumed))
            publication_history.append(np.asarray(published))
            torque_requests.append(np.asarray(requested))
            torque_actual.append(np.asarray(actual))
        return not changed

    advance(warm=True)
    simulation.update_pin_with_mujoco(go2)
    initial_state = world_state_to_paper_state(_paper_world_state(go2))
    initial_x, initial_y, initial_yaw = initial_state[0],initial_state[1],initial_state[5]
    force = paper_force_reference(gait.compute_current_mask(config.warmup))
    actuator = JointResponse(options,config.seed,.001,torque)
    old.dump_json(output/'actuator_gain.json', actuator.gain)
    sensor = BoundarySensor(options,config.seed,config.dt)
    release = ComputationAwareControlRelease(force)
    context = mp.get_context('spawn')
    parent,child = context.Pipe()
    worker = context.Process(target=controller_target,args=(child,config,cores[1],cores[2]),daemon=False)
    worker.start()
    child.close()
    rows,jobs,packets,eligible,training,learn_samples = [],[],[],[],[],[]
    state_history,q_history,dq_history,noise_history,source_indices = [],[],[],[],[]
    status = 'completed'
    started = time.perf_counter()
    gc_enabled = gc.isenabled()

    def capture_boundary(cycle, materialization_ms):
        """Capture one coherent boundary for control or terminal diagnostics."""
        t = cycle * config.dt
        simulation.update_pin_with_mujoco(go2)
        truth = world_state_to_paper_state(_paper_world_state(go2))
        q,dq = sensor.observe(go2.current_config.get_q(),go2.current_config.get_dq())
        sensor_model.update_model(q,dq)
        observed = world_state_to_paper_state(_paper_world_state(sensor_model))
        feet = _negative_world_foot_levers(sensor_model)
        packets.append(dict(time=t,state=observed,q=q,dq=dq,feet=feet))
        q_history.append(q.copy());dq_history.append(dq.copy());noise_history.append(sensor.noise.copy())
        state_history.append(truth.copy())
        source = max(0,cycle-options.observation_delay_steps)
        source_indices.append(source)
        measurement = packets[source]
        row = dict(time=t,**dict(zip(NAMES,truth)),vx_error=truth[6]-config.speed,
            z_error=truth[2]-config.height,y_error=truth[1]-initial_y,
            pace_x_error=truth[0]-initial_x-config.speed*t,
            friction_at_body=(simulation.friction_at(truth[0])[0] if config.variable_friction
                else float(simulation.model.geom('FL').friction[0])),
            payload_mass=(simulation.liquid_properties.liquid_mass+simulation.liquid_config.container_mass if config.liquid else config.payload_mass),
            source_time=measurement['time'],observation_age_ms=1000*(t-measurement['time']),
            solver_busy=False,fresh_control=False,boundary_lateness_ms=0.,plant_interval_ms=0.,
            training_materialization_ms=materialization_ms,
            previous_transition_valid=eligible[-1] if eligible else False)
        return row, truth, measurement, source

    def receive(timeout=30):
        if not parent.poll(timeout):
            raise TimeoutError('Owned controller response timed out')
        result = parent.recv()
        result['received_wall'] = time.perf_counter()
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    try:
        assert receive(60).get('initialized')
        os.sched_setaffinity(0,{cores[0]})
        gc.collect()
        gc.disable()
        last_source = -1
        for cycle in range(round(config.duration/config.dt)+1):
            t = cycle*config.dt
            parent.send({'prepare_latency_cycle':True})
            prepared = receive()
            assert prepared.get('latency_cycle_prepared')
            observation_wall = time.perf_counter()
            row, truth, measurement, source = capture_boundary(cycle, prepared['training_materialization_ms'])
            boundary_state = measurement['state'].copy()
            raw = path_following_body_command(boundary_state,target_speed=config.speed,
                lateral_position_reference=initial_y,heading_reference=initial_yaw,lateral_gain=1.,heading_gain=3.)
            command = .1*raw+.9*command
            gait.target_speed = command[0]
            rows.append(row)
            if truth[2] < .12 or np.max(np.abs(truth[3:5])) > 1 or not np.isfinite(truth).all():
                status='fell';break
            if cycle == round(config.duration/config.dt):
                break
            samples=[]
            if source > last_source:
                sample = aligned_transition(packets,command_history,eligible,source)
                if sample is not None:
                    samples=[sample]
                    learn_samples.append((cycle,source-1,*sample))
            last_source=source
            packet=dict(cycle=cycle,time=t,source_time=measurement['time'],
                state=boundary_state.copy(),q=measurement['q'],dq=measurement['dq'],
                force=force.copy(),command=command.copy(),samples=samples,observation_wall=observation_wall)
            parent.send(packet)
            response=receive()
            latency=1000*(response['received_wall']-observation_wall)
            decision=release.release(response['force'],compute_ms=latency,deadline_ms=20.,
                substep_ms=1.,substep_count=20,candidate_step=cycle,solver_success=response['success'])
            response.update(received_e2e_ms=latency,
                compute_e2e_ms=1000*(response['ready_wall']-observation_wall),
                observation_wall=observation_wall,deadline_missed=latency>20.,
                applied=decision.candidate_accepted,applied_cycle=cycle if decision.candidate_accepted else None,
                application_e2e_ms=float(decision.held_substeps) if decision.candidate_accepted else None,
                publication_time=t+decision.held_substeps*.001 if decision.candidate_accepted else None,
                wbc_consumption_time=t+2*np.ceil(decision.held_substeps/2)*.001 if decision.candidate_accepted else None,
                held_substeps=decision.held_substeps,candidate_substeps=decision.candidate_substeps,
                source_index=source,source_time=measurement['time'],
                sample_source_intervals=[source-1] if samples else [])
            jobs.append(response)
            rows[-1]['fresh_control']=decision.candidate_accepted
            plant_start=time.perf_counter()
            eligible.append(advance(decision=decision,response=response))
            rows[-1]['plant_interval_ms']=1000*(time.perf_counter()-plant_start)
            if cycle % 250 == 0:
                print(f"{config.scenario}/{config.method} t={t:.2f} vx={truth[6]:.3f} z={truth[2]:.3f} E2E={latency:.2f}ms samples={response['cumulative_samples']} version={response['model_version']}",flush=True)
            if time.perf_counter()-started > 420:
                raise TimeoutError('420 s single-trial watchdog; no automatic retry')
            if (output.parent/'STOP').exists():
                raise RuntimeError('STOP requested')
        parent.send(None)
        closed=receive()
        assert closed.get('closed')
        training=closed['training_log']
    except BaseException:
        status='error'
        old.dump_json(output/'error.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        if gc_enabled: gc.enable()
        if worker.is_alive():
            try: parent.send(None)
            except (OSError,EOFError): pass
        worker.join(timeout=12)
        if worker.is_alive(): worker.terminate();worker.join(timeout=2)
        parent.close()
        os.sched_setaffinity(0,set(cpus))
        retain_completed_endpoint(rows, len(command_history), capture_boundary)
        if rows:
            with (output/'trajectory.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        old.dump_json(output/'jobs.json',jobs)
        old.dump_json(output/'training.json',training)
        np.savez_compressed(output/'replay.npz',time=trace_times,qpos=traces)
        np.savez_compressed(output/'boundary_data.npz',time=[r['time'] for r in rows],state=state_history,
            observed_state=[p['state'] for p in packets],observed_feet=[p['feet'] for p in packets],
            q=q_history,dq=dq_history,noise=noise_history,delivered_source_index=source_indices,
            command_schedule=command_history,publication_schedule=publication_history,
            torque_requested=torque_requests,torque_applied=torque_actual,eligible=eligible)
        old.dump_json(output/'learning_samples.json',learn_samples)
        metrics=old.summarize(config,rows,jobs,status,time.perf_counter()-started,len(learn_samples))
        intervals=len(command_history)
        metrics.update(protocol=PROTOCOL,
            observation_delay_steps=options.observation_delay_steps,
            observation_delay_ms=1000*config.dt*options.observation_delay_steps,
            control_intervals=intervals,denominator=intervals,
            intervals_without_accepted_candidate=sum(not j['applied'] for j in jobs),
            intervals_without_new_candidate_effect=sum(
                not j['applied'] or j['wbc_consumption_time'] >= (j['cycle']+1)*config.dt-1e-12
                for j in jobs),
            command_release='quadrotor_ComputationAwareControlRelease_intra_period_1ms',
            deadline_basis='measured_sensor_to_receipt_latency_le_20ms; 1ms publication; 2ms WBC consumption',
            observations='50Hz configurable-delay noisy MPC estimator; separate ideal 500Hz WBC proprioception',
            legacy_config_timing_role='latency_replay_pipeline retains learner readiness gating; command release is intra-period',
            application_sim_delay_ms=None,
            solve_mean_ms=float(np.mean([j['solve_ms'] for j in jobs])) if jobs else None,
            plant_diagnostics=diagnostics.summary(),
            timing_claim='measured_latency_SIL; per_tick_jobs_like_quadrotor; NOT_hard_realtime_busy_queue',
            sensor_actuator_claim='synthetic_stress_test_not_hardware_calibrated')
        old.dump_json(output/'metrics.json',metrics)
    return metrics
