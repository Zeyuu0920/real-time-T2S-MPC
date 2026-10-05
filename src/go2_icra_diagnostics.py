"""Plant-only diagnostics and the explicit world-plan to body-omega WBC bridge.

Diagnostics are never included in the controller packet or residual labels.
"""
import mujoco as mj
import numpy as np
from src.go2_paper_baseline import world_state_to_paper_state


def wbc_state_from_world_plan(state):
    """Keep world p/v and Euler angles; rotate omega into the body tangent."""
    return None if state is None else world_state_to_paper_state(state)


class PlantDiagnostics:
    def __init__(self, simulation, config):
        self.simulation = simulation
        self.config = config
        self.foot_geoms = {mj.mj_name2id(simulation.model,mj.mjtObj.mjOBJ_GEOM,n)
                           for n in ("FL","FR","RL","RR")}
        self.steps = self.contact_steps = self.low_steps = self.contact_count = 0
        self.mu_min, self.mu_max, self.mu_sum = np.inf, -np.inf, 0.
        self.modal_max = 0.
        self.modal_limit_steps = 0

    def observe(self):
        self.steps += 1
        any_contact = any_low = False
        for contact in self.simulation.data.contact:
            if contact.geom1 not in self.foot_geoms and contact.geom2 not in self.foot_geoms:
                continue
            ground = contact.geom2 if contact.geom1 in self.foot_geoms else contact.geom1
            if self.simulation.model.geom_bodyid[ground] != 0:
                continue
            value = float(contact.friction[0])
            self.mu_min = min(self.mu_min,value)
            self.mu_max = max(self.mu_max,value)
            self.mu_sum += value
            self.contact_count += 1
            any_contact = True
            any_low = any_low or value <= .10+1e-12
        self.contact_steps += int(any_contact)
        self.low_steps += int(any_low)
        if self.config.liquid:
            amplitude = float(np.max(np.abs(self.simulation.data.qpos[-2:])))
            self.modal_max = max(self.modal_max,amplitude)
            # The old model starts its artificial limit constraint 5 mm
            # before the nominal +/-70 mm travel. Record that regime honestly.
            self.modal_limit_steps += int(amplitude >= self.simulation.liquid_config.maximum_modal_travel-.005)

    def summary(self):
        return dict(physics_steps=self.steps,
            contact_mu_min=self.mu_min if self.contact_count else None,
            contact_mu_max=self.mu_max if self.contact_count else None,
            contact_mu_contact_count_mean=self.mu_sum/self.contact_count if self.contact_count else None,
            seconds_with_any_foot_contact=self.contact_steps/self.config.physics_hz,
            seconds_with_any_low_mu_contact=self.low_steps/self.config.physics_hz,
            modal_max_abs_m=self.modal_max if self.config.liquid else None,
            fraction_near_artificial_modal_limit=self.modal_limit_steps/self.steps if self.config.liquid and self.steps else None,
            use="plant-only foot/world contact-object audit (not force-weighted); never learner inputs")
