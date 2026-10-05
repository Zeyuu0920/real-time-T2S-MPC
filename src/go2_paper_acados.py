"""Acados transcription of the published Go2 SSI-MPC controller.

Everything in this module follows the experiment protocol in Zhou et al.
(arXiv:2510.15626) except for the requested solver substitution: the paper's
CasADi/IPOPT NLP is replaced by an acados SQP-RTI OCP.  In particular, the
state, input, rigid-body dynamics, Forward-Euler transcription, 20-interval
0.6 s horizon, cost weights, force limits, friction pyramid, and RFF residual
are shared by Nominal and SSI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time

import casadi as cs
import numpy as np
import scipy.linalg
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from src.go2_paper_baseline import (
    AuthorCodeRandomFeatures,
    PAPER_FORCE_REFERENCE_MASS,
    PAPER_FRICTION_COEFFICIENT,
    PAPER_INPUT_COST_DIAGONAL,
    PAPER_MAX_STANCE_FZ,
    PAPER_MIN_STANCE_FZ,
    PAPER_MODEL_MASS,
    PAPER_NODE_DT,
    PAPER_NODES,
    PAPER_RFF_COUNT,
    PAPER_RFF_OUTPUT_DIM,
    PAPER_STATE_COST_DIAGONAL,
    _paper_terms_symbolic,
    paper_force_reference,
)


STATE_DIM = 12
CONTROL_DIM = 12
FOOT_COUNT = 4
# Follow the paper's stated ``N=20`` shooting intervals and 0.6 s horizon.
# The public ROS implementation allocates 20 state nodes (19 controls), which
# is inconsistent with the paper's explicit N * 0.03 = 0.6 s protocol.  This
# Acados comparison intentionally follows the paper because the only requested
# experimental difference is the NLP solver.
PAPER_ACADOS_INTERVALS = PAPER_NODES
PAPER_ACADOS_HORIZON_SECONDS = PAPER_ACADOS_INTERVALS * PAPER_NODE_DT


@dataclass(frozen=True)
class PaperAcadosDynamics:
    model: object
    parameter_dim: int
    alpha_offset: int | None
    realtime_parameter_offset: int | None
    time_embedding_offset: int | None
    residual_output_scale: np.ndarray


@dataclass(frozen=True)
class PaperAcadosResult:
    force: np.ndarray
    states: np.ndarray
    forces: np.ndarray
    solve_ms: float
    end_to_end_ms: float
    status: int
    retry_count: int
    sqp_iterations: int
    qp_iterations: int


def _paper_forward_euler_model(
    *,
    model_name: str,
    random_features: AuthorCodeRandomFeatures | None,
    realtime_residual=None,
    time_embedding_dim: int = 16,
    model_mass: float,
    output_normalization: np.ndarray | None = None,
    realtime_feature_mode: str = "paper15",
) -> PaperAcadosDynamics:
    """Build the paper's 12-state EOM and Forward-Euler transition map."""

    state = cs.MX.sym("paper_state", STATE_DIM)
    force = cs.MX.sym("paper_foot_force", CONTROL_DIM)
    negative_feet = cs.MX.sym("paper_negative_feet", CONTROL_DIM)
    step_duration = cs.MX.sym("paper_step_duration")
    parameter_parts = [negative_feet, step_duration]
    alpha_offset = None
    realtime_parameter_offset = None
    time_embedding_offset = None

    if random_features is not None and realtime_residual is not None:
        raise ValueError("SSI and T2S residual models are mutually exclusive")
    if realtime_feature_mode not in ("paper15", "full_state_control24"):
        raise ValueError("Unknown real-time residual feature mode")
    if realtime_feature_mode != "paper15" and realtime_residual is None:
        raise ValueError("Full-state residual features require a real-time learner")

    # The network predicts a bounded, normalized generalized wrench.  These
    # fixed scales are inherited from the existing real-time Go2 T2S model;
    # they are numerical output units, not tuned MPC cost weights.
    # Diagonal of the published generated inertia at zero roll/pitch.
    inertia_diagonal = np.array(
        [4.4294859e-2, 4.4084322384e-1, 4.6958108284e-1], dtype=float
    )
    residual_output_scale = np.concatenate(
        (
            model_mass * np.array([4.0, 4.0, 5.0]),
            inertia_diagonal * np.array([8.0, 8.0, 6.0]),
        )
    )
    if output_normalization is not None:
        if realtime_residual is None:
            raise ValueError("Output normalization belongs to a learned residual, not Nominal")
        residual_output_scale = np.asarray(output_normalization, dtype=float).reshape(6)
        if not np.isfinite(residual_output_scale).all() or np.any(residual_output_scale <= 0):
            raise ValueError("Residual normalization must have six finite positive entries")

    _, _, feature_jacobian, _ = _paper_terms_symbolic(
        state, negative_feet, model_mass=model_mass
    )
    residual = cs.MX.zeros(PAPER_RFF_OUTPUT_DIM, 1)
    if random_features is not None:
        alpha_parameter = cs.MX.sym(
            "paper_ssi_alpha", PAPER_RFF_OUTPUT_DIM * PAPER_RFF_COUNT
        )
        alpha = cs.reshape(
            alpha_parameter, PAPER_RFF_OUTPUT_DIM, PAPER_RFF_COUNT
        )
        # Match the released controller's feature ordering and scaling:
        # [rpy, world velocity, body omega, J.T @ (u/100)].
        feature_input = cs.vertcat(
            state[3:12], feature_jacobian @ (force / 100.0)
        )
        phi = cs.cos(
            cs.DM(random_features.omega) @ feature_input
            + cs.DM(random_features.phase)
        ) / np.sqrt(PAPER_RFF_COUNT)
        residual = alpha @ phi
        alpha_offset = CONTROL_DIM + 1
        parameter_parts.append(alpha_parameter)
    elif realtime_residual is not None:
        if time_embedding_dim <= 0 or time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be a positive even number")
        embedding = cs.MX.sym("paper_t2s_time_embedding", time_embedding_dim)
        if realtime_feature_mode == "full_state_control24":
            # Complete nominal x/u, in their original coordinates and units.
            # No wrench compression, hidden contact state, or force scaling.
            feature_input = cs.vertcat(state, force, embedding)
        else:
            feature_input = cs.vertcat(
                state[3:12],
                feature_jacobian @ (force / 100.0),
                embedding,
            )
        independent_feature = cs.MX.sym(
            "paper_t2s_network_feature", int(feature_input.shape[0])
        )
        normalized_residual = cs.substitute(
            realtime_residual(independent_feature),
            independent_feature,
            feature_input,
        )
        if int(normalized_residual.shape[0]) != PAPER_RFF_OUTPUT_DIM:
            raise ValueError("Go2 T2S residual must have six outputs")
        residual = cs.DM(residual_output_scale) * normalized_residual
        time_embedding_offset = CONTROL_DIM + 1
        realtime_parameter_offset = time_embedding_offset + time_embedding_dim
        parameter_parts.extend(
            (embedding, realtime_residual.get_sym_params())
        )

    # The paper explicitly specifies Forward Euler.  Evaluate every dynamics
    # term at x_k; the residual feature above is likewise causal at (x_k,u_k).
    mass_matrix, nonlinear, contact_jacobian, q_dot = _paper_terms_symbolic(
        state, negative_feet, model_mass=model_mass
    )
    acceleration = cs.solve(
        mass_matrix,
        contact_jacobian @ force + residual - nonlinear,
    )
    next_position = state[0:6] + step_duration * q_dot
    next_velocity = state[6:12] + step_duration * acceleration
    next_state = cs.vertcat(next_position, next_velocity)

    model = AcadosModel()
    model.name = model_name
    model.x = state
    model.u = force
    model.p = cs.vertcat(*parameter_parts)
    model.disc_dyn_expr = next_state
    return PaperAcadosDynamics(
        model=model,
        parameter_dim=int(model.p.shape[0]),
        alpha_offset=alpha_offset,
        realtime_parameter_offset=realtime_parameter_offset,
        time_embedding_offset=time_embedding_offset,
        residual_output_scale=residual_output_scale,
    )


def _friction_pyramid() -> np.ndarray:
    matrix = np.zeros((4 * FOOT_COUNT, CONTROL_DIM), dtype=float)
    row = 0
    for leg in range(FOOT_COUNT):
        offset = 3 * leg
        matrix[row, [offset, offset + 2]] = [1.0, -PAPER_FRICTION_COEFFICIENT]
        matrix[row + 1, [offset, offset + 2]] = [
            -1.0,
            -PAPER_FRICTION_COEFFICIENT,
        ]
        matrix[row + 2, [offset + 1, offset + 2]] = [
            1.0,
            -PAPER_FRICTION_COEFFICIENT,
        ]
        matrix[row + 3, [offset + 1, offset + 2]] = [
            -1.0,
            -PAPER_FRICTION_COEFFICIENT,
        ]
        row += 4
    return matrix


def paper_contact_force_bounds(
    contact_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Paper force limits, with swing-foot GRFs fixed exactly to zero."""

    contact_mask = np.asarray(contact_mask, dtype=bool).reshape(FOOT_COUNT)
    lower = np.tile(np.array([-1.0e6, -1.0e6, PAPER_MIN_STANCE_FZ]), FOOT_COUNT)
    upper = np.tile(np.array([1.0e6, 1.0e6, PAPER_MAX_STANCE_FZ]), FOOT_COUNT)
    for leg, stance in enumerate(contact_mask):
        if not stance:
            leg_slice = slice(3 * leg, 3 * leg + 3)
            lower[leg_slice] = 0.0
            upper[leg_slice] = 0.0
    return lower, upper


class PaperAcadosMPC:
    """Common paper-protocol Nominal/SSI controller using acados SQP-RTI."""

    def __init__(
        self,
        *,
        project_root: Path,
        random_features: AuthorCodeRandomFeatures | None = None,
        realtime_residual=None,
        time_embedding_dim: int = 16,
        name_suffix: str = "seed42",
        model_mass: float = PAPER_MODEL_MASS,
        nlp_solver_type: str = "SQP_RTI",
        rti_iterations_per_cycle: int = 1,
        output_normalization: np.ndarray | None = None,
        realtime_feature_mode: str = "paper15",
    ) -> None:
        self.random_features = random_features
        self.realtime_residual = realtime_residual
        self.time_embedding_dim = int(time_embedding_dim)
        self.model_mass = float(model_mass)
        if self.model_mass <= 0.0:
            raise ValueError("model_mass must be positive")
        if nlp_solver_type not in {"SQP_RTI", "SQP"}:
            raise ValueError("nlp_solver_type must be SQP_RTI or SQP")
        if rti_iterations_per_cycle <= 0:
            raise ValueError("rti_iterations_per_cycle must be positive")
        if nlp_solver_type != "SQP_RTI" and rti_iterations_per_cycle != 1:
            raise ValueError("RTI refinements are only valid for SQP_RTI")
        self.nlp_solver_type = nlp_solver_type
        self.rti_iterations_per_cycle = int(rti_iterations_per_cycle)
        if random_features is not None and realtime_residual is not None:
            raise ValueError("SSI and T2S residual models are mutually exclusive")
        method = (
            "t2s"
            if realtime_residual is not None
            else "ssi"
            if random_features is not None
            else "nominal"
        )
        safe_suffix = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in name_suffix
        )
        self.dynamics = _paper_forward_euler_model(
            model_name=f"go2_paper_fe_acados_{method}_{safe_suffix}",
            random_features=random_features,
            realtime_residual=realtime_residual,
            time_embedding_dim=self.time_embedding_dim,
            model_mass=self.model_mass,
            output_normalization=output_normalization,
            realtime_feature_mode=realtime_feature_mode,
        )
        self.solver = self._build_solver(Path(project_root))
        self._initialized = False
        self._last_contact_table: np.ndarray | None = None
        self._last_first_element_duration: float | None = None

    @property
    def method(self) -> str:
        if self.realtime_residual is not None:
            return "t2s"
        return "ssi" if self.random_features is not None else "nominal"

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _build_solver(self, project_root: Path) -> AcadosOcpSolver:
        model = self.dynamics.model
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = PAPER_ACADOS_INTERVALS
        ocp.solver_options.tf = PAPER_ACADOS_HORIZON_SECONDS
        ocp.parameter_values = np.zeros(self.dynamics.parameter_dim)
        ocp.parameter_values[CONTROL_DIM] = PAPER_NODE_DT

        ny = STATE_DIM + CONTROL_DIM
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx = np.zeros((ny, STATE_DIM))
        ocp.cost.Vx[:STATE_DIM] = np.eye(STATE_DIM)
        ocp.cost.Vu = np.zeros((ny, CONTROL_DIM))
        ocp.cost.Vu[STATE_DIM:] = np.eye(CONTROL_DIM)
        ocp.cost.Vx_e = np.eye(STATE_DIM)
        # Stage zero contains the fixed current state, whereas the source cost
        # starts at x_1.  Per-stage W matrices are finalized in solve().
        ocp.cost.W = scipy.linalg.block_diag(
            np.zeros((STATE_DIM, STATE_DIM)),
            np.diag(PAPER_INPUT_COST_DIAGONAL),
        )
        ocp.cost.W_e = np.diag(PAPER_STATE_COST_DIAGONAL)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(STATE_DIM)

        ocp.constraints.x0 = np.zeros(STATE_DIM)
        ocp.constraints.idxbu = np.arange(CONTROL_DIM)
        ocp.constraints.lbu = np.tile(
            np.array([-1.0e6, -1.0e6, 0.0]), FOOT_COUNT
        )
        ocp.constraints.ubu = np.tile(
            np.array([1.0e6, 1.0e6, PAPER_MAX_STANCE_FZ]), FOOT_COUNT
        )
        friction = _friction_pyramid()
        ocp.constraints.C = np.zeros((friction.shape[0], STATE_DIM))
        ocp.constraints.D = friction
        ocp.constraints.lg = -1.0e6 * np.ones(friction.shape[0])
        ocp.constraints.ug = np.zeros(friction.shape[0])

        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = self.nlp_solver_type
        ocp.solver_options.nlp_solver_max_iter = (
            1 if self.nlp_solver_type == "SQP_RTI" else 20
        )
        if self.nlp_solver_type == "SQP":
            ocp.solver_options.nlp_solver_tol_stat = 1.0e-4
            ocp.solver_options.nlp_solver_tol_eq = 1.0e-4
            ocp.solver_options.nlp_solver_tol_ineq = 1.0e-4
            ocp.solver_options.nlp_solver_tol_comp = 1.0e-4
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        # HPIPM enforces the stage-wise contact bounds reliably here.  qpOASES
        # reaches its iteration cap after contact switches and can return an
        # infeasible RTI step even though the outer acados status is zero.
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = 5
        ocp.solver_options.qp_solver_iter_max = 100
        ocp.solver_options.hpipm_mode = "ROBUST"
        ocp.solver_options.levenberg_marquardt = 1.0e-6
        ocp.solver_options.print_level = 0
        ocp.code_gen_options.model_external_shared_lib_dir = str(
            Path(os.environ.get("ACADOS_SOURCE_DIR", project_root / "external" / "acados")) / "lib"
        )
        ocp.code_gen_options.model_external_shared_lib_name = "acados"
        export_directory = (
            project_root / "c_generated_code" / model.name
        )
        export_directory.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(export_directory)
        json_path = project_root / f"{model.name}.json"
        solver_library = export_directory / (
            f"libacados_ocp_solver_{model.name}.so"
        )
        reuse_generated_solver = json_path.exists() and solver_library.exists()
        return AcadosOcpSolver(
            ocp,
            json_file=str(json_path),
            generate=not reuse_generated_solver,
            build=not reuse_generated_solver,
        )

    def initialize(
        self,
        state: np.ndarray,
        references: np.ndarray,
        contact_table: np.ndarray,
    ) -> None:
        state = np.asarray(state, dtype=float).reshape(STATE_DIM)
        references = np.asarray(references, dtype=float).reshape(
            STATE_DIM, PAPER_ACADOS_INTERVALS + 1
        )
        contact_table = np.asarray(contact_table, dtype=bool).reshape(
            FOOT_COUNT, PAPER_ACADOS_INTERVALS
        )
        for stage in range(PAPER_ACADOS_INTERVALS):
            self.solver.set(stage, "x", references[:, stage])
            self.solver.set(
                stage,
                "u",
                paper_force_reference(contact_table[:, stage]),
            )
        self.solver.set(PAPER_ACADOS_INTERVALS, "x", references[:, -1])
        self.solver.set(0, "x", state)
        self._initialized = True

    def _shift_warm_start_one_node(self) -> None:
        """Match the public controller's warm-start shift at a new grid node.

        The local planner solves at 200 Hz on a 30 ms prediction grid.  Most
        calls therefore refine the same finite element.  When the 30 ms grid
        advances, the public IPOPT implementation shifts its previous state,
        force, and multiplier guesses forward by one node.  Resetting acados
        at that event discards exactly the trajectory an SQP-RTI step needs.
        """

        states = [
            np.asarray(self.solver.get(stage, "x"), dtype=float).copy()
            for stage in range(PAPER_ACADOS_INTERVALS + 1)
        ]
        controls = [
            np.asarray(self.solver.get(stage, "u"), dtype=float).copy()
            for stage in range(PAPER_ACADOS_INTERVALS)
        ]
        for stage in range(PAPER_ACADOS_INTERVALS):
            self.solver.set(stage, "x", states[stage + 1])
            self.solver.set(
                stage,
                "u",
                controls[min(stage + 1, PAPER_ACADOS_INTERVALS - 1)],
            )
        self.solver.set(PAPER_ACADOS_INTERVALS, "x", states[-1])

    def _repair_contact_warm_start(
        self,
        contact_table: np.ndarray,
        previous_contact_table: np.ndarray,
    ) -> None:
        """Make shifted force guesses feasible at newly changed contacts."""

        for stage in range(PAPER_ACADOS_INTERVALS):
            previous_stage = min(stage + 1, PAPER_ACADOS_INTERVALS - 1)
            if not np.array_equal(
                contact_table[:, stage],
                previous_contact_table[:, previous_stage],
            ):
                self.solver.set(
                    stage,
                    "u",
                    paper_force_reference(contact_table[:, stage]),
                )

    def prepare_warm_start(
        self, initial_state, references, contact_table, *, plan_index: int,
    ) -> None:
        """Prepare ONE shared expansion trajectory before residual Jacobians.

        Opt-in explicit-clock interface. Legacy callers keep the existing
        solve-time shifting path. No cost, model or constraint is changed.
        """
        contact_table = np.asarray(contact_table,dtype=bool).reshape(4,PAPER_ACADOS_INTERVALS)
        if not self._initialized:
            self.initialize(initial_state,references,contact_table)
        else:
            shift = min(max(0,plan_index-getattr(self,"_explicit_plan_index",plan_index)),PAPER_ACADOS_INTERVALS)
            for _ in range(shift):
                self._shift_warm_start_one_node()
            for stage in range(PAPER_ACADOS_INTERVALS):
                previous_stage=min(stage+shift,PAPER_ACADOS_INTERVALS-1)
                if self._last_contact_table is not None and not np.array_equal(
                    contact_table[:,stage],self._last_contact_table[:,previous_stage]):
                    self.solver.set(stage,"u",paper_force_reference(contact_table[:,stage]))
                lower,upper=paper_contact_force_bounds(contact_table[:,stage])
                self.solver.set(stage,"u",np.clip(self.solver.get(stage,"u"),lower,upper))
        self.solver.set(0,"x",np.asarray(initial_state,dtype=float).reshape(STATE_DIM))
        self._explicit_plan_index=plan_index
        self._prepared_plan=(plan_index,contact_table.copy(),np.asarray(initial_state).copy())

    def solve(
        self,
        initial_state: np.ndarray,
        references: np.ndarray,
        negative_foot_positions_body: np.ndarray,
        contact_table: np.ndarray,
        *,
        first_element_duration: float = PAPER_NODE_DT,
        alpha: np.ndarray | None = None,
        time_embeddings: np.ndarray | None = None,
        realtime_parameters: np.ndarray | None = None,
        plan_index: int | None = None,
        warm_start_prepared: bool = False,
        realtime_parameter_refresh=None,
    ) -> PaperAcadosResult:
        cycle_start = time.perf_counter()
        initial_state = np.asarray(initial_state, dtype=float).reshape(STATE_DIM)
        references = np.asarray(references, dtype=float).reshape(
            STATE_DIM, PAPER_ACADOS_INTERVALS + 1
        )
        negative_feet = np.asarray(
            negative_foot_positions_body, dtype=float
        ).reshape(CONTROL_DIM, PAPER_ACADOS_INTERVALS)
        contact_table = np.asarray(contact_table, dtype=bool).reshape(
            FOOT_COUNT, PAPER_ACADOS_INTERVALS
        )
        first_element_duration = float(first_element_duration)
        if not 0.0 < first_element_duration <= PAPER_NODE_DT + 1.0e-9:
            raise ValueError(
                "first_element_duration must lie in (0, PAPER_NODE_DT]"
            )
        if self.random_features is None:
            if alpha is not None:
                raise ValueError("Nominal MPC does not accept SSI parameters")
            alpha_vector = np.empty(0)
        else:
            if alpha is None:
                raise ValueError("SSI-MPC requires the current alpha")
            alpha_vector = np.asarray(alpha, dtype=float).reshape(
                PAPER_RFF_OUTPUT_DIM, PAPER_RFF_COUNT
            ).reshape(-1, order="F")

        if self.realtime_residual is None:
            if time_embeddings is not None or realtime_parameters is not None:
                raise ValueError("Only T2S MPC accepts real-time neural parameters")
            time_embeddings_array = np.empty((PAPER_ACADOS_INTERVALS, 0))
            realtime_parameters_array = np.empty((PAPER_ACADOS_INTERVALS, 0))
        else:
            if time_embeddings is None or realtime_parameters is None:
                raise ValueError("T2S MPC requires embeddings and neural parameters")
            time_embeddings_array = np.asarray(time_embeddings, dtype=float).reshape(
                PAPER_ACADOS_INTERVALS, self.time_embedding_dim
            )
            realtime_parameters_array = np.asarray(
                realtime_parameters, dtype=float
            )
            expected_realtime_dim = (
                self.dynamics.parameter_dim
                - int(self.dynamics.realtime_parameter_offset)
            )
            if realtime_parameters_array.shape != (
                PAPER_ACADOS_INTERVALS,
                expected_realtime_dim,
            ):
                raise ValueError(
                    "realtime_parameters has shape "
                    f"{realtime_parameters_array.shape}, expected "
                    f"({PAPER_ACADOS_INTERVALS}, {expected_realtime_dim})"
                )

        previous_contact_table = self._last_contact_table
        if warm_start_prepared:
            prepared=getattr(self,"_prepared_plan",None)
            if prepared is None or prepared[0]!=plan_index or not np.array_equal(prepared[1],contact_table) or not np.array_equal(prepared[2],initial_state):
                raise ValueError("Prepared expansion trajectory does not match solve request")
            self._prepared_plan=None
        elif not self._initialized:
            self.initialize(initial_state, references, contact_table)
        elif plan_index is not None:
            last_index = getattr(self, "_explicit_plan_index", plan_index)
            for _ in range(min(max(0, plan_index - last_index), PAPER_ACADOS_INTERVALS)):
                self._shift_warm_start_one_node()
            if previous_contact_table is not None:
                self._repair_contact_warm_start(contact_table, previous_contact_table)
        elif (
            self._last_first_element_duration is not None
            and first_element_duration
            > self._last_first_element_duration + 0.5 * PAPER_NODE_DT
        ):
            # first_element_duration counts down from 30 ms to 5 ms, then
            # jumps back to 30 ms when the source plan index increments.
            self._shift_warm_start_one_node()
            if previous_contact_table is not None:
                self._repair_contact_warm_start(
                    contact_table, previous_contact_table
                )
        if plan_index is not None:
            self._explicit_plan_index = plan_index

        # Fix the current state exactly at both bounds.
        self.solver.set(0, "lbx", initial_state)
        self.solver.set(0, "ubx", initial_state)
        for stage in range(PAPER_ACADOS_INTERVALS):
            force_reference = paper_force_reference(contact_table[:, stage])
            yref = np.concatenate((references[:, stage], force_reference))
            self.solver.set(stage, "yref", yref)
            state_weight = np.zeros(STATE_DIM)
            if stage > 0:
                state_weight = PAPER_STATE_COST_DIAGONAL.copy()
                if stage == 1:
                    state_weight *= first_element_duration / PAPER_NODE_DT
            input_weight = PAPER_INPUT_COST_DIAGONAL.copy()
            if stage == 0:
                input_weight *= first_element_duration / PAPER_NODE_DT
            self.solver.cost_set(
                stage,
                "W",
                scipy.linalg.block_diag(
                    np.diag(state_weight), np.diag(input_weight)
                ),
            )
            lower, upper = paper_contact_force_bounds(contact_table[:, stage])
            self.solver.constraints_set(stage, "lbu", lower)
            self.solver.constraints_set(stage, "ubu", upper)
            stage_parameter = np.concatenate(
                (
                    negative_feet[:, stage],
                    np.array(
                        [
                            first_element_duration
                            if stage == 0
                            else PAPER_NODE_DT
                        ]
                    ),
                    alpha_vector,
                    time_embeddings_array[stage],
                    realtime_parameters_array[stage],
                )
            )
            self.solver.set(stage, "p", stage_parameter)
        self.solver.set(
            PAPER_ACADOS_INTERVALS, "yref", references[:, -1]
        )

        solve_start = time.perf_counter()
        status = 0
        for _ in range(self.rti_iterations_per_cycle):
            status = int(self.solver.solve())
            if status != 0:
                break
        retry_count = 0
        if status != 0:
            # A contact-bound change can make the RTI linearization unusable
            # even after projecting the force guess.  The released IPOPT loop
            # simply attempts a fresh solve on the next 200 Hz cycle.  For the
            # one-iteration acados substitute, retry once immediately from the
            # same paper reference and nominal feasible GRFs.
            retry_count = 1
            self.solver.reset()
            self.initialize(initial_state, references, contact_table)
            if realtime_parameter_refresh is not None:
                # A cold retry has a different expansion trajectory too.
                # Reuse the fixed learner snapshot but refresh its Jacobians.
                refreshed=np.asarray(realtime_parameter_refresh(),dtype=float)
                if refreshed.shape != realtime_parameters_array.shape:
                    raise ValueError("Cold-retry residual parameter shape changed")
                for stage in range(PAPER_ACADOS_INTERVALS):
                    self.solver.set(stage,"p",np.concatenate((negative_feet[:,stage],
                        [first_element_duration if stage==0 else PAPER_NODE_DT],
                        alpha_vector,time_embeddings_array[stage],refreshed[stage])))
            for _ in range(self.rti_iterations_per_cycle):
                status = int(self.solver.solve())
                if status != 0:
                    break
        solve_ms = 1000.0 * (time.perf_counter() - solve_start)
        states = np.column_stack(
            [
                self.solver.get(stage, "x")
                for stage in range(PAPER_ACADOS_INTERVALS + 1)
            ]
        )
        forces = np.column_stack(
            [
                self.solver.get(stage, "u")
                for stage in range(PAPER_ACADOS_INTERVALS)
            ]
        )
        self._last_contact_table = contact_table.copy()
        self._last_first_element_duration = first_element_duration
        return PaperAcadosResult(
            force=forces[:, 0].copy(),
            states=states,
            forces=forces,
            solve_ms=solve_ms,
            end_to_end_ms=1000.0 * (time.perf_counter() - cycle_start),
            status=status,
            retry_count=retry_count,
            sqp_iterations=int(self.solver.get_stats("sqp_iter")),
            qp_iterations=int(np.sum(self.solver.get_stats("qp_iter"))),
        )
