import casadi as cs
import numpy as np
import scipy.linalg
import l4casadi as l4c
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel


COST = 'LINEAR_LS'

class MPC:
    def __init__(self, model, N, t_horizon, external_shared_lib_dir, external_shared_lib_name):
        self.model = model
        self.N = N
        self.t_horizon = t_horizon
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property
    def solver(self):
        return AcadosOcpSolver(self.ocp())

    def ocp(self):
        model = self.model

        t_horizon = self.t_horizon
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)
        
        # Dimensions are supplied by the model.  The original repository only
        # used the planar 6-state/2-input quadrotor; inferring them here keeps
        # that case unchanged while allowing the same MPC wrapper to control
        # the full 12-state/4-input vehicle.
        nx = int(model.x.shape[0])
        nu = int(model.u.shape[0])
        ny = nx + nu
        ny_e = nx

        # Create OCP object to formulate the optimization
        ocp = AcadosOcp()
        ocp.model = model_ac
        if hasattr(model, "p") and model.p.shape[0] > 0:
            ocp.dims.np = int(model.p.shape[0])
            ocp.parameter_values = np.zeros((int(model.p.shape[0]),), dtype=np.float64)
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        if COST == 'LINEAR_LS':
            # Initialize cost function
            ocp.cost.cost_type = 'LINEAR_LS'
            ocp.cost.cost_type_e = 'LINEAR_LS'

            # State
            ocp.cost.Vx = np.zeros((ny, nx))
            for i in range(nx):
                ocp.cost.Vx[i, i] = 1 
            # input
            ocp.cost.Vu = np.zeros((ny, nu))
            for i in range(nu):
                ocp.cost.Vu[i + nx, i] = 1
            ocp.cost.Vz = np.array([[]])
            # terminal
            ocp.cost.Vx_e = np.eye(nx)

            l4c_y_expr = None
        else:
            ocp.cost.cost_type = 'NONLINEAR_LS'
            ocp.cost.cost_type_e = 'NONLINEAR_LS'

            x = ocp.model.x
            u = ocp.model.u
            y_expr = cs.vertcat(x, u)

            # Trivial PyTorch index 0
            l4c_y_expr = l4c.L4CasADi(lambda y_expr: y_expr, name='y_expr')

            ocp.model.cost_y_expr = l4c_y_expr(y_expr)
            ocp.model.cost_y_expr_e = x

        # Define weight matrices: a model may provide task-specific weights.
        # The fallback exactly preserves the original planar MPC tuning.
        if hasattr(model, 'cost_Q'):
            Q = np.asarray(model.cost_Q, dtype=float)
        elif nx == 6:
            Q = np.diag([5, 0.1, 5, 0.1, 0.1, 0.1])
        else:
            Q = np.eye(nx)
        if hasattr(model, 'cost_R'):
            R = np.asarray(model.cost_R, dtype=float)
        elif nu == 2:
            R = np.diag([0.1, 0.1])
        else:
            R = 0.1 * np.eye(nu)
        if Q.shape != (nx, nx):
            raise ValueError(f'model.cost_Q must have shape {(nx, nx)}, got {Q.shape}')
        if R.shape != (nu, nu):
            raise ValueError(f'model.cost_R must have shape {(nu, nu)}, got {R.shape}')
        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        ocp.cost.W_e = Q
        ocp.cost.yref = np.zeros((ny,))
        ocp.cost.yref_e = np.zeros((ny_e,))

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start

        # Set constraints
        a_max = model.u_max
        a_min = model.u_min
        ocp.constraints.lbu = a_min
        ocp.constraints.ubu = a_max
        ocp.constraints.idxbu = np.arange(nu)

        # Optional shared state envelope. ``x0`` remains an equality at the
        # measured initial state; these bounds apply to future shooting nodes
        # and the terminal node. Learned and nominal controllers therefore
        # solve the same constrained task.
        if hasattr(model, 'state_bound_indices'):
            state_indices = np.asarray(
                model.state_bound_indices, dtype=int
            ).reshape(-1)
            state_lower = np.asarray(
                model.state_lower_bounds, dtype=float
            ).reshape(-1)
            state_upper = np.asarray(
                model.state_upper_bounds, dtype=float
            ).reshape(-1)
            if not (
                state_indices.size == state_lower.size == state_upper.size
            ):
                raise ValueError(
                    'state-bound indices and lower/upper bounds must have '
                    'the same length'
                )
            if np.any(state_indices < 0) or np.any(state_indices >= nx):
                raise ValueError('state-bound index lies outside the model state')
            if np.any(state_lower >= state_upper):
                raise ValueError(
                    'every state lower bound must be below its upper bound'
                )
            ocp.constraints.idxbx = state_indices
            ocp.constraints.lbx = state_lower
            ocp.constraints.ubx = state_upper
            ocp.constraints.idxbx_e = state_indices
            ocp.constraints.lbx_e = state_lower
            ocp.constraints.ubx_e = state_upper

        # Solver options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        if COST == 'LINEAR_LS':
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name
        else:
            ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr.name
     
        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.name = model.name
        return model_ac
