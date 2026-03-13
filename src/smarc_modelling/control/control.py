# Script for the acados NMPC model
import os
from typing import Optional

import casadi as ca
import numpy as np
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
)


# The original NMPC class. Uses hard constraints.
class NMPC:
    def __init__(self, casadi_model, Ts, N_horizon, update_solver_settings):
        """
        :param casadi_model: The casadi model to be used
        :param Ts: Sampling interval
        :param N_horizon: Control horizon
        :param update_solver_settings: If True, the solver will be updated with the new settings.
        """
        self.ocp = AcadosOcp()
        self.model = self.export_dynamics_model(casadi_model)
        self.ocp.model = self.model
        self.nx = self.model.x.rows()
        self.nu = self.model.u.rows()
        self.Ts = Ts
        self.Tf = Ts * N_horizon
        self.N_horizon = N_horizon
        self.update_solver = update_solver_settings

        # --------------------------- Cost setup ---------------------------------
        # Stage state weight matrix — position + quaternion + surge velocity.
        # Quaternion prevents unnecessary rudder actuation on straight segments
        # by penalising heading deviation from the reference.  Surge velocity
        # provides a direct gradient on the thruster chain, preventing the
        # solver from getting stuck at zero RPM.
        Q_diag = np.ones(8)
        Q_diag[0] = 1000  # x-position
        Q_diag[1] = 1000  # y-position
        Q_diag[2] = 1000  # z-position
        Q_diag[3:7] = 500  # quaternion (heading stability)
        Q_diag[7] = 200   # surge velocity
        Q = np.diag(Q_diag)

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        # SIM Version (also runs on SAM)
        R_diag = np.ones(self.nu)
        R_diag[0] = 1e-2  # 1e-1        # VBS
        R_diag[1] = 1e-1  # LCG
        R_diag[2] = 1e2     # stern angle rate
        R_diag[3] = 1e2     # rudder angle rate
        R_diag[4] = 1e-8  # RPM1 rate: reduced for faster thrust switching during maneuvers
        R_diag[5] = 1e-8  # RPM2 rate: reduced for faster thrust switching during maneuvers
        R = np.diag(R_diag)

        # SAM Tuned
        # R_diag = np.ones(self.nu)
        # R_diag[0] = 1e-2 #1e-1        # VBS
        # R_diag[1] = 1e-1        # LCG
        # R_diag[2] = 5e2
        # R_diag[3] = 5e3
        # R_diag[4: ] = 1e-6
        # R = np.diag(R_diag)*1e-3
        
        # Terminal cost — position + quaternion + velocity (drive to zero at goal).
        # Velocity reference at the terminal node is 0, so Q_e penalises any
        # remaining speed at the end of the horizon.  This, together with the
        # braking/speed-funnel constraint, ensures the vehicle decelerates.
        Q_e_diag = np.ones(13)
        Q_e_diag[0] = 800    # x
        Q_e_diag[1] = 800    # y
        Q_e_diag[2] = 300    # z
        Q_e_diag[3:7] = 1    # quaternion
        Q_e_diag[7] = 3000.0 # surge velocity (u) -> 0
        Q_e_diag[8] = 0.01   # sway  velocity (v)
        Q_e_diag[9] = 500    # heave velocity (w)
        Q_e_diag[10] = 1     # p (roll rate)
        Q_e_diag[11] = 1     # q (pitch rate)
        Q_e_diag[12] = 10    # r (yaw rate)
        Q_e = np.diag(Q_e_diag)

        # Stage costs
        # Parameter vector layout: [state_ref (nx), control_ref (nu), goal_pos (3)]
        # goal_pos = [x_goal, y_goal, z_goal] of the final trajectory waypoint.
        # Used by the braking constraint to compute remaining distance to goal.
        self.model.p = ca.MX.sym("ref_param", self.nx + self.nu + 3, 1)
        self.ocp.parameter_values = np.zeros((self.nx + self.nu + 3,))

        self.n_stage_cost = 8 + self.nu   # pos(3) + quat(4) + surge_vel(1) + rates(6) = 14
        self.n_terminal_cost = 13          # pos(3) + quat(4) + vel(6)  = 13

        self.ocp.cost.yref = np.zeros((self.n_stage_cost,))
        self.ocp.cost.cost_type = "NONLINEAR_LS"
        self.ocp.cost.W = ca.diagcat(Q, R).full()
        self.ocp.model.cost_y_expr = self.x_error(
            self.model.x, self.model.u, self.model.p, terminal=False
        )

        # Terminal cost
        self.ocp.cost.cost_type_e = "NONLINEAR_LS"
        self.ocp.cost.W_e = Q_e
        self.ocp.model.cost_y_expr_e = self.x_error(
            self.model.x, self.model.u, self.ocp.model.p, terminal=True
        )
        self.ocp.cost.yref_e = np.zeros((self.n_terminal_cost,))

        # --------------------- Constraint Setup --------------------------
        vbs_dot = 200  # Maximum rate of change for the VBS
        lcg_dot = 50  # Maximum rate of change for the LCG
        tv_dot = 0.2  # Maximum rate of change for the thrust vectoring

        # Declare initial state
        self.ocp.constraints.x0 = np.zeros(
            (self.nx,)
        )  # Initial state is zero. This is set in the sim. for-loop

        # Set constraints on the control rate of change
        self.ocp.constraints.lbu = np.array([-vbs_dot, -lcg_dot, -tv_dot, -tv_dot])
        self.ocp.constraints.ubu = np.array([vbs_dot, lcg_dot, tv_dot, tv_dot])
        self.ocp.constraints.idxbu = np.arange(4)

        # --- position bounds (NED: z positive down) ---
        # Tank limits in meters
        x_min, x_max = 0.0, 8.0
        y_min, y_max = -2.0, 2.0
        z_min, z_max = -0.5, 3.0

        pos_lbx = np.array([x_min, y_min, z_min])
        pos_ubx = np.array([x_max, y_max, z_max])

        # --- actuator state bounds for x[13:19] = [x_vbs, x_lcg, δs, δr, rpm1, rpm2] ---
        act_lbx = np.array([0.0, 0.0, -np.deg2rad(7), -np.deg2rad(7), -500.0, -500.0])
        act_ubx = np.array([100.0, 100.0, np.deg2rad(7), np.deg2rad(7), 450.0, 450.0])

        ## Hard Constraints
        idxbx = np.r_[[0, 1, 2], [13, 14, 15, 16, 17, 18]]  # 9 indices total
        lbx = np.r_[pos_lbx, act_lbx]  # length 9
        ubx = np.r_[pos_ubx, act_ubx]  # length 9

        self.ocp.constraints.idxbx = idxbx
        self.ocp.constraints.lbx = lbx
        self.ocp.constraints.ubx = ubx

        ## Soft Constraints on position box bounds
        idxsbx = np.array([0, 1, 2])  # soften x, y, z position bounds
        self.ocp.constraints.idxsbx = idxsbx
        n_sb = idxsbx.size  # 3

        # ----- Braking / Speed-Funnel Constraint ----------------------------------
        # Ensures surge velocity is low enough that the vehicle can brake to a stop
        # within the remaining distance to the goal waypoint.
        #
        # Constraint: v_surge^2 - 2 * a_brake * (dist_to_goal + d_eps) <= 0
        #   <=>  v_surge <= sqrt(2 * a_brake * (dist_to_goal + d_eps))
        #
        # The controller sets goal_pos to an arc-length-adjusted virtual goal so
        # that dist_to_goal reflects the remaining PATH distance, not the Euclidean
        # shortcut.  This prevents premature braking on curved trajectories.
        #
        # a_brake controls how early the funnel bites.  With a_brake = 0.1 the
        # speed ceiling at typical distances is well above SAM's cruise speed:
        #   0.5 m → 0.39 m/s,  1 m → 0.49 m/s,  2 m → 0.63 m/s,  4 m → 0.87 m/s
        # The funnel only meaningfully limits speed within ~0.5 m of the goal,
        # which is what we want for trajectory following with end-stop braking.
        a_brake = 0.1   # m/s^2  (was 0.005 — 20× increase for trajectory following)
        d_eps   = 0.75  # m      (was 0.5 — matches final_pos_tolerance)

        x_goal = self.model.p[self.nx + self.nu + 0]
        y_goal = self.model.p[self.nx + self.nu + 1]
        z_goal = self.model.p[self.nx + self.nu + 2]
        dist_to_goal = ca.sqrt(
            (self.model.x[0] - x_goal) ** 2
            + (self.model.x[1] - y_goal) ** 2
            + (self.model.x[2] - z_goal) ** 2
            + 1e-4  # numerical safety to avoid sqrt(0)
        )
        # h(x) = v_surge^2 - 2*a_brake*(d+d_eps) <= 0  (upper bound = 0)
        brake_h = self.model.x[7] ** 2 - 2.0 * a_brake * (dist_to_goal + d_eps)

        # ----- RPM Deadzone Avoidance (distance-independent) ----------------------
        # SAM thrusters have a ±200 RPM deadzone where no thrust is produced.
        #
        # Complementarity constraint: h = t²(1-t²) ≤ 0, where t = rpm/deadzone.
        # Three penalty-free operating points:
        #   rpm = 0    (h = 0, no thrust intended)
        #   |rpm| = D  (h = 0, at the deadzone boundary)
        #   |rpm| > D  (h < 0, producing thrust)
        # Violated only for 0 < |rpm| < D (in the deadzone, where the MPC
        # expects thrust but the real hardware produces none).
        #
        # Gradient behaviour in the deadzone:
        #   0 < |rpm| < D/√2 ≈ 141:  gradient pushes toward rpm=0
        #   D/√2 < |rpm| < D = 200:  gradient pushes toward |rpm|=D
        # This means the solver can freely transit through rpm=0 during
        # direction switches (no penalty barrier at zero), while the upper
        # half of the deadzone still gets pushed past the boundary.
        #
        # The vehicle dynamics model is NOT modified — the full thrust gradient
        # is preserved for fast SQP_RTI convergence.
        rpm_dz = 200.0
        t1 = self.model.x[17] / rpm_dz
        t2 = self.model.x[18] / rpm_dz
        h_dz1 = t1**2 * (1.0 - t1**2)
        h_dz2 = t2**2 * (1.0 - t2**2)

        # con_h layout: [brake_h(1), h_dz1(1), h_dz2(1)]
        self.ocp.model.con_h_expr = ca.vertcat(brake_h, h_dz1, h_dz2)
        self.ocp.constraints.lh = np.array([-1e9, -1e9, -1e9])
        self.ocp.constraints.uh = np.array([ 0.0,  0.0,  0.0])
        self.ocp.constraints.idxsh = np.arange(3)
        n_sh = 3

        # Terminal nonlinear constraint (same structure)
        self.ocp.model.con_h_expr_e = ca.vertcat(brake_h, h_dz1, h_dz2)
        self.ocp.constraints.lh_e = np.array([-1e9, -1e9, -1e9])
        self.ocp.constraints.uh_e = np.array([ 0.0,  0.0,  0.0])
        self.ocp.constraints.idxsh_e = np.arange(3)
        n_sh_e = 3

        # ----- Unified slack penalty vectors ------------------------------------
        # acados orders slack variables as: [idxsbx | idxsh] for stage costs.
        # Terminal stage only has idxsh_e.
        Z_pos   = 1e6   # quadratic penalty for position box violations (hard wall)
        z_pos   = 1e4   # linear   penalty for position box violations
        Z_brake = 1e3   # quadratic penalty for braking funnel violations
        z_brake = 1e1   # linear   penalty for braking funnel violations
        Z_dz    = 200.0 # quadratic penalty for RPM deadzone
        z_dz    = 20.0  # linear   penalty for RPM deadzone
        # Complementarity penalty sizing: max violation h=0.25 at |rpm|=141,
        # giving peak penalty = Z_dz*0.0625 + z_dz*0.25 = 17.5.
        # Position walls: 0.05m violation costs 1e6*0.0025 + 1e4*0.05 = 3000
        # per stage — overwhelms any RPM cost (450 at 300 RPM reference).

        # Stage: [sbx(3), sh_brake(1), sh_dz(2)] = size 6
        self.ocp.cost.Zl = np.r_[Z_pos * np.ones(n_sb), Z_brake, Z_dz, Z_dz]
        self.ocp.cost.Zu = np.r_[Z_pos * np.ones(n_sb), Z_brake, Z_dz, Z_dz]
        self.ocp.cost.zl = np.r_[z_pos * np.ones(n_sb), z_brake, z_dz, z_dz]
        self.ocp.cost.zu = np.r_[z_pos * np.ones(n_sb), z_brake, z_dz, z_dz]

        # Terminal: [sh_e_brake(1), sh_e_dz(2)] = size 3
        self.ocp.cost.Zl_e = np.r_[Z_brake, Z_dz, Z_dz]
        self.ocp.cost.Zu_e = np.r_[Z_brake, Z_dz, Z_dz]
        self.ocp.cost.zl_e = np.r_[z_brake, z_dz, z_dz]
        self.ocp.cost.zu_e = np.r_[z_brake, z_dz, z_dz]

        # ----------------------- Solver Setup --------------------------
        # set prediction horizon
        self.ocp.solver_options.N_horizon = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        self.ocp.solver_options.hpipm_mode = "ROBUST"
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.sim_method_newton_iter = 2  # 3 default

        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.nlp_solver_max_iter = 5
        self.ocp.solver_options.tol = (
            1e-6  # NLP tolerance. 1e-6 is default for tolerances
        )
        self.ocp.solver_options.qp_tol = 1e-6  # QP tolerance
        # self.ocp.solver_options.qp_mu0 = 5e-1       # QP initial barrier

        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        # self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'
        self.ocp.solver_options.levenberg_marquardt = 1e-4
        # self.ocp.solver_options.regularize_method = 'PROJECT'

        # Simulation object based on OCP model.
        self.sim = AcadosSim()
        self.sim.model = self.model
        self.sim.parameter_values = np.zeros(self.nx + self.nu + 3)

        self.sim.solver_options.T = 0.1
        self.sim.solver_options.integrator_type = "ERK"

    # Function to create a Acados model from the casadi model
    def export_dynamics_model(self, casadi_model):
        # Create symbolic state and control variables
        x_sym = ca.MX.sym("x", 19, 1)
        u_sym = ca.MX.sym("u_sym", 6, 1)

        # Create symbolic derivative
        x_dot_sym = ca.MX.sym("x_dot", 19, 1)

        # Set up acados model
        model = AcadosModel()
        model.name = "SAM_equation_system"
        model.x = x_sym
        model.xdot = x_dot_sym
        model.u = u_sym

        # Declaration of explicit and implicit expressions
        x_dot = casadi_model.dynamics(export=True)  # extract casadi.MX function
        f_expl = ca.vertcat(x_dot(x_sym[:13], x_sym[13:]), u_sym)
        f_impl = x_dot_sym - f_expl
        model.f_expl_expr = f_expl
        model.f_impl_expr = f_impl

        return model

    def setup(self):
        """
        Acados setup function for the MPC. Everything is already definied in
        the init, since it's shared with the path planner MPC.
        """
        print("\033[92mNMPC setup is running\033[0m")

        # Define the folder path for the .json and c_generated code inside the home directory
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "acados_generated_code")
        self.ocp.code_export_directory = save_dir

        # Make sure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Setup the solver
        solver_json = os.path.join(save_dir, "acados_ocp_" + self.model.name + ".json")

        acados_ocp_solver = AcadosOcpSolver(
            self.ocp,
            json_file=solver_json,
            generate=self.update_solver,
            build=self.update_solver,
        )

        sim_json = os.path.join(save_dir, "acados_sim_" + self.model.name + ".json")

        acados_integrator = AcadosSimSolver(
            self.sim,
            json_file=sim_json,
            generate=self.update_solver,
            build=self.update_solver,
        )

        return acados_ocp_solver, acados_integrator

    def setup_path_planner(self, map_instance):
        """
        Acados setup function for the path planner MPC. Most is already definied in
        the init, since it's shared with the regular MPC. Some additional
        constraints for the planner because it needs a longer trajectory at the
        end to work.
        """

        ## --------------------- Constraint Setup --------------------------
        # vbs_dot = 10    # Maximum rate of change for the VBS
        # lcg_dot = 15    # Maximum rate of change for the LCG
        # ds_dot  = 7     # Maximum rate of change for stern angle
        # dr_dot  = 7     # Maximum rate of change for rudder angle
        # rpm_dot = 1000  # Maximum rate of change for rpm

        ## Declare initial state
        # self.ocp.constraints.x0 = x0

        ## Set constraints on the control rate of change
        # self.ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot, -ds_dot, -dr_dot, -rpm_dot, -rpm_dot])
        # self.ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot,  ds_dot,  dr_dot,  rpm_dot,  rpm_dot])
        # self.ocp.constraints.idxbu = np.arange(nu)

        # Set constraint x in XFREE
        pointA = self.compute_trajectory_ends(self.model.x, forward=True)  ## CHANGE
        pointB = self.compute_trajectory_ends(self.model.x, forward=False)
        goal_constraints_pointA = ca.vertcat(pointA[0], pointA[1], pointA[2])
        constraints_point_B = ca.vertcat(pointB[0], pointB[1], pointB[2])
        bound = 0.1
        xMax = map_instance["x_max"] - bound
        yMax = map_instance["y_max"] - bound
        zMax = map_instance["z_max"] - bound
        xMin = map_instance["x_min"] + bound
        yMin = map_instance["y_min"] + bound
        zMin = map_instance["z_min"] + bound
        self.ocp.model.con_h_expr = ca.vertcat(
            goal_constraints_pointA, constraints_point_B
        )

        self.ocp.constraints.lh = np.array([xMin, yMin, zMin, xMin, yMin, zMin])
        self.ocp.constraints.uh = np.array([xMax, yMax, zMax, xMax, yMax, zMax])

        ## Set constraints on the states
        # x_ubx = np.ones(nx)
        # x_ubx[  :13] = 400

        ## Set constraints on the control
        # x_ubx[13:15] = 100
        # x_ubx[15:17] = np.deg2rad(7)
        # x_ubx[17:  ] = 400

        # x_lbx = -x_ubx
        # x_lbx[13:15] = 0

        # self.ocp.constraints.lbx = x_lbx
        # self.ocp.constraints.ubx = x_ubx
        # self.ocp.constraints.idxbx = np.arange(nx)
        # self.ocp.constraints.lbx_e = x_lbx
        # self.ocp.constraints.ubx_e = x_ubx
        # self.ocp.constraints.idxbx_e = np.arange(nx)

        # Define the folder path for the .json and c_generated code inside the home directory
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "acados_generated_code")
        self.ocp.code_export_directory = save_dir

        # Make sure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Setup the solver
        solver_json = os.path.join(
            save_dir, "acados_path_ocp_" + self.model.name + ".json"
        )

        acados_ocp_solver = AcadosOcpSolver(
            self.ocp,
            json_file=solver_json,
            generate=self.update_solver,
            build=self.update_solver,
        )

        sim_json = os.path.join(
            save_dir, "acados_path_sim_" + self.model.name + ".json"
        )

        acados_integrator = AcadosSimSolver(
            self.sim,
            json_file=sim_json,
            generate=self.update_solver,
            build=self.update_solver,
        )

        return acados_ocp_solver, acados_integrator

        # ----------------------- Solver Setup --------------------------
        # set prediction horizon
        # self.ocp.solver_options.N_horizon = self.N_horizon
        # self.ocp.solver_options.tf = self.Tf

        # self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        # self.ocp.solver_options.hpipm_mode = 'ROBUST'
        # self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        # self.ocp.solver_options.integrator_type = 'IRK'
        # self.ocp.solver_options.sim_method_newton_iter = 3 #3 default

        # self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        # self.ocp.solver_options.nlp_solver_max_iter = 80
        # self.ocp.solver_options.tol    = 1e-6       # NLP tolerance. 1e-6 is default for tolerances
        # self.ocp.solver_options.qp_tol = 1e-6       # QP tolerance

        # self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        # self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'

        # solver_json = 'acados_ocp_' + self.model.name + '.json'

        ## Set directory for code generation
        # this_file_dir = os.path.dirname(os.path.abspath(__file__))
        ##root_files_dir = '/home/parallels/Desktop/smarc_modelling-master/src/smarc_modelling/motion_planning/MotionPrimitives'
        ##package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        # package_root = os.path.abspath(this_file_dir)
        # codegen_dir = os.path.join(package_root, 'optimization_double_mpc')
        # ocp_dir = os.path.join(codegen_dir, 'acados_ocp_')
        # os.makedirs(codegen_dir, exist_ok=True)
        # self.ocp.code_export_directory = codegen_dir
        # print(f"ext package acados dir: {codegen_dir}")

        ##acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = solver_json, generate=False, build=False)
        # acados_ocp_solver = AcadosOcpSolver(self.ocp, json_file = ocp_dir + self.model.name + '.json', generate=True, build=True)

        ## create an integrator with the same settings as used in the OCP solver.
        ##acados_integrator = AcadosSimSolver(self.ocp, json_file = solver_json)
        # acados_integrator = AcadosSimSolver(self.ocp, json_file = ocp_dir + self.model.name + '.json')

        # return acados_ocp_solver, acados_integrator

    def compute_trajectory_ends(
        self, state, distance=0.655, forward: Optional[bool] = True
    ):
        """
        Compute the point forward along the vehicle's longitudinal axis using CasADi.
        """
        # Get current state elements
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]

        # Normalize quaternion
        norm_q = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= norm_q
        q1 /= norm_q
        q2 /= norm_q
        q3 /= norm_q

        # Forward direction in body frame (longitudinal axis)
        forward_body = ca.vertcat(1, 0, 0)  # X-axis in body frame

        # Rotation matrix from quaternion
        R = ca.vertcat(
            ca.horzcat(
                1 - 2 * (q2**2 + q3**2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ),
            ca.horzcat(
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1**2 + q3**2),
                2 * (q2 * q3 - q0 * q1),
            ),
            ca.horzcat(
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1**2 + q2**2),
            ),
        )

        # Transform to world frame
        forward_world = R @ forward_body

        # Normalize forward vector
        forward_norm = np.sqrt(
            forward_world[0] ** 2 + forward_world[1] ** 2 + forward_world[2] ** 2
        )
        forward_world /= forward_norm

        # Compute new point
        if forward:
            new_point = ca.vertcat(x, y, z) + distance * forward_world
        else:
            new_point = ca.vertcat(x, y, z) - distance * forward_world

        return new_point

    # Create an OCP object
    def setup_double_tree_ocp(self, map_instance):

        # self.ocp.cost.yref_e = x_last
        self.ocp.model.cost_y_expr_e = self.model.x
        # Constraints
        # self.ocp.constraints.x0 = x0

        # Goal constraint for front of SAM

        # pointA = self.compute_A_point_forward_casadi(self.model.x)
        # pointB = self.compute_B_point_backward_casadi(self.model.x)
        pointA = self.compute_trajectory_ends(self.model.x, forward=True)  ## CHANGE
        pointB = self.compute_trajectory_ends(self.model.x, forward=False)
        goal_constraints_pointA = ca.vertcat(pointA[0], pointA[1], pointA[2])
        constraints_point_B = ca.vertcat(pointB[0], pointB[1], pointB[2])

        # Constraint: x in XFREE
        bound = 0.1
        xMax = map_instance["x_max"] - bound
        yMax = map_instance["y_max"] - bound
        zMax = map_instance["z_max"] - bound
        xMin = map_instance["x_min"] + bound
        yMin = map_instance["y_min"] + bound
        zMin = map_instance["z_min"] + bound

        self.ocp.model.con_h_expr = ca.vertcat(
            goal_constraints_pointA, constraints_point_B
        )
        self.ocp.constraints.lh = np.array([xMin, yMin, zMin, xMin, yMin, zMin])
        self.ocp.constraints.uh = np.array([xMax, yMax, zMax, xMax, yMax, zMax])

        ## Set constraints on the rate of change of inputs
        # vbs_dot = 10    # Maximum rate of change for the VBS
        # lcg_dot = 15    # Maximum rate of change for the LCG
        # ds_dot  = 7     # Maximum rate of change for stern angle
        # dr_dot  = 7     # Maximum rate of change for rudder angle
        # rpm_dot = 1000  # Maximum rate of change for rpm
        # ocp.constraints.lbu = np.array([-vbs_dot,-lcg_dot, -ds_dot, -dr_dot, -rpm_dot, -rpm_dot])
        # ocp.constraints.ubu = np.array([ vbs_dot, lcg_dot,  ds_dot,  dr_dot,  rpm_dot,  rpm_dot])
        # ocp.constraints.idxbu = np.arange(nu)

        ## Set constraints on the states
        # x_ubx = np.ones(nx)
        # x_ubx[  :13] = 1000

        ## Set bounds on the state and inputs
        # x_ubx[13:15] = 100
        # x_ubx[15:17] = np.deg2rad(7)
        # x_ubx[17:  ] = 1300
        # x_lbx = -x_ubx
        # x_lbx[13:15] = 0
        # ocp.constraints.lbx = x_lbx
        # ocp.constraints.ubx = x_ubx
        # ocp.constraints.idxbx = np.arange(nx)

        # Set constraints on the final state
        # self.ocp.constraints.lbx_e = x_lbx
        # self.ocp.constraints.ubx_e = x_ubx
        # self.ocp.constraints.idxbx_e = np.arange(nx)

        # Solver setup
        # Set directory for code generation
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        # root_files_dir = '/home/parallels/Desktop/smarc_modelling-master/src/smarc_modelling/motion_planning/MotionPrimitives'
        # package_root = os.path.abspath(os.path.join(this_file_dir, '..'))
        package_root = os.path.abspath(this_file_dir)
        codegen_dir = os.path.join(package_root, "optimization_double_connection")
        ocp_dir = os.path.join(codegen_dir, "acados_ocp.json")
        os.makedirs(codegen_dir, exist_ok=True)
        self.ocp.code_export_directory = codegen_dir
        print(f"ext package acados dir: {codegen_dir}")

        # Solve Acados (For compiling, change both flags to true)
        ocp_solver = AcadosOcpSolver(
            self.ocp,
            json_file=ocp_dir,
            generate=self.update_solver,
            build=self.update_solver,
        )

        return ocp_solver

    def compute_A_point_forward_casadi(self, state, distance=0.655):
        """
        Compute the point forward along the vehicle's longitudinal axis using CasADi.
        """
        # Get current state elements
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]

        # Normalize quaternion
        norm_q = ca.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= norm_q
        q1 /= norm_q
        q2 /= norm_q
        q3 /= norm_q

        # Forward direction in body frame (longitudinal axis)
        forward_body = ca.vertcat(1, 0, 0)  # X-axis in body frame

        # Rotation matrix from quaternion
        R = ca.vertcat(
            ca.horzcat(
                1 - 2 * (q2**2 + q3**2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ),
            ca.horzcat(
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1**2 + q3**2),
                2 * (q2 * q3 - q0 * q1),
            ),
            ca.horzcat(
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1**2 + q2**2),
            ),
        )

        # Transform to world frame
        forward_world = R @ forward_body

        # Normalize forward vector
        forward_norm = sqrt(
            forward_world[0] ** 2 + forward_world[1] ** 2 + forward_world[2] ** 2
        )
        forward_world /= forward_norm

        # Compute new point
        new_point = vertcat(x, y, z) + distance * forward_world

        return new_point

    def compute_B_point_backward_casadi(self, state, distance=0.655):
        """
        Compute the point backward along the vehicle's longitudinal axis using CasADi.
        """
        # Get current state elements
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]

        # Normalize quaternion
        norm_q = sqrt(q0**2 + q1**2 + q2**2 + q3**2)
        q0 /= norm_q
        q1 /= norm_q
        q2 /= norm_q
        q3 /= norm_q

        # Forward direction in body frame (longitudinal axis)
        forward_body = vertcat(1, 0, 0)  # X-axis in body frame

        # Rotation matrix from quaternion
        R = vertcat(
            horzcat(
                1 - 2 * (q2**2 + q3**2),
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ),
            horzcat(
                2 * (q1 * q2 + q0 * q3),
                1 - 2 * (q1**2 + q3**2),
                2 * (q2 * q3 - q0 * q1),
            ),
            horzcat(
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                1 - 2 * (q1**2 + q2**2),
            ),
        )

        # Transform to world frame
        forward_world = R @ forward_body

        # Normalize forward vector
        forward_norm = sqrt(
            forward_world[0] ** 2 + forward_world[1] ** 2 + forward_world[2] ** 2
        )
        forward_world /= forward_norm

        # Compute new point (backward)
        new_point = vertcat(x, y, z) - distance * forward_world

        return new_point

    def x_error(self, x, u, ref, terminal):
        """
        Calculates the cost residual vector.

        Stage  (terminal=False): [pos_error(3), q_att_error(4), surge_vel_error(1), u(6)] = 14
            Position, quaternion, and surge velocity are tracked.  Quaternion
            keeps the heading aligned with the reference, preventing unnecessary
            rudder actuation.  Surge velocity gives the solver a direct gradient
            to engage thrusters.  The rate-of-change u is penalised via R to
            smooth actuation.

        Terminal (terminal=True): [pos_error(3), q_att_error(4), vel_error(6)] = 13
            Full velocity is included at the terminal node to drive the vehicle
            to a stop at the correct heading.
        """
        pos_error = x[:3] - ref[:3]

        q1 = ref[3:7]
        q1 = q1 / ca.norm_2(q1)
        q2 = x[3:7]
        q_conj = ca.vertcat(q2[0], -q2[1], -q2[2], -q2[3])
        q2 = q_conj / ca.norm_2(q2)

        q_w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q_x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q_y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q_z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        q_error = ca.vertcat(q_w, q_x, q_y, q_z)
        q_error = ca.if_else(q_w < 0, -q_error, q_error)
        q_att_error = ca.vertcat(1.0 - q_error[0], q_error[1], q_error[2], q_error[3])

        if terminal:
            vel_error = x[7:13] - ref[7:13]
            return ca.vertcat(pos_error, q_att_error, vel_error)
        else:
            surge_vel_error = x[7] - ref[7]
            return ca.vertcat(pos_error, q_att_error, surge_vel_error, u)
