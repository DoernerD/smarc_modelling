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
        # State weight matrix
        Q_diag = np.ones(self.nx)
        # Position weights reduced from 1000/1000/500 to close the pos/vel cost ratio.
        # Original ratio surge_pos/surge_vel = 1000/15 ≈ 67:1 caused oscillation at the
        # end of trajectories: position pull >> velocity braking at any realistic speed.
        # New ratio ≈ 600/200 = 3:1 — position still dominates during tracking but the
        # MPC can now meaningfully penalise velocity to resist overshoot.
        Q_diag[0] = 600   # x-position  (was 1000)
        Q_diag[1] = 600   # y-position  (was 1000)
        Q_diag[2] = 1000   # z-position  (was  500)
        Q_diag[3:7] = 500  # Quaternion: HIGH penalty for turbo turn - must face correct direction

        # Velocity costs:
        # - Surge (u): raised so the braking constraint and velocity reference together
        #   actually influence the MPC.  With Q_pos=600 and Q_vel_surge=1000, the
        #   velocity cost equals the position cost when v ≈ sqrt(Q_pos/Q_vel)*e_pos
        #   = sqrt(0.6)*e_pos ≈ 0.77*e_pos.  At e_pos=3m → equal at v=2.3 m/s (not
        #   binding for normal tracking), but the hard braking constraint caps speed
        #   much lower, so the velocity weight mainly prevents speeding between waypoints.
        # - Sway (v): uncontrollable, keep minimal
        # - Heave (w): partially controllable, keep moderate
        Q_diag[7] = 1000.0  # surge velocity (u) — was 200 (5× increase)
        Q_diag[8] = 0.01   # sway  velocity (v)  — uncontrollable, unchanged
        Q_diag[9] = 500    # heave velocity (w)  — raised from 5; makes ref[9] a meaningful dive/surface signal
        Q_diag[10] = 1     # p (roll  rate)      — unchanged
        Q_diag[11] = 1     # q (pitch rate)      — unchanged
        Q_diag[12] = 1.0   # r (yaw   rate)      — unchanged

        # Control weight matrix - Costs set according to Bryson's rule
        Q_diag[13] = 1e-5  # VBS:      Standard: 1e-4
        Q_diag[14] = 1e-4  # LCG:      Standard: 1e-4
        Q_diag[15] = 1e2  # stern_angle:   Standard: 100
        Q_diag[16] = 1e2  # rudder_angle: Increased for smoother control (was 1e0)
        Q_diag[17] = 1e-5  # RPM1: increased to discourage bang-bang (was 1e-8)
        Q_diag[18] = 1e-5  # RPM2: increased to discourage bang-bang (was 1e-8)
        Q = np.diag(Q_diag)

        # Control rate of change weight matrix - control inputs as [x_vbs, x_lcg, delta_s, delta_r, rpm1, rpm2]
        # SIM Version (also runs on SAM)
        R_diag = np.ones(self.nu)
        R_diag[0] = 1e-2  # 1e-1        # VBS
        R_diag[1] = 1e-1  # LCG
        R_diag[2] = 1e0     # stern angle
        R_diag[3] = 1e0     # rudder angle
        R_diag[4] = 1e-6  # RPM1 rate: increased to smooth out bang-bang (was 1e-9)
        R_diag[5] = 1e-6  # RPM2 rate: increased to smooth out bang-bang (was 1e-9)
        R = np.diag(R_diag)

        # SAM Tuned
        # R_diag = np.ones(self.nu)
        # R_diag[0] = 1e-2 #1e-1        # VBS
        # R_diag[1] = 1e-1        # LCG
        # R_diag[2] = 5e2
        # R_diag[3] = 5e3
        # R_diag[4: ] = 1e-6
        # R = np.diag(R_diag)*1e-3
        
        # Terminal Costs
        # The terminal node (end of 3 s horizon) is what the MPC uses to plan final
        # stops. Q_e_surge >> Q_e_pos means "be slow at the end of your horizon,
        # even if you haven't quite reached the position target yet."
        # With Q_e_surge = 1500 and Q_e_pos = 600:
        #   equal cost at v = sqrt(600/1500) * e_pos ≈ 0.63 * e_pos
        #   e.g. at 0.5 m from goal, velocity > 0.32 m/s costs more than the position.
        Q_e_diag = np.ones(self.nx)
        Q_e_diag[0] = 600   # x  (was 1000, matches stage)
        Q_e_diag[1] = 600   # y  (was 1000, matches stage)
        Q_e_diag[2] = 300   # z  (was  500, matches stage)
        Q_e_diag[3:7] = 500 # quaternion — unchanged
        Q_e_diag[7] = 3000.0 # surge velocity — was 1500 (2× increase; terminal stop strong)
        Q_e_diag[8] = 0.01  # sway  — unchanged
        Q_e_diag[9] = 500   # heave — raised from 5 (matches stage weight)
        Q_e_diag[10:12] = 1 # roll/pitch rates — unchanged
        Q_e_diag[12] = 10   # yaw rate — unchanged
        Q_e_diag[13:17] = 1e-5 # vbs, lcg, stern, rudder — unchanged
        Q_e_diag[17:19] = 1e-5 # rpm1, rpm2 — unchanged
        Q_e = np.diag(Q_e_diag) # terminal cost

        # Stage costs
        # Parameter vector layout: [state_ref (nx), control_ref (nu), goal_pos (3)]
        # goal_pos = [x_goal, y_goal, z_goal] of the final trajectory waypoint.
        # Used by the braking constraint to compute remaining distance to goal.
        self.model.p = ca.MX.sym("ref_param", self.nx + self.nu + 3, 1)
        self.ocp.parameter_values = np.zeros((self.nx + self.nu + 3,))

        self.ocp.cost.yref = np.zeros(
            (self.nx + self.nu,)
        )  # Init ref point. The true references are declared in the controller for-loop
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
        self.ocp.cost.yref_e = np.zeros((self.nx,))

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
        y_min, y_max = -1.5, 1.5
        z_min, z_max = -0.5, 3.0

        pos_lbx = np.array([x_min, y_min, z_min])
        pos_ubx = np.array([x_max, y_max, z_max])

        # --- velocity constraints
        # Note, these are arbitrary guesses...
        x_dot_min, x_dot_max = -5.0, 5.0
        y_dot_min, y_dot_max = -2.0, 2.0
        z_dot_min, z_dot_max = -2.0, 2.0

        vel_lbx = np.array([x_dot_min, y_dot_min, z_dot_min])
        vel_ubx = np.array([x_dot_max, y_dot_max, z_dot_max])

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
        # a_brake: effective deceleration [m/s^2].
        #   Must be ≤ the vehicle's real worst-case braking capability so the
        #   constraint is always feasible.  Lower = tighter speed ceiling at a given
        #   distance = earlier forced deceleration.  At d metres from the goal the
        #   constraint enforces v ≤ sqrt(2 * a_brake * (d + d_eps)).
        #   Rule of thumb: start at half the observed deceleration, then tune up.
        # d_eps: distance offset so the allowed speed does not collapse to 0 exactly
        #   at the goal (avoids fighting the position cost near the goal).
        #   Should match final_pos_tolerance in the controller (≈ 0.5 m).
        a_brake = 0.001  # m/s^2  — was 0.10; tightened to match real SAM capability
        d_eps   = 0.5   # m      — was 1.5; reduced to match final_pos_tolerance

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

        # ----- RPM Funnel Constraint -------------------------------------------
        # On the real SAM the thrusters have a deadzone of roughly ±rpm_deadzone RPM:
        # commands in that range produce no thrust regardless of direction, so the
        # vehicle coasts even when the MPC thinks it is braking (or accelerating).
        #
        # The funnel magnitude is the same in both directions:
        #
        #   rpm_mag(d) = (rpm_max + rpm_deadzone) * min(d / d_rpm_trigger, 1) - rpm_deadzone
        #
        #   d >= d_rpm_trigger  →  rpm_mag = rpm_max   (constraint inactive)
        #   d = d_rpm_trigger/2 →  rpm_mag ≈ rpm_max/2
        #   d = 0               →  rpm_mag = -rpm_deadzone  (forced past the deadzone)
        #
        # The direction is determined by the sign of the current surge velocity:
        #   surge > 0  (moving forward)  →  upper bound:  rpm <=  rpm_mag
        #   surge < 0  (moving backward) →  lower bound:  rpm >= -rpm_mag
        #   surge ≈ 0                    →  both bounds collapse toward ±rpm_deadzone,
        #                                   keeping RPM outside the deadzone
        #
        # This ensures the thruster is always pushed *through* the ±200 RPM deadzone
        # into active braking territory, regardless of which direction the AUV is
        # approaching from.
        rpm_deadzone  = 200.0   # [RPM] deadzone on real SAM thrusters
        d_rpm_trigger = 2.0     # [m]   distance at which RPM cap starts tightening
        rpm_max_val   = act_ubx[4]  # 450 RPM — matches the state box constraint

        rpm_mag = (rpm_max_val + rpm_deadzone) * ca.fmin(
            dist_to_goal / d_rpm_trigger, 1.0
        ) - rpm_deadzone

        surge = self.model.x[7]  # surge velocity (body-frame x)

        # Upper bound active when moving forward; lower bound active when moving backward.
        # ca.if_else is smooth in CasADi for SQP — the branch is chosen symbolically.
        rpm_upper_cap = ca.if_else(surge >= 0,  rpm_mag,  rpm_max_val)
        rpm_lower_cap = ca.if_else(surge <  0, -rpm_mag, -rpm_max_val)

        # h_upper = rpm - rpm_upper_cap <= 0
        # h_lower = rpm_lower_cap - rpm <= 0  (i.e. rpm >= rpm_lower_cap)
        rpm_h = ca.vertcat(
            self.model.x[17] - rpm_upper_cap,   # rpm1 upper
            self.model.x[18] - rpm_upper_cap,   # rpm2 upper
            rpm_lower_cap - self.model.x[17],   # rpm1 lower
            rpm_lower_cap - self.model.x[18],   # rpm2 lower
        )

        # Stage nonlinear constraint: stack velocity funnel + RPM funnel (4 rpm terms)
        # con_h layout: [brake_h(1), rpm1_upper(1), rpm2_upper(1), rpm1_lower(1), rpm2_lower(1)]
        self.ocp.model.con_h_expr = ca.vertcat(brake_h, rpm_h)
        self.ocp.constraints.lh = np.array([-1e9, -1e9, -1e9, -1e9, -1e9])
        self.ocp.constraints.uh = np.array([0.0,  0.0,  0.0,  0.0,  0.0])
        self.ocp.constraints.idxsh = np.arange(5)  # soften all five
        n_sh = 5

        # Terminal nonlinear constraint (same expressions, evaluated at terminal node)
        self.ocp.model.con_h_expr_e = ca.vertcat(brake_h, rpm_h)
        self.ocp.constraints.lh_e = np.array([-1e9, -1e9, -1e9, -1e9, -1e9])
        self.ocp.constraints.uh_e = np.array([0.0,  0.0,  0.0,  0.0,  0.0])
        self.ocp.constraints.idxsh_e = np.arange(5)
        n_sh_e = 5

        # ----- Unified slack penalty vectors ------------------------------------
        # acados orders slack variables as: [idxsbx | idxsh] for stage costs.
        # Terminal stage only has idxsh_e.
        Z_pos   = 1e3  # quadratic penalty for position box violations
        z_pos   = 1e1  # linear   penalty for position box violations
        # Braking constraint penalties.
        # With Q_pos = 600 and a 3 m position error the position cost is ~5 400.
        # Z_brake must dominate that to make the constraint binding.
        # At Z_brake = 1e5, even a 0.07 m/s violation costs ~500 (10 % of position
        # cost at 3 m), making the constraint effectively hard without numerics blowing up.
        # Increase further if the vehicle still exceeds the speed envelope.
        Z_brake = 1e5  # quadratic penalty — was 1e2 (1 000× increase)
        z_brake = 1e3  # linear   penalty — was 1e1 (100× increase)
        # RPM funnel penalties — large enough to drive RPM through the deadzone
        # but softer than the velocity constraint so the solver has headroom.
        Z_rpm   = 5e4
        z_rpm   = 5e2

        # Stage: [sbx(3), sh_brake(1), sh_rpm(4)] = size 8
        self.ocp.cost.Zl = np.r_[Z_pos * np.ones(n_sb), Z_brake * np.ones(1), Z_rpm * np.ones(4)]
        self.ocp.cost.Zu = np.r_[Z_pos * np.ones(n_sb), Z_brake * np.ones(1), Z_rpm * np.ones(4)]
        self.ocp.cost.zl = np.r_[z_pos * np.ones(n_sb), z_brake * np.ones(1), z_rpm * np.ones(4)]
        self.ocp.cost.zu = np.r_[z_pos * np.ones(n_sb), z_brake * np.ones(1), z_rpm * np.ones(4)]

        # Terminal: [sh_e_brake(1), sh_e_rpm(4)] = size 5
        self.ocp.cost.Zl_e = np.r_[Z_brake * np.ones(1), Z_rpm * np.ones(4)]
        self.ocp.cost.Zu_e = np.r_[Z_brake * np.ones(1), Z_rpm * np.ones(4)]
        self.ocp.cost.zl_e = np.r_[z_brake * np.ones(1), z_rpm * np.ones(4)]
        self.ocp.cost.zu_e = np.r_[z_brake * np.ones(1), z_rpm * np.ones(4)]

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
        # self.ocp.solver_options.nlp_solver_type = 'SQP'
        # self.ocp.solver_options.nlp_solver_type = 'SQP_WITH_FEASIBLE_QP'
        # self.ocp.solver_options.search_direction_mode = 'BYRD_OMOJOKUN'
        # self.ocp.solver_options.allow_direction_mode_switch_to_nominal = False
        self.ocp.solver_options.nlp_solver_max_iter = 1  # 80
        self.ocp.solver_options.tol = (
            1e-6  # NLP tolerance. 1e-6 is default for tolerances
        )
        self.ocp.solver_options.qp_tol = 1e-6  # QP tolerance
        # self.ocp.solver_options.qp_mu0 = 5e-1       # QP initial barrier

        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        # self.ocp.solver_options.regularize_method = 'NO_REGULARIZE'
        self.ocp.solver_options.levenberg_marquardt = 1e-2
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
        Calculates the state deviation.

        :param x: State vector
        :param ref: Reference vector
        :return: error vector
        """
        q1 = ref[3:7]
        q1 = q1 / ca.norm_2(q1)
        q2 = x[3:7]
        # Sice unit quaternion, quaternion inverse is equal to its conjugate
        q_conj = ca.vertcat(q2[0], -q2[1], -q2[2], -q2[3])
        q2 = q_conj / ca.norm_2(q2)

        # q_error = q1 @ q2^-1
        q_w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        q_x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        q_y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        q_z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        q_error = ca.vertcat(q_w, q_x, q_y, q_z)

        # Quaternion double-cover: q and -q represent same orientation.
        # Choose the hemisphere that gives smaller rotation (q_w > 0).
        # CRITICAL FIX: If q_w < 0, we're measuring the LONG way around (>90° error).
        # Flip to the short path: -q represents same orientation but <90° error.
        q_error = ca.if_else(q_w < 0, -q_error, q_error)

        # Make attitude error zero at perfect alignment:
        # q_error = [1, 0, 0, 0] -> [0, 0, 0, 0]
        q_att_error = ca.vertcat(1.0 - q_error[0], q_error[1], q_error[2], q_error[3])

        # NOTE: usually I'd have ref - state, the standard closed loop, i.e.
        # Astroem 2019. Since this error is squared, it should work, too,
        # Liniger 2014 uses it in their vanilla MPC formulation
        # Also, since the error is squared in the cost, it doesn't matter
        pos_error = x[:3] - ref[:3]
        vel_error = x[7:13] - ref[7:13]
        u_error = x[13:19] - ref[13:19]

        # If the error for terminal cost is calculated, don't include delta_u
        if terminal:
            x_error = ca.vertcat(pos_error, q_att_error, vel_error, u_error)
        else:
            x_error = ca.vertcat(
                pos_error, q_att_error, vel_error, u_error, u
            )  # delta_u(u))
        return x_error
