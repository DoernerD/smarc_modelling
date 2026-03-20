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

        # ========================= MPCC Cost Setup ================================
        # Stage cost: contour/lag errors from a locally-linearized path, heading
        # alignment with the path tangent, and progress-speed tracking via yref.
        # Terminal cost: position + quaternion + velocity tracking (unchanged).
        #
        # Residual layout
        #   Stage:    [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1), u_phys(6), delta_v_theta(1)] = 15
        #   Terminal: [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1), heave_vel(1)] = 9

        # Stage Q: contour(3) + lag(1) + heading(1) + v_theta(1) = 6
        #Q_diag = np.array([1000.0, 1000.0, 1000.0,   # contour (cross-track)
        #                   500.0,                      # lag (along-track)
        #                   200.0,                      # heading alignment with path tangent
        #                   10.0])                      # v_theta (progress pull)
        #Q = np.diag(Q_diag)

        ## Stage R: penalise physical actuator rates only (u[0:6]).
        ## v_theta (u[6]) is NOT included — its cost comes from Q_vt via yref.
        #R_diag = np.array([1e-2,   # VBS rate
        #                   1e-1,   # LCG rate
        #                   1e2,    # stern angle rate
        #                   1e2,    # rudder angle rate
        #                   1e-8,   # RPM1 rate
        #                   1e-8])  # RPM2 rate
        #R = np.diag(R_diag)
        
        
        Q_diag = np.array([500.0,   # contour x (reduced from 1000 for turn flexibility)
                           500.0,   # contour y (reduced from 1000 for turn flexibility)
                           2000.0,  # contour z
                           200.0,   # lag (along-track)
                           50.0,    # progress reward weight
                           1000.0,  # heading alignment (1-cos: drives yaw turns)
                           200.0,   # pitch alignment (sin: drives trim recovery)
                           100.0])  # v_theta-to-vehicle-velocity synchronization
        Q = np.diag(Q_diag)


        R_diag = np.array([5e-2,   # VBS rate (increased to smooth out VBS actuation)
                           1e-1,   # LCG rate
                           1e-1,   # stern angle rate
                           1e-1,   # rudder angle rate (low penalty allows fast swing into turns)
                           1e-8,   # RPM1 rate
                           1e-8,   # RPM2 rate
                           1e0])   # delta_v_theta rate
        R = np.diag(R_diag)

        # Terminal cost — position + quaternion + velocity (unchanged)
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


        # Parameter vector layout:
        #   [state_ref(21), control_ref(7), goal_pos(3), t_hat(3), theta_hat(1)]
        # Total = nx + nu + 3 + 3 + 1 = 35
        n_params = self.nx + self.nu + 3 + 3 + 1
        self.ocp.parameter_values = np.zeros((n_params,))

        # Declare the symbolic parameter vector so CasADi expressions can index it.
        p_sym = ca.MX.sym("p", n_params)
        self.model.p = p_sym

        self.n_stage_cost = 8 + self.nu   # 8 MPCC residuals + 7 control rates = 15
        self.n_terminal_cost = 8 + 1       # 8 MPCC residuals + heave velocity

        # We have the cost defined in the model.
        self.ocp.cost.yref = np.zeros((self.n_stage_cost,))
        self.ocp.cost.cost_type = "NONLINEAR_LS"
        self.ocp.cost.W = ca.diagcat(Q, R).full()
        self.ocp.model.cost_y_expr = self.compute_stage_cost(
            self.model.x, self.model.u, self.model.p, terminal=False
        )

        # Terminal cost: MPCC tracking + heave velocity damping to prevent
        # depth overshoot.  No surge penalty (braking constraint handles that).
        # Weight is moderate so it doesn't dominate the MPCC gradient.
        w_heave = 200.0
        self.ocp.cost.cost_type_e = "NONLINEAR_LS"
        self.ocp.cost.W_e = ca.diagcat(Q, np.diag([w_heave])).full()
        self.ocp.model.cost_y_expr_e = self.compute_stage_cost(
            self.model.x, self.model.u, self.ocp.model.p, terminal=True
        )
        self.ocp.cost.yref_e = np.zeros((self.n_terminal_cost,))

        # --------------------- Constraint Setup --------------------------
        vbs_dot = 200  # Maximum rate of change for the VBS
        lcg_dot = 50  # Maximum rate of change for the LCG
        tv_dot = 0.5  # Maximum rate of change for the thrust vectoring (was 0.2; raised for faster turns)
        delta_v_theta_max = 0.5  # Maximum progress speed (m/s arc-length)

        # Declare initial state
        self.ocp.constraints.x0 = np.zeros(
            (self.nx,)
        )  # Initial state is zero. This is set in the sim. for-loop

        # Control bounds: physical actuator rates + delta_v_theta (allow deceleration)
        self.ocp.constraints.lbu = np.array([-vbs_dot, -lcg_dot, -tv_dot, -tv_dot, -delta_v_theta_max])
        self.ocp.constraints.ubu = np.array([vbs_dot, lcg_dot, tv_dot, tv_dot, delta_v_theta_max])
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3, 6])

        # --- position bounds (NED: z positive down) ---
        x_min, x_max = 0.0, 8.0
        y_min, y_max = -2.0, 2.0
        z_min, z_max = -0.5, 3.0

        pos_lbx = np.array([x_min, y_min, z_min])
        pos_ubx = np.array([x_max, y_max, z_max])

        # --- actuator state bounds for x[13:19] = [x_vbs, x_lcg, δs, δr, rpm1, rpm2] ---
        act_lbx = np.array([0.0, 0.0, -np.deg2rad(7), -np.deg2rad(7), -500.0, -500.0])
        act_ubx = np.array([100.0, 100.0, np.deg2rad(7), np.deg2rad(7), 450.0, 450.0])

        # x[19] = theta (arc-length progress)
        #theta_lbx = np.array([0.0])
        #theta_ubx = np.array([1e6])

        # x[20] = v_theta (progress speed along path)
        v_theta_max = 0.6  # m/s — must be >= v_near in reference propagation (was 0.4)
        v_theta_lbx = np.array([0.0])
        v_theta_ubx = np.array([v_theta_max])

        idxbx = np.r_[[0, 1, 2], [13, 14, 15, 16, 17, 18], [20]]
        lbx = np.r_[pos_lbx, act_lbx, v_theta_lbx]
        ubx = np.r_[pos_ubx, act_ubx, v_theta_ubx]

        self.ocp.constraints.idxbx = idxbx
        self.ocp.constraints.lbx = lbx
        self.ocp.constraints.ubx = ubx

        # Soft constraints on position box bounds (first 3 entries in idxbx)
        # v_theta bound (index 9 in idxbx) is kept hard — it must never exceed v_theta_max
        idxsbx = np.array([0, 1, 2])
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
        a_brake = 0.001 # Sim break: 0.1   # m/s^2  (was 0.005 — 20× increase for trajectory following)
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

        # ----- Track (tube) constraint ------------------------------------------------
        # Keep the vehicle within a tube of radius r_track around the path.
        # h_track = ||e_c||^2   (cross-track error squared, perpendicular to tangent)
        # Bound:  0 <= h_track <= r_track^2   (upper bound set via uh)
        #
        # To tighten/loosen at runtime per stage k:
        #   uh = solver.constraints_get(k, "uh")
        #   uh[-1] = new_r ** 2
        #   solver.constraints_set(k, "uh", uh)

        # Compute e_c_vec symbolically from model state and parameters (same
        # path geometry as the stage cost) so the track constraint shares the
        # same linearization point as the cost.
        _idx_t = self.nx + self.nu + 3
        _t_hat = self.model.p[_idx_t : _idx_t + 3]
        _theta_hat = self.model.p[_idx_t + 3]
        _theta = self.model.x[self.N_PHYS_STATES]
        _p_ref = self.model.p[:3]
        _path_pos = _p_ref + _t_hat * (_theta - _theta_hat)
        _pos_diff = self.model.x[:3] - _path_pos
        _e_l = ca.dot(_pos_diff, _t_hat)
        self.model.e_c_vec = _pos_diff - _e_l * _t_hat

        h_track = ca.dot(self.model.e_c_vec, self.model.e_c_vec)
        self.r_track = 1.0  # [m] default tube radius (was 0.5; widened for turns)
        self.IDX_TRACK = 2  # index of h_track inside con_h (after brake_h removal)

        # con_h layout: [brake_h(1), h_dz1(1), h_dz2(1), h_track(1)]
        r_sq = self.r_track ** 2
        self.ocp.model.con_h_expr = ca.vertcat(brake_h, h_dz1, h_dz2, h_track)
        self.ocp.constraints.lh = np.array([-1e9, -1e9, -1e9, 0.0])
        self.ocp.constraints.uh = np.array([ 0.0,  0.0,  0.0, r_sq])
        self.ocp.constraints.idxsh = np.arange(4)
        n_sh = 4
        #self.ocp.model.con_h_expr = ca.vertcat(h_dz1, h_dz2, h_track)
        #self.ocp.constraints.lh = np.array([-1e9, -1e9, 0.0])
        #self.ocp.constraints.uh = np.array([ 0.0,  0.0, r_sq])
        #self.ocp.constraints.idxsh = np.arange(3)
        #n_sh = 3

        # Terminal nonlinear constraint (same structure + track)
        self.ocp.model.con_h_expr_e = ca.vertcat(brake_h, h_dz1, h_dz2, h_track)
        self.ocp.constraints.lh_e = np.array([-1e9, -1e9, -1e9, 0.0])
        self.ocp.constraints.uh_e = np.array([ 0.0,  0.0,  0.0, r_sq])
        self.ocp.constraints.idxsh_e = np.arange(4)
        n_sh_e = 4
        #self.ocp.model.con_h_expr_e = ca.vertcat(h_dz1, h_dz2, h_track)
        #self.ocp.constraints.lh_e = np.array([-1e9, -1e9, 0.0])
        #self.ocp.constraints.uh_e = np.array([ 0.0,  0.0, r_sq])
        #self.ocp.constraints.idxsh_e = np.arange(3)
        #n_sh_e = 3

        # ----- Unified slack penalty vectors ------------------------------------
        # acados orders slack variables as: [idxsbx | idxsh] for stage costs.
        # Terminal stage only has idxsh_e.
        Z_pos   = 1e6   # quadratic penalty for position box violations (hard wall)
        z_pos   = 1e4   # linear   penalty for position box violations

        # Brake penalty sizing: with a_brake = 0.001 the constraint is
        # permanently violated at any cruise speed (brake_h ≈ 0.03–0.09).
        # The penalty must be LOW so the permanent violation doesn't corrupt
        # the QP gradient — otherwise the brake gradient (reduce speed) fights
        # the MPCC progress gradient (increase speed) and causes QP failure
        # under SQP_RTI.  With Z=1, z=0.1 the per-stage penalty is < 0.02,
        # negligible vs the tracking cost (~500).  The constraint still
        # provides a gentle nudge to slow down near the goal.
        Z_brake = 1.0   # quadratic penalty for braking funnel violations
        z_brake = 0.1   # linear   penalty for braking funnel violations

        Z_dz    = 200.0 # quadratic penalty for RPM deadzone
        z_dz    = 20.0  # linear   penalty for RPM deadzone

        Z_track = 1e5   # quadratic penalty for track tube violation
        z_track = 1e3   # linear   penalty for track tube violation

        # Stage: [sbx(3), sh_brake(1), sh_dz(2), sh_track(1)] = size 7
        self.ocp.cost.Zl = np.r_[Z_pos * np.ones(n_sb), Z_brake, Z_dz, Z_dz, Z_track]
        self.ocp.cost.Zu = np.r_[Z_pos * np.ones(n_sb), Z_brake, Z_dz, Z_dz, Z_track]
        self.ocp.cost.zl = np.r_[z_pos * np.ones(n_sb), z_brake, z_dz, z_dz, z_track]
        self.ocp.cost.zu = np.r_[z_pos * np.ones(n_sb), z_brake, z_dz, z_dz, z_track]
        #self.ocp.cost.Zl = np.r_[Z_pos * np.ones(n_sb), Z_dz, Z_dz, Z_track]
        #self.ocp.cost.Zu = np.r_[Z_pos * np.ones(n_sb), Z_dz, Z_dz, Z_track]
        #self.ocp.cost.zl = np.r_[z_pos * np.ones(n_sb), z_dz, z_dz, z_track]
        #self.ocp.cost.zu = np.r_[z_pos * np.ones(n_sb), z_dz, z_dz, z_track]

        # Terminal: [sh_e_brake(1), sh_e_dz(2), sh_e_track(1)] = size 4
        self.ocp.cost.Zl_e = np.r_[Z_brake, Z_dz, Z_dz, Z_track]
        self.ocp.cost.Zu_e = np.r_[Z_brake, Z_dz, Z_dz, Z_track]
        self.ocp.cost.zl_e = np.r_[z_brake, z_dz, z_dz, z_track]
        self.ocp.cost.zu_e = np.r_[z_brake, z_dz, z_dz, z_track]
        #self.ocp.cost.Zl_e = np.r_[Z_dz, Z_dz, Z_track]
        #self.ocp.cost.Zu_e = np.r_[Z_dz, Z_dz, Z_track]
        #self.ocp.cost.zl_e = np.r_[z_dz, z_dz, z_track]
        #self.ocp.cost.zu_e = np.r_[z_dz, z_dz, z_track]

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
        self.ocp.solver_options.nlp_solver_max_iter = 1
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
        self.sim.parameter_values = np.zeros(n_params)

        self.sim.solver_options.T = 0.1
        self.sim.solver_options.integrator_type = "ERK"

    # Number of physical states / controls (before MPCC augmentation).
    N_PHYS_STATES = 19
    N_PHYS_CONTROLS = 6

    def export_dynamics_model(self, casadi_model):
        # Augmented state: [physical(13), actuator_state(6), theta(1), v_theta(1)] = 21
        # Augmented control: [actuator_rates(6), delta_v_theta(1)] = 7
        x_sym = ca.MX.sym("x", self.N_PHYS_STATES + 2, 1)
        u_sym = ca.MX.sym("u_sym", self.N_PHYS_CONTROLS + 1, 1)
        x_dot_sym = ca.MX.sym("x_dot", self.N_PHYS_STATES + 2, 1)

        x_dot = casadi_model.dynamics(export=True)
        f_expl = ca.vertcat(
            x_dot(x_sym[:13], x_sym[13:19]),   # 13 physical state derivatives
            u_sym[:6],                         # 6 actuator rate-of-change
            x_sym[20],                         # theta_dot = v_theta (x[20])
            u_sym[6],                          # v_theta_dot = delta_v_theta
        )
        f_impl = x_dot_sym - f_expl
        
        
        model = AcadosModel()
        model.name = "SAM_equation_system"
        model.x = x_sym
        model.xdot = x_dot_sym
        model.u = u_sym

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


    def set_track_radius(self, solver, r_track, stages=None):
        """
        Dynamically change the track tube radius at runtime.

        :param solver: AcadosOcpSolver instance
        :param r_track: Tube radius in metres.  Scalar applies to all stages,
                        or pass an array of length len(stages) for per-stage radii.
        :param stages: Iterable of stage indices (0..N).  None = all stages.
        """
        if stages is None:
            stages = range(self.N_horizon + 1)

        r_sq = np.atleast_1d(np.asarray(r_track, dtype=float)) ** 2
        broadcast = r_sq.size == 1

        for i, k in enumerate(stages):
            val = float(r_sq[0] if broadcast else r_sq[i])
            if k < self.N_horizon:
                uh = solver.constraints_get(k, "uh")
                uh[self.IDX_TRACK] = val
                solver.constraints_set(k, "uh", uh)
            else:
                uh_e = solver.constraints_get(k, "uh")
                uh_e[self.IDX_TRACK] = val
                solver.constraints_set(k, "uh", uh_e)
                
    def compute_stage_cost(self, x, u, p, terminal):
        """
        Compute the stage or terminal cost residual.

        Stage  (terminal=False):
          [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1), u_phys(6), delta_v_theta(1)] = 15

        Terminal (terminal=True):
          [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1), heave_vel(1)] = 9

        p vector layout (set in DiveControllerMPC.update):
          p[0 : nx]            = state ref   (nx = 21)
          p[nx : nx+nu]        = control ref (nu = 7)
          p[nx+nu : nx+nu+3]   = goal_pos    (3)
          p[nx+nu+3 : nx+nu+6] = t_hat       (3)
          p[nx+nu+6]           = theta_hat   (1)
          Total = 21 + 7 + 3 + 3 + 1 = 35
        """
        #if terminal:
        #    pos_error = x[:3] - p[:3]

        #    q1 = p[3:7]
        #    q1 = q1 / ca.norm_2(q1)
        #    q2 = x[3:7]
        #    q_conj = ca.vertcat(q2[0], -q2[1], -q2[2], -q2[3])
        #    q2 = q_conj / ca.norm_2(q2)

        #    q_w = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
        #    q_x = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
        #    q_y = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
        #    q_z = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

        #    q_error = ca.vertcat(q_w, q_x, q_y, q_z)
        #    q_error = ca.if_else(q_w < 0, -q_error, q_error)
        #    q_att_error = ca.vertcat(
        #        1.0 - q_error[0], q_error[1], q_error[2], q_error[3]
        #    )
        #    vel_error = x[7:13] - p[7:13]
        #    return ca.vertcat(pos_error, q_att_error, vel_error)

        # ---- MPCC stage cost ----
        p_ref = p[:3]
        idx_t = x.rows() + u.rows() + 3   # skip ref_row + goal_pos
        t_hat = p[idx_t : idx_t + 3]
        theta_hat = p[idx_t + 3]
        theta = x[self.N_PHYS_STATES]          # x[19]

        path_pos = p_ref + t_hat * (theta - theta_hat)
        pos_diff = x[:3] - path_pos
        e_l = ca.dot(pos_diff, t_hat)
        e_c_vec = pos_diff - e_l * t_hat

        # Vehicle forward axis from quaternion (first column of rotation matrix)
        q0_s, q1_s, q2_s, q3_s = x[3], x[4], x[5], x[6]
        fwd_x = 1 - 2 * (q2_s**2 + q3_s**2)
        fwd_y = 2 * (q1_s * q2_s + q0_s * q3_s)
        fwd_z = 2 * (q1_s * q3_s - q0_s * q2_s)
        cos_align = fwd_x * t_hat[0] + fwd_y * t_hat[1] + fwd_z * t_hat[2]
        e_heading = 1 - cos_align

        # Pitch alignment: sin(pitch_state) - sin(pitch_ref).
        # sin(pitch) = -fwd_z = 2*(q0*q2 - q1*q3), smooth polynomial in quaternion.
        # sin(pitch_ref) = -t_hat[2] (from the path tangent).
        # Unlike 1-cos (quartic near zero), this has a LINEAR gradient for
        # small pitch errors, giving the solver real incentive to level out.
        e_pitch = (-fwd_z) - (-t_hat[2])

        # v_theta: set yref[4] = v_target to pull progress speed toward v_target.
        v_theta = x[self.N_PHYS_STATES + 1]   # x[20]

        # Surge velocity projected onto path tangent.  Couples v_theta to
        # actual vehicle motion so the solver cannot advance theta without
        # producing physical velocity (the root cause of the no-movement bug).
        v_along = x[7] * cos_align
        e_sync = v_theta - v_along

        if terminal:
            return ca.vertcat(
                e_c_vec, e_l, v_theta, e_heading, e_pitch, e_sync,
                x[9],   # heave velocity (prevent depth overshoot)
            )
        else:
            return ca.vertcat(
                e_c_vec, e_l, v_theta, e_heading, e_pitch, e_sync,
                u[:self.N_PHYS_CONTROLS], u[self.N_PHYS_CONTROLS],
            )

    def x_error(self, x, u, ref, terminal):
        """
        MPCC cost residual.

        Stage  (terminal=False): [e_c_vec(3), e_l(1), e_heading(1), v_theta(1), u_phys(6)] = 12
            Contour/lag errors from a locally-linearized path.  e_heading aligns
            the vehicle's forward axis with the path tangent, preventing lateral
            drift and overshoots at turns.  v_theta is the raw progress speed —
            the controller sets yref[5] = v_target so the quadratic cost acts as
            a progress reward.

        Terminal (terminal=True): [pos_error(3), q_att_error(4), vel_error(6)] = 13
            Position + heading + velocity tracking to the final waypoint.
        """
        if terminal:
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
            q_att_error = ca.vertcat(
                1.0 - q_error[0], q_error[1], q_error[2], q_error[3]
            )
            vel_error = x[7:13] - ref[7:13]
            return ca.vertcat(pos_error, q_att_error, vel_error)

        # ---- MPCC stage cost ----
        p_ref = ref[:3]
        idx_t = self.nx + self.nu + 3          # start of t_hat in param vector
        t_hat = ref[idx_t : idx_t + 3]
        theta_hat = ref[idx_t + 3]
        theta = x[self.N_PHYS_STATES]          # x[19]

        path_pos = p_ref + t_hat * (theta - theta_hat)
        pos_diff = x[:3] - path_pos
        e_l = ca.dot(pos_diff, t_hat)
        e_c_vec = pos_diff - e_l * t_hat

        # Heading alignment: vehicle forward axis (from quaternion) vs path tangent.
        # e_heading = 0 when aligned, 2 when facing backward.
        q0, q1q, q2q, q3q = x[3], x[4], x[5], x[6]
        fwd_x = 1 - 2 * (q2q**2 + q3q**2)
        fwd_y = 2 * (q1q * q2q + q0 * q3q)
        fwd_z = 2 * (q1q * q3q - q0 * q2q)
        e_heading = 1 - (fwd_x * t_hat[0] + fwd_y * t_hat[1] + fwd_z * t_hat[2])

        v_theta = u[self.N_PHYS_CONTROLS]      # u[6]

        return ca.vertcat(e_c_vec, e_l, e_heading, v_theta, u[:self.N_PHYS_CONTROLS])
