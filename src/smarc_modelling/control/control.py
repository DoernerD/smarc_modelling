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
        #   Stage:    [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1),
        #              e_rudder_auth(1), e_stern_auth(1), yaw_rate(1), u_phys(6), delta_v_theta(1)] = 18
        #   Terminal: [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1),
        #              e_rudder_auth(1), e_stern_auth(1), yaw_rate(1), heave_vel(1)] = 12

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
        
        #        e_c_vec, e_l, v_theta, e_heading, e_pitch, e_sync,
        #        e_rudder_auth, e_stern_auth,
        #        u[:self.N_PHYS_CONTROLS], u[self.N_PHYS_CONTROLS],
        
        Q_diag = np.array([700.0,   # SIM: contour x (500 -> 700 -> 1200 -> 700). 1200 amplified a "wrong-direction at start of turn" pathology: e_c_vec is decomposed against the LOOKAHEAD tangent (heading_offset=2 m ahead in DiveControllerMPC), so on a turn the perpendicular-to-lookahead direction differs from perpendicular-to-local-path, and high Q_contour_y pushed the rudder along that biased direction.  Reverted to 700 and instead halved heading_offset (2 m -> 1 m in the controller) to bring the lookahead inside the prediction horizon and reduce the decomposition bias at its source.
                           700.0,   # SIM: contour y (500 -> 700 -> 1200 -> 700). Same rationale as contour x.
                           1500.0,  # SIM: contour z (real-vehicle was 850; raised so the solver commits to the dive faster)
                           80.0,    # SIM: lag (was 300). At the dive transition the path tangent rotates (acquires a z-component) and the projection re-linearizes, leaving the vehicle with a large positive e_l = (pos - path_pos)·t_hat along the new tangent. With Q_lag=300 vs Q_sync=30 the cheapest way to reduce e_l was NEGATIVE surge (pull vehicle back along tangent), not advancing theta — which v_theta is already saturated doing. Reversing surge also inverts the rudder's lift sign, producing the "random" turning you see.  Dropped hard so lag is resolved by v_theta catch-up and mild drag, not by reversing.
                           50.0,    # SIM: progress speed penalty (real-vehicle used 10, but with sync=100 that makes rest cheaper than cruise; bumped back toward the pre-real-vehicle value of 50)
                           300.0,   # SIM: heading yaw alignment (400 -> 150 -> 300). 400 caused the 90-degree turn pathology on straight lines (high heading gain + cheap rudder rate). 150 fixed that but made the solver under-react on curves — the "turn too late, then too much" pattern on gentle-turn-dive: Q_rudder_angle/Q_heading = 500/150 ≈ 3.3, so yaw error has to grow to ~12° before it's cheaper to deflect the rudder than accept the error, and once it does the response is sharp. 300 brings the ratio to 500/300 ≈ 1.7, giving earlier (smaller-error) rudder engagement on curves. Straight-line oscillation that originally drove 400 -> 150 is now held by R_rudder=2e0 (rate) + Q[8]=500 (state), both of which were added/raised after the 400 regime.
                           1000.0,  # SIM: pitch alignment (was 600). Brought closer to contour_z=1500 so the vehicle tilts along the path preemptively rather than chasing z-error once it's already grown.
                           120.0,   # SIM: v_theta-to-vehicle-velocity synchronization (was 30, briefly 100).  Now the DOMINANT longitudinal cost (vs Q_lag=80): the solver is forced to keep x[7]·cos_align ≈ v_theta, which makes "reverse surge to reduce lag" expensive enough that it's no longer optimal.  Also naturally pulls v_theta down as cos_align drops through the pitch transition, so the marker slows in sync with the vehicle instead of running ahead.
                           500.0,   # SIM: rudder angle penalty (800 -> 500 -> 350 -> 500). Full-rudder saturation crossover: Q_heading=300 balances Q[8] at e_heading = sqrt(Q[8]*δ_max²/Q_heading), so 800 → ~12°, 500 → ~9°, 350 → ~7.6°.  Raised 350 -> 500 to tame post-pivot oversteer on the 180°-return trajectory: with Q[8]=350 the solver saturated the rudder at yaw errors as small as 7.6°, which combined with the residual lateral offset after the pivot was enough to slam the nose past the tangent and then oscillate.  At Q[8]=500 the saturation crossover moves to ~9°, so the solver uses PARTIAL rudder (proportional response) through the 5–9° regime where recovery happens, giving the heading loop time to converge smoothly instead of bang-banging.  Earlier 350 value was chosen to get aggressive rudder engagement on gentle-turn-dive curves; the 180°-return trajectory reveals that the same aggressiveness causes recovery overshoot.  If future trajectories require the earlier-engagement behaviour again, this can be split into a trajectory-adaptive weight, but the 500 value still commits full rudder by ~9° so curve tracking should remain acceptable.  Straight-line wiggle still held by R_rudder=2e0 (rate cost).
                           200.0,   # SIM: stern angle penalty (was 300; reduced so the pitch plane is less damped than the yaw plane — stern has a structural job rudder doesn't have).
                           20.0])   # SIM: yaw-rate (x[12] = r) damping (30 -> 10 -> 20).  30 was chosen to tame the monotonic over-rotation pathology at trace row 177 (yaw_err growing -107°->-113° with restoring rudder saturated), and it DID help there: track-slack biting ticks dropped 1562/2378 -> 971/2265 (66% -> 43%) and the old bang-bang rudder recovery signature disappeared.  But it introduced a new regime: 135/2265 (6%) of ticks at |yaw_err|>120°, compared to 0/2378 in the no-damping run.  Mechanism: at Q_r=30, holding a perturbed heading with zero yaw rate is cheaper than rotating back through a yaw-rate burst, so the solver settles into extreme-yaw equilibria (observed at row 686: yaw_err stable at -144° for 5 consecutive ticks, full-reverse commanded) instead of bang-bang recovering.  The "orientation suddenly 90° then goes straight again" observation on the dive segment matches this: a small heading perturbation is now cheaper to hold than to correct quickly.  10 is the minimum value still expected to prevent row-177-style runaway (at r=2 rad/s it contributes 40/stage, comparable to Q_sync at v_along error), while being weak enough that rotating through a moderate yaw-rate burst (~1 rad/s, cost = 10/stage) is preferred over sitting at large heading error (Q_heading * π² ≈ 3000/stage if stuck near ±π).  If |yaw|>120° ticks return, the next knob is bumping R_rudder rate from 2e0 to 4e0 (slows bang-bang rudder without adding a state-hold attractor) rather than raising Q_r back.  10 -> 20 (180°-return-y0 trajectory): the |yaw|>120° regime returned, but as a pure end-of-trajectory hover-spin rather than a transient — once theta locks near theta_total - 0.7 with the body 1 m cross-track of the final WP (mpcc_run_1777064774 row 1700+), yaw monotonically spins through 360° while drifting around at low surge.  This is the cheap-state-hold equilibrium the original Q_r=30 comment described.  Going to 20 (not back to 30) because the row-177 dive pathology that originally drove 30 -> 10 was on a different trajectory; 20 keeps the rotation-through-yaw-rate-burst cost at 20/stage at r=1 rad/s, still lower than Q_heading at moderate error, but doubles the resistance to the spin-equilibrium attractor.  R_rudder bump deferred until we see whether 20 alone closes the post-pivot pinwheel.
       # NOTE: a context-gated anti-reverse residual (straight_gate ·
        # max(0, -u), Q=100) was tried briefly as a 12th MPCC weight to
        # break the reverse-during-dive pathology on gentle_dive_test_wide_2.
        # It had no observable effect — the solver still commanded reverse
        # on the straight dive.  Plausible reasons the gate approach
        # failed: (a) the gate itself was fine (gate≈1 on the dive) but
        # Q=100 gives only ~1/stage cost at u=-0.1, too small vs whatever
        # other cost was pulling reverse; (b) the pathology is not a
        # "reverse-surge basin" steady state at all, but a short transient
        # at a specific path feature (dive-transition pitch kick?) that
        # the horizon-integrated anti-reverse penalty can't prevent.
        # Reverted; see git history for the full residual implementation.
        # NOTE: a Q_fwd_bias entry was briefly added here as an 11th weight to
        # tie-break between the forward/reverse basins of a symmetric
        # e_heading = sin(yaw_err) cost.  That approach failed structurally:
        # symmetric heading produces a bifurcated landscape (two zero-cost
        # basins separated by a ridge at yaw_err = ±π/2), and the solver's
        # warm-start non-deterministically picks a basin.  No scalar weight
        # can simultaneously (a) keep the return leg pinned in the forward
        # basin against lateral-correction overshoot and (b) let the pivot
        # hop into the reverse basin cheaply enough for a K-turn to emerge.
        # The ridge disappears only when Q_fwd_bias ≥ Q_heading/2 = 150, by
        # which point K-turns are prohibitively expensive (600/stage at π).
        # Reverted to atan2 heading (monotonic, single basin); K-turn
        # accessibility now depends on e_l cost overcoming heading cost
        # during sharp pivots — bumping Q_lag is the next knob if the
        # tightest turns time out.
        # NOTE: an LCG-toward-neutral trim attractor (e_lcg_neutral =
        # (x[14] - 50)·gate_flat with gate_flat = 1/(1+K·t_hat[2]²)) and
        # a companion body-pitch-toward-zero residual (e_pitch_flat =
        # (-fwd_z)·gate_flat) were tried as 12th and 13th MPCC weights
        # to break the LCG-forward absorbing basin and the surge-coupled
        # depth oscillation it enabled (pitched body + variable surge →
        # vertical motion overshoots depth → reverse → overshoots up →
        # repeat).  Multiple gating + weight combinations were tried:
        #   * Ungated, Q_lcg=0.02, 0.05: too weak — LCG still parked at 95.
        #   * t_horiz = sqrt(tx²+ty²) gate, Q_lcg=0.5, Q_pitch_flat=5000:
        #     gate stayed ~0.9 across the 17–27° dive sections of the
        #     gentle_dive trajectories, so the residuals fought the dive
        #     and the vehicle would not descend.
        #   * gate_flat = 1/(1+200·t_hat[2]²), Q_lcg=0.5, Q_pitch_flat=5000:
        #     dive-fighting fixed (gate ≈ 0.06 at 17°), but flat-section
        #     LCG-park behaviour was not measurably improved either.
        # Reverted; the loop is not closable from this single cost knob
        # with the current dynamics model.  Possible follow-ups if this
        # is revisited: separate "depth-hold" mode that hard-routes
        # depth through VBS, decoupling pitch reference from path tangent
        # entirely (replace e_pitch with pitch-toward-zero everywhere
        # and let lag/contour terms handle path tracking), or modelling
        # changes that make VBS-driven depth correction locally cheaper
        # than pitch+thrust inside a 4.5 s horizon.
        Q = np.diag(Q_diag)


        R_diag = np.array([5e-4,   # SIM: VBS rate (was 2e-3).  Q_pitch=1000 and Q_contour_z=1500 are strong but VBS rate cost was still ~80/stage at full sweep, which was suppressing the VBS' contribution to the dive kick-off.  5e-4 makes VBS cheap enough that the solver actually uses it at the moment the path starts descending (fast actuator, no surge dependency), rather than waiting for stern-driven pitch to build up.
                           2e-3,   # SIM: LCG rate (was 1e-2).  LCG creates the static pitch-down moment and does so without surge, so it should be leading the dive alongside VBS.  1e-2 was still expensive enough (~25/stage at full rate) to compete with Q_pitch, dropping it 5x lets LCG move aggressively into a dive-trim position early.
                           5e-1,   # SIM: stern angle rate (was 2e0; reduced because stern has a structural pitch-control role rudder doesn't — over-damping it muted the dive. Q[9]=200 state penalty still prevents parking at ±7° on small pitch errors).
                           2e0,    # SIM: rudder angle rate (pre-real-vehicle 1e-1 -> 2e0). Kills the small-amplitude rudder wiggle that Q[8] alone can't catch (wiggles near 0 deflection have ~0 state cost). 180° turns still feasible: full-sweep rate cost ~0.7 vs heading benefit of ~45000.
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
        #   [state_ref(21), control_ref(7), goal_pos(3), t_hat(3), theta_hat(1), t_local_hat(3)]
        # Total = nx + nu + 3 + 3 + 1 + 3 = 38
        #
        # t_hat is the LOOKAHEAD tangent (path tangent ~heading_offset ahead
        # of theta_hat, see DiveControllerMPC.get_current_ref_array).  It
        # feeds the COST (position + attitude + sync residuals) so the
        # vehicle anticipates upcoming curvature.  heading_offset is kept
        # short enough that t_hat does not sample past the next tight curve
        # from the dive approach — that was the root cause of the
        # "premature rotation at waypoint 4" pathology on
        # gentle_dive_test_wide_2.
        # t_local_hat is the LOCAL tangent at theta_hat itself.  It feeds
        # the TRACK CONSTRAINT only (h_track = ||e_c||^2 with the local
        # perpendicular): using the lookahead tangent there made the
        # constraint measure distance to a line that pointed slightly off-
        # path on curves, letting the vehicle drift far from the spline
        # while still satisfying ||e_c|| <= r_track.  Splitting the two
        # restores the geometric meaning of the tube ("perpendicular
        # distance from the local path") while keeping the lookahead
        # anticipation in the cost.
        # (An earlier revision fed t_local_hat into the attitude/sync
        # residuals as well; that version regressed on this trajectory
        # because stage-0 attitude lost its anticipation, so the vehicle
        # could not distribute the hairpin's 180° yaw change over enough
        # arc length to stay below rudder saturation.  Reverted; see the
        # long comment in compute_stage_cost.)
        n_params = self.nx + self.nu + 3 + 3 + 1 + 3
        self.ocp.parameter_values = np.zeros((n_params,))

        # Declare the symbolic parameter vector so CasADi expressions can index it.
        p_sym = ca.MX.sym("p", n_params)
        self.model.p = p_sym

        self.n_stage_cost = 11 + self.nu   # 11 MPCC residuals (incl. yaw-rate) + 7 control rates = 18
        self.n_terminal_cost = 11 + 1       # 11 MPCC residuals (incl. yaw-rate) + heave velocity = 12

        # We have the cost defined in the model.
        self.ocp.cost.yref = np.zeros((self.n_stage_cost,))
        self.ocp.cost.cost_type = "NONLINEAR_LS"
        self.ocp.cost.W = ca.diagcat(Q, R).full()
        self.ocp.model.cost_y_expr = self.compute_stage_cost(
            self.model.x, self.model.u, self.model.p, terminal=False
        )

        # Terminal cost: MPCC tracking + heave velocity damping.
        # Penalizes residual downward velocity at the end of the horizon to
        # discourage momentum build-up during dives.
        w_heave = 50.0
        self.ocp.cost.cost_type_e = "NONLINEAR_LS"
        self.ocp.cost.W_e = ca.diagcat(Q, np.diag([w_heave])).full()
        self.ocp.model.cost_y_expr_e = self.compute_stage_cost(
            self.model.x, self.model.u, self.ocp.model.p, terminal=True
        )
        self.ocp.cost.yref_e = np.zeros((self.n_terminal_cost,))

        # --------------------- Constraint Setup --------------------------
        vbs_dot = 200  # Maximum rate of change for the VBS
        lcg_dot = 50  # Maximum rate of change for the LCG
        # SIM: tv_dot restored to 0.5.  With R[2]=R[3]=2e0, the solver pays a
        # real cost for fast rate — which is what we actually wanted.  The
        # earlier drop to 0.3 was a physical floor added *because* R was too
        # cheap to damp oscillation; now that R does that job, we don't want
        # the physical floor throttling the rudder during intentional sharp
        # turns (the 180°+ trajectories the planner is meant to demonstrate).
        tv_dot = 0.5  # Maximum rate of change for the thrust vectoring [rad/s]
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
        x_min, x_max = 0.0, 8.0     # Wall lock handles the minimum once the vehicle moves forward.
        y_min, y_max = -2.0, 2.0
        z_min, z_max = -0.5, 2.7    # -0.5 is in case the pressure sensor gives funky readings at the surface

        pos_lbx = np.array([x_min, y_min, z_min])
        pos_ubx = np.array([x_max, y_max, z_max])

        # --- surge velocity bound for x[7] ---
        # SIM: raised from 0.2 (real-vehicle cap, which was conservative because
        # the real vehicle is faster than the model at 300 RPM).  In sim the
        # dynamics match the model, so 0.2 was capping the solver below what
        # the sim can actually achieve and contributing to the stall.
        # SIM: tightened 0.5 -> 0.4 to reduce end-of-trajectory overshoot.
        # With _v_target=0.4, surge_max=0.5 gave the solver 25 % headroom
        # above cruise target — useful for catching up lag, but it also
        # meant the vehicle entered the approach phase carrying more kinetic
        # energy than the terminal brake + drag could fully dissipate inside
        # a 3 s horizon.  Matching surge_max to _v_target removes that
        # headroom so the solver can't cruise faster than the brake can
        # stop.  v_theta_max tracks surge_max via the line below so the
        # sync coupling stays consistent.
        # SIM: briefly softened 0.4 -> 0.3 to tame post-pivot recovery on
        # the y=0.5 return trajectory, then reverted back to 0.4.  The
        # problem with 0.3 was that surge_min stayed at -0.4 (reverse
        # authority UNCHANGED), so the forward/reverse bounds became
        # asymmetric — reverse had 33 % more authority than forward,
        # which biased the solver toward "go backwards to close lag"
        # over "go forward to close lag" whenever the local trade-off
        # was close.  That showed up as "goes backward for no reason"
        # and "stuck in pivot" runs: with reverse locally cheaper,
        # the solver kept finding reverse-dominant paths that never
        # completed the forward approach to the return leg.  Back at
        # ±0.4 the two directions are symmetric again and the solver
        # has no structural reason to prefer reverse.  The recovery
        # overshoot that motivated 0.3 was addressed geometrically by
        # relaxing the return-leg offset (y=0.5 -> y=1.0) in the
        # trajectory, so the ballistic-overshoot concern that made 0.3
        # attractive no longer applies.
        surge_max = 0.4   # m/s
        # SIM: reopened from -0.1 to -0.4 so reverse has the same top speed
        # as forward — lets the solver use reverse thrust for tight turns,
        # station-keeping and repositioning (the "maneuverability" payoff).
        # The reverse-at-dive pathology that motivated -0.1 is now fully
        # handled by the cost rebalancing (Q_lag=80, Q_sync=120): at a dive
        # transition with e_l=0.3 m and cos_align≈0.7, reversing to -0.4 m/s
        # costs ~55/stage in e_sync while saving only ~7/stage in e_l, so
        # reverse is ~7x more expensive than accepting the lag.  Balance
        # holds for the whole dive phase; bug does not resurface at -0.4.
        surge_min = -0.4  # m/s — symmetric reverse authority for maneuverability
        surge_lbx = np.array([surge_min])
        surge_ubx = np.array([surge_max])

        # --- actuator state bounds for x[13:19] = [x_vbs, x_lcg, δs, δr, rpm1, rpm2] ---
        act_lbx = np.array([0.0, 0.0, -np.deg2rad(7), -np.deg2rad(7), -500.0, -500.0])
        act_ubx = np.array([100.0, 100.0, np.deg2rad(7), np.deg2rad(7), 450.0, 450.0])

        # x[20] = v_theta (progress speed along path)
        v_theta_max = surge_max  # must match surge_max — e_sync couples v_theta to surge
        v_theta_lbx = np.array([0.0])
        v_theta_ubx = np.array([v_theta_max])

        idxbx = np.r_[[0, 1, 2], [7], [13, 14, 15, 16, 17, 18], [20]]
        lbx = np.r_[pos_lbx, surge_lbx, act_lbx, v_theta_lbx]
        ubx = np.r_[pos_ubx, surge_ubx, act_ubx, v_theta_ubx]

        self.ocp.constraints.idxbx = idxbx
        self.ocp.constraints.lbx = lbx
        self.ocp.constraints.ubx = ubx

        # Terminal state box constraints (same bounds as intermediate stages)
        self.ocp.constraints.idxbx_e = idxbx
        self.ocp.constraints.lbx_e = lbx
        self.ocp.constraints.ubx_e = ubx

        # Soft constraints on position + surge bounds (first 4 entries in idxbx)
        # v_theta bound (last in idxbx) is kept hard — it must never exceed v_theta_max
        idxsbx = np.array([0, 1, 2, 3])
        self.ocp.constraints.idxsbx = idxsbx
        self.ocp.constraints.idxsbx_e = idxsbx
        n_sb = idxsbx.size  # 4

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
        # a_brake controls how early the funnel bites.
        # With a_brake = 0.1 and d_eps = 0.5 (current sim values):
        #   d=0.0 m → cap = sqrt(2*0.1*0.5) = 0.316 m/s (below surge_max=0.4, active)
        #   d=0.3 m → cap = sqrt(2*0.1*0.8) = 0.400 m/s (= surge_max, just touches)
        #   d=0.5 m → cap = sqrt(2*0.1*1.0) = 0.447 m/s (above surge_max, inactive)
        # So the funnel is a soft floor that bites only in the last ~0.3 m
        # before the goal.  Stern-plane pitching authority (∝ surge²) is
        # preserved for the entire approach, and the funnel picks up the
        # last bit of kinetic energy the terminal cost + drag can't quite
        # dissipate inside the horizon — which is the direct fix for the
        # overshoot.
        # SIM: reverted to 0.1.  0.007 was introduced to compensate for the
        # real vehicle being faster than the model at cruise RPM (model
        # mismatch → needed earlier braking).  In sim the dynamics match the
        # model, so that safety margin just throttles surge to ~0.1 m/s for
        # the last ~2.5 m.  At that low surge, the stern plane loses ~17x
        # authority (moment ∝ surge²), and the dive controller is forced onto
        # VBS/LCG for depth hold — which overshoots and produces the
        # end-of-trajectory depth oscillation.
        a_brake = 0.01    # m/s^2 — bites only in the last ~0.3 m at cruise speed
        # SIM: d_eps 1.5 -> 0.5.  d_eps=1.5 left the cap at goal (0.548 m/s)
        # ABOVE surge_max, which made the funnel effectively inactive and
        # meant the only thing braking the vehicle at the goal was drag +
        # terminal cost — not enough to prevent overshoot.  Dropping to 0.5
        # re-engages the funnel for the last ~0.3 m, giving the solver a
        # hard upper bound on approach speed while leaving stern authority
        # intact for the entire cruise phase.
        d_eps   = 0.5    # m — funnel bites in the last ~0.3 m before the goal

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
        # SAM thrusters have a ~200 RPM deadzone where no thrust is produced.
        # The exact boundary varies with water speed, prop loading, etc.
        #
        # Complementarity constraint: h = t²(1-t²) ≤ 0, where t = rpm/rpm_dz.
        # Penalty-free operating points: rpm = 0 and |rpm| >= rpm_dz.
        # rpm_dz is set ABOVE the nominal hardware deadzone so the solver
        # commits to either 0 (no thrust) or beyond rpm_dz (real thrust),
        # never parking at the hardware boundary where thrust is ambiguous.
        # Tune rpm_dz based on the highest deadzone observed in practice.
        #
        # Gradient at rpm = 0 is zero (flat saddle), so the solver can freely
        # transit through 0 during direction switches (e.g. reverse thrust
        # for braking).  No penalty barrier blocks the sign change.
        #
        # SIM: reduced from 300 to 50.  The SAM_casadi dynamics used for sim
        # have a CONTINUOUS thrust curve — no physical deadzone — so the
        # "avoid ambiguous-thrust regime" rationale doesn't apply.  With
        # rpm_dz=300, any low-thrust RPM (e.g. ~100 for a gentle 0.1–0.2 m/s
        # cruise) cost ~3.8/stage soft penalty = ~114 over the horizon.
        # That pushed the solver to choose RPM=0 (coast on drag) over small
        # positive RPM commands — the direct cause of "stops and drifts at
        # the end of the mission, a small push would be enough".  With
        # rpm_dz=50 the constraint is effectively only active for RPM<50, so
        # any RPM ≥ 50 is free and the solver can command the small forward
        # bias the approach-to-goal phase needs.
        rpm_dz = 50.0  # SIM: effectively inactive for useful RPM ranges
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

        # Compute e_c_vec symbolically for the track constraint using the
        # LOCAL tangent (t_local_hat), not the lookahead tangent (t_hat) used
        # by the cost.  Reason: e_c_vec - e_l * t_hat is the perpendicular
        # distance from the vehicle to the line through p_ref in direction
        # t_hat.  When t_hat is the lookahead (a tangent ~1 m further down
        # the path), that line points away from the local path on curves,
        # which lets the vehicle drift far from the spline while still
        # satisfying ||e_c|| <= r_track.  t_local_hat is the tangent at
        # p_ref itself, so the perpendicular plane through p_ref is the
        # actual osculating-plane normal direction at first order — the
        # tube measured against it is a meaningful "distance to local path"
        # to within O(curvature * |theta - theta_hat|^2).
        _idx_t = self.nx + self.nu + 3
        _t_hat = self.model.p[_idx_t : _idx_t + 3]               # lookahead (cost)
        _theta_hat = self.model.p[_idx_t + 3]
        _t_local_hat = self.model.p[_idx_t + 4 : _idx_t + 7]      # local (constraint)
        _theta = self.model.x[self.N_PHYS_STATES]
        _p_ref = self.model.p[:3]
        # Cost-side e_c_vec (uses lookahead tangent — kept on self.model for
        # any external diagnostics that may grab it).
        _path_pos_cost = _p_ref + _t_hat * (_theta - _theta_hat)
        _pos_diff_cost = self.model.x[:3] - _path_pos_cost
        _e_l_cost = ca.dot(_pos_diff_cost, _t_hat)
        self.model.e_c_vec = _pos_diff_cost - _e_l_cost * _t_hat

        # Constraint-side e_c_vec (uses local tangent).  Note that the
        # perpendicular component is invariant to the parallel "slide" along
        # the tangent, so we evaluate it directly against p_ref instead of
        # along _t_local_hat (the (theta - theta_hat) shift would only add a
        # vector along _t_hat, which is not perpendicular to _t_local_hat
        # except on straight paths and would corrupt the tube on curves).
        _pos_diff_track = self.model.x[:3] - _p_ref
        _e_l_track = ca.dot(_pos_diff_track, _t_local_hat)
        _e_c_track_vec = _pos_diff_track - _e_l_track * _t_local_hat

        h_track = ca.dot(_e_c_track_vec, _e_c_track_vec)
        # SIM: 1.0 -> 0.5.  The 1.0 m tube was chosen to accommodate the
        # tight y=0.5 pivot where the vehicle had to swing wide through a
        # forward/reverse sequence and could deviate nearly a metre from
        # the linearized tangent during the maneuver.  That was a "worst-
        # case safety margin" tube, not a tracking tube — below 1 m the
        # only pull toward the path was Q_contour=700, which the solver
        # was willing to pay (e.g. trading 343/stage contour cost for
        # ~400/stage lag or heading-smoothing benefit on the dive curve),
        # so the vehicle routinely cruised 0.5–1.0 m off-path on the
        # descending straight leg.  User reported this as "track
        # constraints don't seem to be obeyed, overshoots quite a bit on
        # the dive".
        #
        # 0.5 m is a TRACKING tube: below that the contour cost still
        # dominates, but any excursion beyond 0.5 m now pays the soft
        # slack penalty (Z_track=1e5 quadratic + z_track=1e3 linear on
        # e_c²), which at ||e_c||=0.7 m gives an extra ~6000/stage on top
        # of the contour cost — enough to make even short-term excursions
        # expensive.  With the relaxed y=1.0 trajectory the pivot no
        # longer needs the 1 m safety margin (effective curvature radius
        # is ~0.5 m now, not 0.25 m).  If a future trajectory is tighter
        # and the pivot itself starts violating, either raise r_track
        # back toward 1.0 or use set_track_radius() to widen the tube
        # only for the pivot stages.
        #
        # NOTE: as of the t_local_hat split, the tube is measured against
        # the LOCAL path tangent (not the lookahead used by the cost), so
        # 0.5 m here now reflects the actual perpendicular distance from
        # the spline to first order in path curvature.  Previously the
        # constraint was measuring distance to a line ~1 m further down
        # the path, which on curves let the vehicle sit considerably
        # further from the spline than r_track suggested — that was the
        # "track constraint isn't keeping me within 0.5 m" pathology.
        self.r_track = 0.5  # [m] tracking tube — excursions beyond this bite the slack penalty
        self.IDX_TRACK = 3  # index of h_track inside con_h for dynamically updating track radius

        # ----- Wall-lock parameters ---------------------------------------------------
        # Once the vehicle moves past wall_lock_threshold (x), the x lower
        # box-constraint is raised to wall_lock_min to prevent drifting back
        # into the wall.  The constraint stays active for the rest of the mission.
        self.wall_lock_threshold = 1.7    # [m] x value that arms the lock
        self.wall_lock_min       = 1.5    # [m] enforced x lower bound once locked
        self.IDX_X_BOX           = 0      # index of x inside lbx/ubx vectors

        # ----- Depth-lock parameters --------------------------------------------------
        # Once the vehicle descends past depth_lock_threshold (NED z), the z lower
        # box-constraint is raised to depth_lock_min to prevent resurfacing.
        # The constraint stays active for the rest of the mission.
        self.depth_lock_threshold = 0.5   # [m] z value that arms the lock
        self.depth_lock_min       = 0.3   # [m] enforced z lower bound once locked
        self.IDX_Z_BOX            = 2     # index of z inside lbx/ubx vectors

        # ----- Pitch angle constraint ------------------------------------------------
        # Limit the vehicle pitch to ±pitch_max_deg by constraining sin(pitch).
        # sin(pitch) = -fwd_z = 2*(q0*q2 - q1*q3), a smooth polynomial in
        # quaternion components — avoids arcsin singularities and gives the SQP
        # well-behaved gradients everywhere.
        # SIM: raised back to 45° (real-vehicle used 30° to protect DR at
        # extreme angles; in sim there's no DR drift to worry about so let
        # the solver commit to steeper dives).
        self.pitch_max_deg = 45.0
        sin_pitch_max = np.sin(np.deg2rad(self.pitch_max_deg))
        q0_c = self.model.x[3]
        q1_c = self.model.x[4]
        q2_c = self.model.x[5]
        q3_c = self.model.x[6]
        h_pitch = 2.0 * (q0_c * q2_c - q1_c * q3_c)   # = sin(pitch)

        # con_h layout: [brake_h(1), h_dz1(1), h_dz2(1), h_track(1), h_pitch(1)]
        r_sq = self.r_track ** 2
        self.ocp.model.con_h_expr = ca.vertcat(brake_h, h_dz1, h_dz2, h_track, h_pitch)
        self.ocp.constraints.lh = np.array([-1e9, -1e9, -1e9, 0.0, -sin_pitch_max])
        self.ocp.constraints.uh = np.array([ 0.0,  0.0,  0.0, r_sq,  sin_pitch_max])
        self.ocp.constraints.idxsh = np.arange(5)
        n_sh = 5

        # Terminal nonlinear constraint (same structure)
        self.ocp.model.con_h_expr_e = ca.vertcat(brake_h, h_dz1, h_dz2, h_track, h_pitch)
        self.ocp.constraints.lh_e = np.array([-1e9, -1e9, -1e9, 0.0, -sin_pitch_max])
        self.ocp.constraints.uh_e = np.array([ 0.0,  0.0,  0.0, r_sq,  sin_pitch_max])
        self.ocp.constraints.idxsh_e = np.arange(5)
        n_sh_e = 5

        # ----- Unified slack penalty vectors ------------------------------------
        # acados orders slack variables as: [idxsbx | idxsh] for stage costs.
        # Terminal stage only has idxsh_e.
        Z_pos_x = 1e9 # quadratic penalty for x position box violation (end wall — hard to brake)
        Z_pos_y = 1e7   # quadratic penalty for y position box violation
        Z_pos_z = 1e9   # quadratic penalty for z position box violation (near-hard: protects depth lock)
        Z_surge = 1e6   # quadratic penalty for surge velocity violation
        z_pos_x = 1e7   # linear   penalty for x position box violation (end wall)
        z_pos_y = 1e5   # linear   penalty for y position box violation
        z_pos_z = 1e6   # linear   penalty for z position box violation (near-hard: protects depth lock)
        z_surge = 1e4   # linear   penalty for surge velocity violation

        # Brake penalty: with a_brake = 0.009 the funnel bites at ~1.5 m from
        # the goal.  Strong penalties force the solver to actively brake
        # (reverse thrust) rather than coast on drag — same mechanism that
        # makes the solver brake near the wall constraints.
        Z_brake = 1e5    # quadratic penalty for braking funnel violations
        z_brake = 1e3    # linear   penalty for braking funnel violations

        Z_dz    = 200.0 # quadratic penalty for RPM deadzone
        z_dz    = 20.0  # linear   penalty for RPM deadzone

        Z_track = 1e7   # quadratic penalty for track tube violation
        z_track = 1e5   # linear   penalty for track tube violation

        Z_pitch = 1e7   # quadratic penalty for pitch limit violation (softened to prevent solver crashes on transient overshoots)
        z_pitch = 1e5   # linear   penalty for pitch limit violation

        # Stage: [sbx_x(1), sbx_y(1), sbx_z(1), sbx_surge(1), sh_brake(1), sh_dz(2), sh_track(1), sh_pitch(1)] = size 9
        Zl_sbx = np.array([Z_pos_x, Z_pos_y, Z_pos_z, Z_surge])
        zl_sbx = np.array([z_pos_x, z_pos_y, z_pos_z, z_surge])
        self.ocp.cost.Zl = np.r_[Zl_sbx, Z_brake, Z_dz, Z_dz, Z_track, Z_pitch]
        self.ocp.cost.Zu = np.r_[Zl_sbx, Z_brake, Z_dz, Z_dz, Z_track, Z_pitch]
        self.ocp.cost.zl = np.r_[zl_sbx, z_brake, z_dz, z_dz, z_track, z_pitch]
        self.ocp.cost.zu = np.r_[zl_sbx, z_brake, z_dz, z_dz, z_track, z_pitch]

        # Terminal: [sbx_e_x(1), sbx_e_y(1), sbx_e_z(1), sbx_e_surge(1), sh_e_brake(1), sh_e_dz(2), sh_e_track(1), sh_e_pitch(1)] = size 9
        self.ocp.cost.Zl_e = np.r_[Zl_sbx, Z_brake, Z_dz, Z_dz, Z_track, Z_pitch]
        self.ocp.cost.Zu_e = np.r_[Zl_sbx, Z_brake, Z_dz, Z_dz, Z_track, Z_pitch]
        self.ocp.cost.zl_e = np.r_[zl_sbx, z_brake, z_dz, z_dz, z_track, z_pitch]
        self.ocp.cost.zu_e = np.r_[zl_sbx, z_brake, z_dz, z_dz, z_track, z_pitch]

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
        self.ocp.solver_options.levenberg_marquardt = 1e-2 # before: 1e-2
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
          [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1),
           e_rudder_auth(1), e_stern_auth(1), yaw_rate(1), u_phys(6),
           delta_v_theta(1)] = 18

        Terminal (terminal=True):
          [e_c_vec(3), e_l(1), v_theta(1), e_heading(1), e_pitch(1), e_sync(1),
           e_rudder_auth(1), e_stern_auth(1), yaw_rate(1), heave_vel(1)] = 12

        Direction convention: e_sync (|v_along|) and e_pitch (smooth-signed)
        are invariant under vehicle-heading reversal (fwd -> -fwd), so
        reverse surge through a dive is dynamically and sync-consistent.
        e_heading is NOT symmetric on purpose: atan2 gives a monotonic,
        single-basin cost over (-π, π] that strongly pins the vehicle
        forward along the path tangent.  A symmetric heading residual
        (tried briefly as sin(yaw_err), with a Q_fwd_bias tiebreaker) was
        removed because it produced a bifurcated cost landscape the SQP
        warm-start picks non-deterministically — the return leg either
        over-rotated to the reverse basin (270° overshoot) or the pivot
        couldn't hop into the reverse basin cheaply enough.  K-turn
        emergence in the current formulation depends on e_l / e_c_vec
        cost overwhelming e_heading at very tight curvatures; Q_lag is
        the knob if the tightest turns don't clear.  For anything that
        won't clear with this cost alone, shape the trajectory instead
        (e.g. a wider tangent-matched arc at the pivot, or K-turn
        waypoints made explicit in the path).

        p vector layout (set in DiveControllerMPC.update):
          p[0 : nx]              = state ref     (nx = 21)
          p[nx : nx+nu]          = control ref   (nu = 7)
          p[nx+nu : nx+nu+3]     = goal_pos      (3)
          p[nx+nu+3 : nx+nu+6]   = t_hat         (3)  — lookahead tangent (cost)
          p[nx+nu+6]             = theta_hat     (1)
          p[nx+nu+7 : nx+nu+10]  = t_local_hat   (3)  — local tangent (track constraint)
          Total = 21 + 7 + 3 + 3 + 1 + 3 = 38
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
        # t_local_hat is still parsed here (used by the track constraint
        # elsewhere in this file), but compute_stage_cost does NOT use it
        # for attitude or sync residuals any more — see the big comment
        # below for the history of that decision.
        theta = x[self.N_PHYS_STATES]          # x[19]

        # All cost residuals below use t_hat (the lookahead tangent).
        # heading_offset is tuned in DiveControllerMPC so that the
        # lookahead distance is short enough NOT to reach past the next
        # tight curve from the dive approach (see the matching comment
        # there).
        path_pos = p_ref + t_hat * (theta - theta_hat)
        pos_diff = x[:3] - path_pos
        e_l = ca.dot(pos_diff, t_hat)
        e_c_vec = pos_diff - e_l * t_hat

        # Vehicle forward axis from quaternion (first column of rotation matrix)
        q0_s, q1_s, q2_s, q3_s = x[3], x[4], x[5], x[6]
        fwd_x = 1 - 2 * (q2_s**2 + q3_s**2)
        fwd_y = 2 * (q1_s * q2_s + q0_s * q3_s)
        fwd_z = 2 * (q1_s * q3_s - q0_s * q2_s)
        # SIM: attitude/sync residuals use t_hat (lookahead tangent).
        # Brief history: an earlier revision switched these to t_local_hat
        # (tangent at the linearization point, no lookahead) because the
        # lookahead form was driving premature rotation on the dive
        # approach of gentle_dive_test_wide_2: at theta ≈ 5.88 (0.4 m
        # before the pivot start at s = 6.28), stage 0's t_hat sampled
        # the y-spline derivative at s = 6.88 (inside the +Y pivot leg),
        # giving |t_hat_y| ≈ 0.999.  That made yaw_ref swing to +90°
        # while the vehicle was still on the straight dive.  The
        # t_local_hat switch fixed that but introduced two new failure
        # modes on the same trajectory: (i) tail-first reverse cruise
        # on the return leg (cos_align agnostic about surge sign once
        # combined with the earlier magnitude-based e_sync), (ii)
        # a transient reverse-surge at the very start of the dive where
        # the projection-based t_local at theta ≈ 0 gave the solver a
        # locally preferred reverse initialisation.  Reverting to t_hat
        # here AND shrinking heading_offset from 1.0 → 0.3 in
        # DiveControllerMPC.get_current_ref_array keeps the premature-
        # rotation fix while not perturbing the rest of the tuning
        # (surge_min, Q_sync, Q_heading balances were all chosen under
        # the t_hat form).  e_sync stays signed (see below).
        cos_align = fwd_x * t_hat[0] + fwd_y * t_hat[1] + fwd_z * t_hat[2]
        # Yaw-only heading: atan2(cross, dot) of horizontal projections.
        # Returns the signed yaw error in radians, decoupled from pitch.
        # Monotonic over (-π, π] with a single minimum at yaw_err = 0 and
        # maximum at ±π (reverse-aligned, cost Q_heading·π² ≈ 3000/stage).
        # This is STRONGLY asymmetric on purpose: the alternative tried
        # briefly — e_heading = h_cross = sin(yaw_err), symmetric between
        # forward and reverse — produced a bifurcated cost landscape
        # (two zero-cost basins separated by a ridge at yaw_err = ±π/2)
        # whose basin selection was decided non-deterministically by the
        # SQP warm-start.  That manifested as either 270° over-rotation
        # on the return leg (solver fell into the reverse basin during a
        # lateral-correction overshoot) or failure to pivot at all (no
        # Q_fwd_bias tiebreaker weight could simultaneously satisfy both
        # cases, since the ridge disappears only at Q_fwd_bias ≥
        # Q_heading/2 at which point K-turns cost 600/stage).  The atan2
        # form trades K-turn emergence for predictability; sharp-turn
        # trajectories should be shaped geometrically (wider arc, or
        # explicit K-turn waypoints) rather than relying on the cost to
        # invert direction autonomously.
        h_dot = fwd_x * t_hat[0] + fwd_y * t_hat[1]
        h_cross = fwd_x * t_hat[1] - fwd_y * t_hat[0]
        e_heading = ca.atan2(h_cross, h_dot)

        # SIM: direction-symmetric pitch residual.  smooth_sign is a smooth
        # approximation of sign(cos_align) — +1 when vehicle is forward-
        # aligned with the path, -1 when reverse-aligned, continuous through
        # 0 (the ε regulariser keeps the Jacobian bounded).  Meaning:
        #   forward leg (cos_align ≈ +1):  e_pitch = (-fwd_z) - (-t_hat[2])
        #     — match nose pitch to path pitch (original behaviour).
        #   reverse leg (cos_align ≈ -1):  e_pitch = -(-fwd_z) - (-t_hat[2])
        #     — nose pitch inverts relative to path pitch: on a descending
        #     path traversed backwards, the nose correctly points UP out
        #     of the dive.
        # Previously `e_pitch = (-fwd_z) - (-t_hat[2])` with no smooth_sign,
        # which forced nose-down on any descending tangent and made reverse
        # legs through dives dynamically incoherent.
        smooth_sign = cos_align / ca.sqrt(cos_align**2 + 1e-4)
        e_pitch = smooth_sign * (-fwd_z) - (-t_hat[2])

        # v_theta: set yref[4] = v_target to pull progress speed toward v_target.
        v_theta = x[self.N_PHYS_STATES + 1]   # x[20]

        # SIM: SIGNED progress/velocity synchronization.  Open design
        # question: the signed form e_sync = v_theta - u·cos_align has a
        # cos_align-dependent bifurcation.  With v_theta >= 0 and
        # u ∈ [-0.4, +0.4]:
        #   cos_align > 0 (forward-aligned): minimum at u > 0.
        #   cos_align < 0 (reverse-aligned): minimum at u < 0.
        # That second basin is intentional for trajectories that include
        # reverse-thrust segments (the vehicle is supposed to be able to
        # drive tail-first when the path requires it), but it also
        # creates a trap: any transient over-rotation past 90° off-
        # tangent drops the solver into the reverse basin, and rudder
        # authority flips sign at u < 0, so the solver cannot easily
        # rotate back.  Seen on gentle_dive_test_wide_2 row 691:
        # yaw_err=-113°, cos_align=-0.35, RPM=-500, sustained ~17 s
        # drifting 1.27 m off path.
        # The magnitude form (|v_along|) removes the trap but also
        # removes the forward/tail-first gradient during forward
        # cruise, letting actuator inertia pick direction post-pivot.
        # Both failure modes are well-documented on this trajectory;
        # the right fix is probably a direction-aware term (path
        # segments flagged as "forward" vs "reverse" in the trajectory
        # definition), not a scalar tweak to this residual.
        v_along = x[7] * cos_align
        e_sync = v_theta - v_along

        # SIM: direct rudder / stern angle penalty.
        # Previously this was `x[16] * exp(-3·rpm²/rpm_dz²)` — a steering-
        # "authority" gating that was meant to penalise thrust-vectoring
        # deflection when RPMs were too low to generate prop wash.  At
        # cruise RPM that gate collapses the penalty by ~20×, which in sim
        # meant rudder angle was effectively unpenalised and the solver
        # parked the rudder at ±7° during even mild overshoots (rate
        # penalty alone can't fix this: once saturated, holding it costs
        # nothing).  Using the raw state gives a proper quadratic cost on
        # deflection and damps the oscillation.  The variable names are
        # kept so the Q_diag indices stay readable.
        e_rudder_auth = x[16]
        e_stern_auth = x[15]

        # Body yaw rate (x[12] = r) feeds a direct Q_r * r² penalty — see
        # the Q_diag comment on the 11th entry for the sizing rationale.
        # Deliberately NOT using a reference (so yref[10] stays 0): we want
        # to penalise magnitude, not track a target yaw rate.
        yaw_rate = x[12]

        if terminal:
            return ca.vertcat(
                e_c_vec, e_l, v_theta, e_heading, e_pitch, e_sync,
                e_rudder_auth, e_stern_auth,
                yaw_rate,
                x[9],
            )
        else:
            return ca.vertcat(
                e_c_vec, e_l, v_theta, e_heading, e_pitch, e_sync,
                e_rudder_auth, e_stern_auth,
                yaw_rate,
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
