#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from smarc_modelling.vehicles.SAM_PIML import SAM_PIML
from smarc_modelling.piml.utils.utility_functions import load_data_from_bag, eta_quat_to_deg
import matplotlib.pyplot as plt
import torch
import scienceplots # For fancy plotting


class SIM:
    """Simulator for SAM / other UAVs"""

    def __init__(self, piml_type: str, states: list, time_vec: list, control_vec: list, state_update: bool):

        # Initial pose
        self.x0 = torch.Tensor.tolist(torch.cat([states[0][0], states[1][0], states[2][0]]))

        # Create vehicle instance
        self.vehicle = SAM_PIML(dt=0.01, piml_type=piml_type)

        # Controls and sim variables
        self.controls = control_vec
        self.n_sim = np.shape(time_vec)[0]
        self.var_dt = np.diff(time_vec)

        # Decide if we are going to update the state or not
        self.state_update = state_update

    def run_sim(self):
        print(f" Running simulator...")
        
        # For storing results of sim
        data = np.empty((len(self.x0), self.n_sim))
        data[:, 0] = self.x0

        # For getting last value before quat errors
        end_val = 10e10
        once = True
        time_since_update = 0
        times =[]
        
        for i in range(self.n_sim-1):

            if i % 3 == 0 and self.state_update:
                data[:, i] = torch.Tensor.tolist(torch.cat([states[0][i], states[1][i], states[2][i]]))
                times.append(time_since_update)
                time_since_update = 0

            # Get the current time step
            dt = self.var_dt[i]
            self.vehicle.update_dt(dt)
            time_since_update += dt

            # Do sim step using rk4 or ef
            try:
                data[:, i+1] = self.rk4(data[:, i], self.controls[i], dt, self.vehicle.dynamics)
            except:
                data[:, i+1] = data[:, i-10]
                if once:
                    once = False
                    end_val = i-10

        if self.state_update:
            print(f" Average times between resets: {np.mean(times)}")

        return data, end_val
    
    def rk4(self, x, u, dt, fun):
        # https://github.com/smarc-project/smarc_modelling/blob/master/src/smarc_modelling/sam_sim.py#L38C1-L46C15 
        # Runge Kutta 4
        k1 = fun(x, u)
        k2 = fun(x+dt/2*k1, u)
        k3 = fun(x+dt/2*k2, u)
        k4 = fun(x+dt*k3, u)
        dxdt = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return dxdt
    
    def ef(self, x, u, dt, fun):
        # Euler forward
        dxdt = x + dt * fun(x, u)
        return dxdt
    
if __name__ == "__main__":
    print(f" Starting simulator...")

    # Loading ground truth data
    eta, nu, u_fb, u_cmd, Dv_comp, Mv_dot, Cv, g_eta, tau, t, M, nu_dot = load_data_from_bag("src/smarc_modelling/piml/data/rosbags/rosbag_twosecs", "torch")
    states = [eta, nu, u_fb]

    # Initial positions used for flipping coordinate frames later
    x0 = eta[0, 0].item()
    y0 = eta[0, 1].item()
    z0 = eta[0, 2].item()

    # Setting up model for simulations
    reset_state = True
    sam_wb = SIM(None, states, t, u_cmd, reset_state) # White-box sim
    sam_pinn = SIM("pinn", states, t, u_cmd, reset_state) # Physics Informed Neural Network sim
    
    # Running the simulators
    print(f" Running white-box simulation...")
    results_wb, end_val_wb = sam_wb.run_sim()
    results_wb = torch.tensor(results_wb).T
    eta_wb = results_wb[:, 0:7]
    eta_wb[:, 0] = 2 * x0 - eta_wb[:, 0] # Flipping to NED frame
    eta_wb[:, 2] = 2 * z0 - eta_wb[:, 2]
    nu_wb = results_wb[:, 7:13]
    print(f" Done with the white-box sim!")

    print(f" Running PINN simulation...")
    results_pinn, end_val_pinn = sam_pinn.run_sim()
    results_pinn = torch.tensor(results_pinn).T
    eta_pinn = results_pinn[:, 0:7]
    eta_pinn[:, 0] = 2 * x0 - eta_pinn[:, 0] # Flipping to NED frame
    eta_pinn[:, 2] = 2 * z0 - eta_pinn[:, 2]
    nu_pinn = results_pinn[:, 7:13]
    print(f" Done with the PINN sim!")

    print(f" Done with all sims making plots!")

    # Flipping gt into NED frame for plots instead of ENU
    eta[:, 0] = 2 * x0 - eta[:, 0]
    #eta[:, 1] = 2 * y0 - eta[:, 1]
    eta[:, 2] = 2 * z0 - eta[:, 2] 

    end_val = int(np.min([end_val_wb, end_val_pinn]))
    print(end_val)

    # 3D trajectory plot
    if True:
        # Plotting trajectory in 3d
        plt.style.use('science')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for vector, label in zip([eta, eta_wb, eta_pinn], ["Ground Truth", "White-Box", "PINN"]):
            # Plotting trajectory
            vector = np.array(vector)
            points = vector[:end_val, :3].T
            ax.plot(points[0], points[1], points[2], label=label)

        # Equal axis scaling
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        max_range = max(x_range, y_range, z_range)

        # Midpoints
        x_middle = 0.5 * (xlim[0] + xlim[1])
        y_middle = 0.5 * (ylim[0] + ylim[1])
        z_middle = 0.5 * (zlim[0] + zlim[1])

        # Set new limits
        ax.set_xlim(x_middle - max_range/2, x_middle + max_range/2)
        ax.set_ylim(y_middle - max_range/2, y_middle + max_range/2)
        ax.set_zlim(z_middle - max_range/2, z_middle + max_range/2)

        # Axis labels
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        plt.legend()

    # Cumulative error plots
    if True:
        # Quat to deg
        eta_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta[:end_val]])
        eta_wb_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_wb[:end_val]])
        eta_pinn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_pinn[:end_val]])

        # Errors WB
        eta_wb_mse = (eta_wb_deg - eta_deg)**2
        nu_wb_mse = (nu_wb - nu)**2

        # Errors PINN
        eta_pinn_mse = (eta_pinn_deg - eta_deg)**2
        nu_pinn_mse = (nu_pinn - nu)**2

        # Cumulative error
        eta_wb_error = np.cumsum(eta_wb_mse, axis=0)
        nu_wb_error = np.cumsum(nu_wb_mse, axis=0)
        eta_pinn_error = np.cumsum(eta_pinn_mse, axis=0)
        nu_pinn_error = np.cumsum(nu_pinn_mse, axis=0)

        fig, axes = plt.subplots(4, 3, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "roll", "pitch", "yaw"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]

        # Plotting error in eta
        for i in range(6):
            axes[i].plot(eta_wb_error[:end_val, i], label="White-box")
            axes[i].plot(eta_pinn_error[:end_val, i], label="PINN")
            axes[i].set_title(f"{labels_eta[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("Cumulative Error")
            axes[i].legend()

        # Plot nu errors (next 6 plots)
        for i, j in enumerate([0, 2, 1, 3, 4, 5]):
            axes[j+6].plot(nu_wb_error[:end_val, i], label="White-box")
            axes[j+6].plot(nu_pinn_error[:end_val, i], label="PINN")
            axes[j+6].set_title(f"{labels_nu[i]}")
            axes[j+6].set_xlabel("Timestep")
            axes[j+6].set_ylabel("Cumulative Error")
            axes[j+6].legend()

        plt.tight_layout()

    # Plots of each state
    if True:
        # Quat to deg
        eta_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta[:end_val]])
        eta_wb_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_wb[:end_val]])
        eta_pinn_deg = np.array([eta_quat_to_deg(eta_vec) for eta_vec in eta_pinn[:end_val]])

        fig, axes = plt.subplots(4, 3, figsize=(12, 10))
        axes = axes.flatten()

        labels_eta = ["x", "y", "z", "roll", "pitch", "yaw"]
        labels_nu = ["u", "v", "w", "p", "q", "r"]

        # Plotting error in eta
        for i in range(6):
            axes[i].plot(eta_deg[:end_val, i], label="Ground Truth")
            axes[i].plot(eta_wb_deg[:end_val, i], label="White-box")
            axes[i].plot(eta_pinn_deg[:end_val, i], label="PINN")
            axes[i].set_title(f"{labels_eta[i]}")
            axes[i].set_xlabel("Timestep")
            axes[i].set_ylabel("State")
            axes[i].legend()

        # Plot nu errors (next 6 plots)
        for i, j in enumerate([0, 2, 1, 3, 4, 5]):
            axes[i+6].plot(nu[:end_val, j], label="Ground Truth")
            axes[i+6].plot(nu_wb[:end_val, j], label="White-box")
            axes[i+6].plot(nu_pinn[:end_val, j], label="PINN")
            axes[i+6].set_title(f"{labels_nu[i]}")
            axes[i+6].set_xlabel("Timestep")
            axes[i+6].set_ylabel("State")
            axes[i+6].legend()




    # Displaying plots
    try:
        plt.show()
    except KeyboardInterrupt:
        plt.close("all")