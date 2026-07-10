import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the parent folder to the python path to load the project modules
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

from dynamics import dynamics, step
from solver_ltv_lqr import backward_riccati
import data
import control as ctrl

# 1. Load optimal trajectory from Task 2
data_path = os.path.join(project_dir, 'data', 'optimal_trajectory_task2.npy')
try:
    data_dict = np.load(data_path, allow_pickle=True).item()
    x_ref_raw = data_dict['x']
    u_ref_raw = data_dict['u']
    t_axis = data_dict['t']
    
    ns, ni = data.ns, data.ni
    if x_ref_raw.ndim == 3:
        x_ref_raw = x_ref_raw[:, :, -1]
    x_ref_traj = x_ref_raw.T if x_ref_raw.shape[0] == ns else x_ref_raw
    u_ref_traj = u_ref_raw.T if u_ref_raw.shape[0] == ni else u_ref_raw
    
    if u_ref_traj.ndim == 1:
        u_ref_traj = u_ref_traj.reshape(-1, 1)
        
    steps = x_ref_traj.shape[0] - 1
    print(f"Loaded reference trajectory with {steps} steps.")
except FileNotFoundError:
    print(f"Error: {data_path} not found. Please run task2_main.py first.")
    sys.exit(1)

# 2. Linearization and LQR Gain Design
A_list, B_list = [], []
for t in range(steps):
    _, A, B = dynamics(x_ref_traj[t], u_ref_traj[t].flatten())
    A_list.append(A)
    B_list.append(B)

x_goal = x_ref_traj[-1]
u_goal = u_ref_traj[-1]
_, A_eq, B_eq = dynamics(x_goal, u_goal.flatten())
Q_lqr = data.Q_track
R_lqr = data.R_track
QQf = ctrl.dare(A_eq, B_eq, Q_lqr, R_lqr)[0]

K_gains, _ = backward_riccati(A_list, B_list, Q_lqr, R_lqr, QQf, steps)
print("TV-LQR Gains computed.")

# 3. Stability Sweep
# We define a grid of initial shoulder and elbow perturbations
grid_n = 31
d_theta1_range = np.linspace(-1.2, 1.2, grid_n)
d_theta2_range = np.linspace(-1.2, 1.2, grid_n)

stability_grid = np.zeros((grid_n, grid_n))
final_errors = np.zeros((grid_n, grid_n))

print("Running stability sweep...")
for j in range(grid_n):
    d_th2 = d_theta2_range[j]
    for i in range(grid_n):
        d_th1 = d_theta1_range[i]
        
        pert = np.array([d_th1, d_th2, 0.0, 0.0])
        x_curr = x_ref_traj[0] + pert
        
        diverged = False
        for t in range(steps):
            delta_x = x_curr - x_ref_traj[t]
            delta_u = -K_gains[t] @ delta_x
            u_curr = u_ref_traj[t] + delta_u
            
            x_curr = step(x_curr, u_curr.flatten())
            
            # Check for divergence or numerical instability
            if np.any(np.isnan(x_curr)) or np.any(np.isinf(x_curr)) or np.any(np.abs(x_curr) > 100):
                diverged = True
                break
                
        if diverged:
            stability_grid[i, j] = 0  # Unstable
            final_errors[i, j] = np.inf
        else:
            err_final = np.linalg.norm(x_curr - x_goal)
            final_errors[i, j] = err_final
            if err_final < 0.1:  # Converged to goal within 0.1 rad
                stability_grid[i, j] = 1  # Stable
            else:
                stability_grid[i, j] = 0.5  # Bounded but did not converge/high error

# 4. Plot results
plt.figure(figsize=(9, 7))
# Create meshgrid for plotting (x is elbow perturbation, y is shoulder perturbation)
X, Y = np.meshgrid(np.degrees(d_theta2_range), np.degrees(d_theta1_range))
plt.contourf(X, Y, stability_grid, levels=[-0.5, 0.25, 0.75, 1.5], colors=['#ff9999', '#ffffb3', '#99ff99'])
# Add labels
plt.title("Acrobot TV-LQR Stability Region (Region of Attraction)")
plt.xlabel(r"Elbow angle perturbation $\delta \theta_2$ [deg]")
plt.ylabel(r"Shoulder angle perturbation $\delta \theta_1$ [deg]")
plt.grid(True, alpha=0.3)

# Add custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#99ff99', edgecolor='g', label='Stable / Converged (Error < 0.1 rad)'),
    Patch(facecolor='#ffffb3', edgecolor='y', label='Bounded (No divergence, high error)'),
    Patch(facecolor='#ff9999', edgecolor='r', label='Unstable / Diverged')
]
plt.legend(handles=legend_elements, loc='upper right')

# Save plot
fig_path = os.path.join(project_dir, 'figs', 'stability_region_sweep.png')
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path, dpi=300)
print(f"Stability map plotted and saved to: {fig_path}")
