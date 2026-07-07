#
# Task 3 — Trajectory Tracking via TV-LQR (Time-Varying LQR)
# Optimal Control Project — Parameter Set 3
#

import numpy as np
import matplotlib.pyplot as plt
import signal
import os

# Ensure plots don't block Ctrl+C
signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 13})

from dynamics        import dynamics, step
from solver_ltv_lqr  import backward_riccati
import data
import animation as an

# =============================================================================
# SECTION 1 — CONFIGURATION & SETTINGS
# =============================================================================
print("=" * 60)
print("   Task 3: Trajectory Tracking — TV-LQR")
print("=" * 60)

ns, ni = data.ns, data.ni

# Toggle for rendering the visual animation at the end
SHOW_ANIMATION = True  

# TV-LQR Weights
Q_lqr = data.Q_track   # state weight
R_lqr = data.R_track   # input weight

# =============================================================================
# SECTION 2 — REFERENCE TRAJECTORY LOADING
# =============================================================================
print("\nLoading reference trajectory (Task 2)...")
try:
    data_dict    = np.load('data/optimal_trajectory_task2.npy', allow_pickle=True).item()
    x_ref_raw    = data_dict['x']
    u_ref_raw    = data_dict['u']
    t_axis       = data_dict['t']
    QQT_dare     = data_dict.get('QQT', None)

    #  dimension formatting
    if x_ref_raw.ndim == 3:
        x_ref_raw = x_ref_raw[:, :, -1]
    x_ref_traj = x_ref_raw.T if x_ref_raw.shape[0] == ns else x_ref_raw
    u_ref_traj = u_ref_raw.T if u_ref_raw.shape[0] == ni else u_ref_raw

    if u_ref_traj.ndim == 1:
        u_ref_traj = u_ref_traj.reshape(-1, 1)

    steps = x_ref_traj.shape[0] - 1  # 2999 transitions for 3000 states
    print(f"  Trajectory loaded: {steps} steps,  T = {t_axis[-1]:.1f}s")

except FileNotFoundError:
    print("ERROR: 'optimal_trajectory_task2.npy' not found. Run task2_main.py first.")
    exit()

# =============================================================================
# SECTION 3 — LINEARIZATION ALONG REFERENCE TRAJECTORY
# =============================================================================
print("\nLinearizing the system along reference trajectory...")
A_list, B_list = [], []
for t in range(steps):
    _, A, B = dynamics(x_ref_traj[t], u_ref_traj[t].flatten())
    A_list.append(A)
    B_list.append(B)
print(f"  Linearization complete: {steps} pairs (A_t, B_t)")

# =============================================================================
# SECTION 4 — BACKWARD RICCATI (TV-LQR Design)
# =============================================================================
# Compute the DARE terminal cost for tracking using the tracking weights
import control as ctrl
A_eq, B_eq = A_list[-1], B_list[-1]
QQT_track = ctrl.dare(A_eq, B_eq, Q_lqr, R_lqr)[0]
QQf = QQT_track
print(f"\nUsing terminal cost Q_T from TV-LQR tracking DARE calculation.")

print("Computing Riccati backward (TV-LQR)...")
K_gains, P_list = backward_riccati(A_list, B_list, Q_lqr, R_lqr, QQf, steps)
print("  Backward Riccati complete.")

# =============================================================================
# SECTION 5 — TRACKING SIMULATION WITH MULTIPLE PERTURBATIONS
# =============================================================================
perturbations = {
    'Pert. shoulder -0.2 rad': np.array([-0.2,  0.0, 0.0, 0.0]),
    'Pert. elbow +0.3 rad'   : np.array([ 0.0,  0.3, 0.0, 0.0]),
    'Pert. vel. shoulder'    : np.array([ 0.0,  0.0, 0.3, 0.0]),
}

results = {}
print("\nSimulating LQR tracking with multiple perturbations...")

for label, pert in perturbations.items():
    x_sim = np.zeros((steps + 1, ns))
    u_sim = np.zeros((steps + 1, ni))  # size TT for plotting
    x_sim[0] = x_ref_traj[0] + pert

    for t in range(steps):
        delta_x = x_sim[t] - x_ref_traj[t]
        delta_u = -K_gains[t] @ delta_x
        u_curr  = u_ref_traj[t] + delta_u

        u_sim[t] = u_curr.flatten()
        x_sim[t+1] = step(x_sim[t], u_curr.flatten())

    u_sim[steps] = u_sim[steps - 1]  # Zero-Order Hold for the last step

    err_final = np.linalg.norm(x_sim[-1] - x_ref_traj[-1])
    print(f"  [{label}]: final error ||x_T - x*_T|| = {err_final:.4e}")
    results[label] = {'x_sim': x_sim, 'u_sim': u_sim}

# =============================================================================
# SECTION 5.5 — SAVE DATA
# =============================================================================
np.save('data/lqr_data_task3.npy', {
    'K_gains': K_gains,
    'P_list' : P_list,
    'Q_lqr'  : Q_lqr,
    'R_lqr'  : R_lqr,
    'QQf'    : QQf
})
print("\nLQR gains and Riccati matrices P_t saved to 'lqr_data_task3.npy'")

# =============================================================================
# SECTION 6 — PLOT RESULTS
# =============================================================================
os.makedirs('figs', exist_ok=True)
time = t_axis

# --- PLOT 1: COMPACT TRACKING ERRORS (All perturbations compared together) ---
fig_err, axs_err = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig_err.suptitle('Task 3 — Tracking Errors for Different Initial Conditions', fontsize=14, fontweight='bold')
colors = ['blue', 'orange', 'green']

for idx, (label, res) in enumerate(results.items()):
    x_sim_plot = res['x_sim']
    err_pos1 = np.degrees(x_sim_plot[:, 0] - x_ref_traj[:, 0])
    err_pos2 = np.degrees(x_sim_plot[:, 1] - x_ref_traj[:, 1])

    axs_err[0].plot(time, err_pos1, color=colors[idx], lw=2, label=label)
    axs_err[1].plot(time, err_pos2, color=colors[idx], lw=2, label=label)

axs_err[0].set_ylabel(r'Err $\theta_1$ [deg]')
axs_err[0].grid(True, alpha=0.3)
axs_err[0].legend(loc='best', fontsize='small')
axs_err[1].set_ylabel(r'Err $\theta_2$ [deg]')
axs_err[1].set_xlabel('Time [s]')
axs_err[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figs/task3_tracking_errors.png', dpi=300)
plt.show(block=False)


# --- PLOT 2: NOMINAL VS OPTIMAL TRACKING (Shown for the shoulder perturbation) ---
primary_label = 'Pert. shoulder -0.2 rad'
primary_res = results[primary_label]
x_sim_primary = primary_res['x_sim']
u_sim_primary = primary_res['u_sim']

fig_traj, axs_traj = plt.subplots(ns + ni, 1, figsize=(11, 10), sharex=True)
fig_traj.suptitle(f'Task 3 — State & Input Tracking ({primary_label})', fontsize=14, fontweight='bold')

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
state_colors = ['blue', 'cyan', 'green', 'purple']

for i in range(ns):
    axs_traj[i].plot(time, x_sim_primary[:, i], color=state_colors[i], lw=2, label='Actual (LQR)')
    axs_traj[i].plot(time, x_ref_traj[:, i], color='black', lw=1.5, ls='--', label='Reference')
    axs_traj[i].set_ylabel(labels_x[i])
    axs_traj[i].legend(loc='best', fontsize=9)
    axs_traj[i].grid(alpha=0.3)

axs_traj[ns].plot(time, u_sim_primary[:, 0], color='red', lw=2, label='Actual (LQR)')
axs_traj[ns].plot(time[:-1], u_ref_traj[:, 0], color='orange', lw=1.5, ls='--', label='Reference')
axs_traj[ns].set_ylabel(r'$\tau$ [Nm]')
axs_traj[ns].set_xlabel('Time [s]')
axs_traj[ns].legend(loc='best', fontsize=9)
axs_traj[ns].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figs/task3_state_input_tracking.png', dpi=300)

# Set the last plot call to block=False so that the script doesn't block headless execution
plt.show(block=False)

# =============================================================================
# SECTION 7 — ANIMATION (Animate only the first perturbation if enabled)
# =============================================================================
if SHOW_ANIMATION:
    print(f"\nStarting animation for: {primary_label}")
    # Transpose arrays for animation (4, T)
    an.animate_trajectory(time, x_sim_primary.T, x_ref_traj.T, title=f"Task 3 LQR Tracking: {primary_label}")


