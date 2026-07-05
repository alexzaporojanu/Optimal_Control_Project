#
# Task 3 — Trajectory Tracking via TV-LQR (Time-Varying LQR)
# Optimal Control Project — Parameter Set 3
#

import numpy as np
import matplotlib.pyplot as plt
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 13})

from dynamics        import dynamics, step
from solver_ltv_lqr  import backward_riccati   
import animation as an

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================
print("=" * 60)
print("   Task 3: Trajectory Tracking — TV-LQR")
print("=" * 60)

ns, ni = 4, 1

# TV-LQR Weights
Q_lqr = np.diag([1000.0, 1000.0, 100.0, 100.0])   # state weight (stage)
R_lqr = np.eye(ni) * 0.1                             # input weight

# =============================================================================
# SECTION 2 — REFERENCE TRAJECTORY LOADING
# =============================================================================
print("\nLoading reference trajectory (Task 2)...")
try:
    data_dict         = np.load('data/optimal_trajectory_task2.npy', allow_pickle=True).item()
    x_ref_raw    = data_dict['x']    # (4, TT) or (TT, 4) — handle both
    u_ref_raw    = data_dict['u']    # (1, TT) or (TT, 1)
    t_axis       = data_dict['t']    # (TT,)
    QQT_dare     = data_dict.get('QQT', None)   # DARE matrix (if saved from Task 2)

    # ---- ROBUST DIMENSION FIX ----
    if x_ref_raw.ndim == 3:
        x_ref_raw = x_ref_raw[:,:,-1]    # Take last iteration if 3D
    if x_ref_raw.shape[0] == ns and x_ref_raw.shape[1] != ns:
        x_ref_traj = x_ref_raw.T         # (TT, 4)
        u_ref_traj = u_ref_raw.T if u_ref_raw.shape[0] == ni else u_ref_raw
    else:
        x_ref_traj = x_ref_raw           # already (TT, 4)
        u_ref_traj = u_ref_raw

    if u_ref_traj.ndim == 1:
        u_ref_traj = u_ref_traj.reshape(-1, 1)

    steps = x_ref_traj.shape[0]

    if x_ref_traj.shape[1] != 4:
        raise ValueError(f"x_ref has shape {x_ref_traj.shape}, expected (TT, 4)")

    print(f"  Trajectory loaded: {steps} steps,  T = {t_axis[-1]:.1f}s")

except FileNotFoundError:
    print("ERROR: 'optimal_trajectory_task2.npy' not found.")
    print("Please run task2_main.py first.")
    exit()
except Exception as e:
    print(f"Error in loading: {e}")
    exit()

# =============================================================================
# SECTION 3 — LINEARIZATION ALONG REFERENCE TRAJECTORY
# =============================================================================
print("\nLinearizing the system along reference trajectory...")
A_list, B_list = [], []
for t in range(steps):
    x_t = x_ref_traj[t]
    u_t = u_ref_traj[t]
    _, A, B = dynamics(x_t, u_t)
    A_list.append(A)
    B_list.append(B)
print(f"  Linearization complete: {steps} pairs (A_t, B_t)")

# =============================================================================
# SECTION 4 — BACKWARD RICCATI (TV-LQR Design)
# =============================================================================
if QQT_dare is not None:
    QQf = QQT_dare
    print("\nUsing Q_T from DARE (loaded from Task 2).")
else:
    QQf = Q_lqr
    print("\nUsing Q_T = Q_lqr (DARE not available).")

print("Computing Riccati backward (TV-LQR)...")
K_gains, P_list = backward_riccati(A_list, B_list, Q_lqr, R_lqr, QQf, steps)
print(f"  Backward Riccati complete.")
print(f"  K_gains[0] = {K_gains[0].round(3)}")

# =============================================================================
# SECTION 5 — TRACKING SIMULATION WITH MULTIPLE PERTURBATIONS
# =============================================================================
perturbations = {
    'Pert. shoulder -0.2 rad': np.array([-0.2,  0.0, 0.0, 0.0]),
    'Pert. elbow +0.3 rad': np.array([ 0.0,  0.3, 0.0, 0.0]),
    'Pert. vel. shoulder'    : np.array([ 0.0,  0.0, 0.3, 0.0]),
}

results = {}

print("\nSimulating LQR tracking with multiple perturbations...")

for label, pert in perturbations.items():
    x_sim = np.zeros((steps + 1, ns))
    u_sim = np.zeros((steps, ni))
    x_sim[0] = x_ref_traj[0] + pert

    for t in range(steps):
        delta_x = x_sim[t] - x_ref_traj[t]

        delta_u  = -K_gains[t] @ delta_x
        u_curr   = u_ref_traj[t] + delta_u

        u_sim[t] = u_curr.flatten()
        x_sim[t+1] = step(x_sim[t], u_curr.flatten())

    err_final = np.linalg.norm(x_sim[-2] - x_ref_traj[-1])
    print(f"  [{label}]: final error ||x_T - x*_T|| = {err_final:.4e}")
    results[label] = {'x_sim': x_sim, 'u_sim': u_sim}

# =============================================================================
# SECTION 6 — PLOT RESULTS
# =============================================================================

def plot_results_task3(time, xx_ref, uu_ref, xx_sim, uu_sim, label_name, pert_idx):
    """
    Plot comparison Reference (Task 2 Opt) vs LQR Tracking (Task 3).
    Layout matches reference project.
    """
    # --- FIGURE 1: STATE TRACKING (Pos & Vel) & ERRORS ---
    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(f'Task 3: LQR Tracking Performance - {label_name}', fontsize=16, fontweight='bold')

    # --- 1.1 Theta 1 Position ---
    ax = axs[0, 0]
    ax.plot(time, np.degrees(xx_ref[:, 0]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 0]), 'b-', linewidth=2, label='LQR Link 1')
    ax.set_ylabel(r'Pos $\theta_1$ [deg]')
    ax.set_title('Link 1 Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 1.2 Theta 2 Position ---
    ax = axs[0, 1]
    ax.plot(time, np.degrees(xx_ref[:, 1]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 1]), 'r-', linewidth=2, label='LQR Link 2')
    ax.set_ylabel(r'Pos $\theta_2$ [deg]')
    ax.set_title('Link 2 Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 2.1 Error Theta 1 ---
    ax = axs[1, 0]
    err_pos1 = np.degrees(xx_sim[:, 0] - xx_ref[:, 0])
    ax.plot(time, err_pos1, color='dodgerblue', linewidth=1.5, label=r'Error $\theta_1$')
    ax.set_ylabel(r'Err $\theta_1$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 2.2 Error Theta 2 ---
    ax = axs[1, 1]
    err_pos2 = np.degrees(xx_sim[:, 1] - xx_ref[:, 1])
    ax.plot(time, err_pos2, color='tomato', linewidth=1.5, label=r'Error $\theta_2$')
    ax.set_ylabel(r'Err $\theta_2$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 3.1 Velocity Theta 1 ---
    ax = axs[2, 0]
    ax.plot(time, np.degrees(xx_ref[:, 2]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 2]), 'b-', linewidth=2, label='LQR Vel 1')
    ax.set_ylabel(r'Vel $\dot{\theta}_1$ [deg/s]')
    ax.set_title('Link 1 Velocity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 3.2 Velocity Theta 2 ---
    ax = axs[2, 1]
    ax.plot(time, np.degrees(xx_ref[:, 3]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 3]), 'r-', linewidth=2, label='LQR Vel 2')
    ax.set_ylabel(r'Vel $\dot{\theta}_2$ [deg/s]')
    ax.set_title('Link 2 Velocity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 4.1 Error Vel 1 ---
    ax = axs[3, 0]
    err_vel1 = np.degrees(xx_sim[:, 2] - xx_ref[:, 2])
    ax.plot(time, err_vel1, color='dodgerblue', linewidth=1.5, label=r'Error $\dot{\theta}_1$')
    ax.set_ylabel(r'Err $\dot{\theta}_1$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 4.2 Error Vel 2 ---
    ax = axs[3, 1]
    err_vel2 = np.degrees(xx_sim[:, 3] - xx_ref[:, 3])
    ax.plot(time, err_vel2, color='tomato', linewidth=1.5, label=r'Error $\dot{\theta}_2$')
    ax.set_ylabel(r'Err $\dot{\theta}_2$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'figs/task3_states_pert_{pert_idx}.png', dpi=300)
    plt.show(block=False)

    # --- FIGURE 2: INPUT & INPUT ERROR ---
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle(f'Task 3: Control Input ({label_name})', fontsize=14, fontweight='bold')
    
    uu_ref_plot = np.copy(uu_ref); uu_ref_plot[-1, :] = uu_ref_plot[-2, :]
    uu_sim_plot = np.copy(uu_sim); uu_sim_plot[-1, :] = uu_sim_plot[-2, :]

    # 1. Input Comparison
    ax = axs2[0]
    ax.step(time, uu_ref_plot[:, 0], 'k--', where='post', label='Ref Input', alpha=0.6)
    ax.step(time, uu_sim_plot[:, 0], 'g-', where='post', linewidth=2, label='LQR Input')
    ax.set_ylabel(r'Torque [Nm]')
    ax.set_title('Control Input', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 2. Input Error
    ax = axs2[1]
    err_u = uu_sim_plot[:, 0] - uu_ref_plot[:, 0]
    ax.step(time, err_u, 'seagreen', where='post', label=r'$\Delta u$')
    ax.set_ylabel(r'Err $\tau$ [Nm]')
    ax.set_xlabel('Time [s]')
    ax.set_title(r'Input Deviation ($\Delta u$)', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'figs/task3_input_pert_{pert_idx}.png', dpi=300)
    plt.show(block=False)

import os
os.makedirs('figs', exist_ok=True)

for i, (label, res) in enumerate(results.items()):
    x_sim_plot = res['x_sim'][:-1, :]
    u_sim_plot = res['u_sim']
    plot_results_task3(t_axis, x_ref_traj, u_ref_traj, x_sim_plot, u_sim_plot, label, i+1)
    
    # Animate the trajectory vs the reference
    print(f"\nStarting animation for: {label}")
    # Transpose arrays for animation (4, T)
    an.animate_trajectory(t_axis, x_sim_plot.T, x_ref_traj.T, title=f"Task 3 LQR Tracking: {label}")

# =============================================================================
# SECTION 7 — SAVE DATA
# =============================================================================
np.save('data/lqr_data_task3.npy', {
    'K_gains': K_gains,
    'P_list' : P_list,
    'Q_lqr'  : Q_lqr,
    'R_lqr'  : R_lqr,
    'QQf'    : QQf
})
print("\nLQR gains and Riccati matrices P_t saved to 'lqr_data_task3.npy'")
