#
# Task 4 — Trajectory Tracking via MPC (Model Predictive Control)
#           with Time-Varying Linearized Dynamics (LTV)
# Optimal Control Project — Parameter Set 3
#

import numpy as np
import matplotlib.pyplot as plt
import signal
import os
import data
from dynamics       import dynamics, step
import animation as an

# Ensure plots don't block Ctrl+C
signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 13})

# Import CasADi for MPC optimization
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    print("ERROR: CasADi not found. Install with: pip install casadi")
    exit()

# =============================================================================
# SECTION 1 — MPC CONFIGURATION & SETTINGS
# =============================================================================
print("=" * 60)
print("   Task 4: Trajectory Tracking — LTV-MPC")
print("   Solver: CasADi IPOPT")
print("=" * 60)

ns, ni = data.ns, data.ni

# Toggle for rendering the visual animation at the end
SHOW_ANIMATION = True

# MPC Parameters
T_pred  = 100       # Prediction horizon [steps]
U_MAX   = 30.0      # Absolute torque limit [Nm]

# Weights (loaded from data.py)
Q_mpc  = data.Q_track
R_mpc  = data.R_track

# IPOPT options (quiet to keep output clean)
ipopt_opts = {
    "ipopt.print_level": 0,
    "print_time"       : 0,
    "ipopt.max_iter"   : 2000,
    "ipopt.tol"        : 1e-7,
}

# =============================================================================
# SECTION 2 — DATA LOADING (Trajectory + Riccati Gains)
# =============================================================================
print("\nLoading data...")

try:
    data_dict2   = np.load('data/optimal_trajectory_task2.npy', allow_pickle=True).item()
    x_ref_raw    = data_dict2['x']
    u_ref_raw    = data_dict2['u']
    t_axis       = data_dict2['t']

    # Clean dimension formatting
    if x_ref_raw.ndim == 3:
        x_ref_raw = x_ref_raw[:, :, -1]
    x_ref_traj = x_ref_raw.T if x_ref_raw.shape[0] == ns else x_ref_raw
    u_ref_traj = u_ref_raw.T if u_ref_raw.shape[0] == ni else u_ref_raw

    if u_ref_traj.ndim == 1:
        u_ref_traj = u_ref_traj.reshape(-1, 1)

    steps = x_ref_traj.shape[0]
    print(f"  Trajectory ref: {steps} steps, T={t_axis[-1]:.1f}s")

except FileNotFoundError:
    print("ERROR: 'optimal_trajectory_task2.npy' not found. Run task2_main.py first.")
    exit()

try:
    data_dict3   = np.load('data/lqr_data_task3.npy', allow_pickle=True).item()
    P_list  = data_dict3['P_list']
    print(f"  LQR Task 3 data loaded: {len(P_list)} Riccati matrices")
    USE_RICCATI_TERMINAL = True

except FileNotFoundError:
    print("  WARNING: 'data/lqr_data_task3.npy' not found.")
    print("  Using Q_T = Q_mpc as terminal cost (sub-optimal).")
    USE_RICCATI_TERMINAL = False
    P_list = [Q_mpc] * (steps + T_pred + 10)

# =============================================================================
# SECTION 3 — LINEARIZATION ALONG REFERENCE TRAJECTORY
# =============================================================================
print("\nLinearization along reference trajectory...")
A_list, B_list = [], []
for t in range(steps):
    _, A, B = dynamics(x_ref_traj[t], u_ref_traj[t].flatten())
    A_list.append(A)
    B_list.append(B)
print(f"  Linearization complete: {steps} pairs (A_t, B_t)")

# =============================================================================
# SECTION 4 — MPC SOLVER FUNCTION (CasADi Opti)
# =============================================================================
def solve_mpc_step(delta_x0, t_idx, horizon):
    """
    Solves the finite-horizon MPC subproblem from time t_idx.
    Deviation variables formulation:
        min  sum_{k=0}^{T_pred-1} (delta_x_k^T Q delta_x_k + delta_u_k^T R delta_u_k) + delta_x_T^T P_T delta_x_T
        s.t. delta_x_{k+1} = A_{t+k} delta_x_k + B_{t+k} delta_u_k   (LTV dynamics)
             delta_x_0 = delta_x0
             |u*_{t+k} + delta_u_k| <= U_MAX                         (absolute torque constraint)
    """
    opti = ca.Opti()
    DX   = opti.variable(ns, horizon + 1)
    DU   = opti.variable(ni, horizon)

    cost = 0.0

    for k in range(horizon):
        t_k = min(t_idx + k, steps - 1)
        A_k = ca.DM(A_list[t_k])
        B_k = ca.DM(B_list[t_k])

        cost += ca.mtimes([DX[:, k].T, ca.DM(Q_mpc), DX[:, k]])
        cost += ca.mtimes([DU[:, k].T, ca.DM(R_mpc), DU[:, k]])

        opti.subject_to(DX[:, k+1] == A_k @ DX[:, k] + B_k @ DU[:, k])

        t_k_abs  = min(t_idx + k, steps - 1)
        u_ref_k  = float(u_ref_traj[t_k_abs, 0])
        opti.subject_to(opti.bounded(-U_MAX - u_ref_k, DU[:, k], U_MAX - u_ref_k))

    t_term = min(t_idx + horizon, len(P_list) - 1)
    P_term = P_list[t_term] if USE_RICCATI_TERMINAL else Q_mpc
    cost   += ca.mtimes([DX[:, horizon].T, ca.DM(P_term), DX[:, horizon]])

    opti.subject_to(DX[:, 0] == ca.DM(delta_x0.reshape(ns, 1)))

    opti.minimize(cost)
    opti.solver("ipopt", ipopt_opts)

    try:
        sol         = opti.solve()
        delta_u0    = np.array(sol.value(DU[:, 0])).flatten()
        delta_x_pred = np.array(sol.value(DX))
        return delta_u0, delta_x_pred, 'ok'

    except Exception as e:
        print(f"  [MPC] IPOPT failed at t={t_idx}: {str(e)[:60]}...")
        return np.zeros(ni), np.zeros((ns, horizon + 1)), 'failed'


# =============================================================================
# SECTION 5 — MPC RECEDING HORIZON LOOP
# =============================================================================
perturbations = {
    'Pert. shoulder -0.2 rad': np.array([-0.2,  0.0, 0.0, 0.0]),
    'Pert. elbow +0.3 rad'   : np.array([ 0.0,  0.3, 0.0, 0.0]),
}

results = {}

for label, pert in perturbations.items():
    print(f"\n{'='*50}")
    print(f"  MPC Simulation: {label}")
    print(f"{'='*50}")

    x_sim = np.zeros((steps + 1, ns))
    u_sim = np.zeros((steps, ni))
    x_sim[0] = x_ref_traj[0] + pert

    failed_steps = 0

    for t in range(steps):
        if t % 100 == 0:
            err_t = np.linalg.norm(x_sim[t] - x_ref_traj[t])
            print(f"  t={t:4d}/{steps}: ||delta_x|| = {err_t:.3e}", end="")

        current_horizon = min(T_pred, steps - 1 - t)
        if current_horizon < 1:
            delta_u0 = np.zeros(ni)
            status = 'ok'
        else:
            delta_x0 = x_sim[t] - x_ref_traj[t]
            delta_u0, _, status = solve_mpc_step(delta_x0, t, current_horizon)

        if status == 'failed':
            failed_steps += 1

        u_applied   = u_ref_traj[t].flatten() + delta_u0
        u_sim[t]    = u_applied

        if t % 100 == 0:
            print(f"  | u_applied = {u_applied[0]:.2f}")

        x_sim[t+1] = step(x_sim[t], u_applied)

    err_final = np.linalg.norm(x_sim[-2] - x_ref_traj[-1])
    print(f"\n  Final error: ||x_T - x*_T|| = {err_final:.4e}")
    print(f"  IPOPT failed steps: {failed_steps}/{steps}")
    results[label] = {'x_sim': x_sim, 'u_sim': u_sim}

# =============================================================================
# SECTION 6 — PLOT RESULTS
# =============================================================================
os.makedirs('figs', exist_ok=True)
time = t_axis

# --- PLOT 1: COMPACT TRACKING ERRORS (Both perturbations compared together) ---
fig_err, axs_err = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig_err.suptitle('Task 4 — MPC Tracking Errors for Different Initial Conditions', fontsize=14, fontweight='bold')
colors = ['blue', 'orange']

for idx, (label, res) in enumerate(results.items()):
    x_sim_plot = res['x_sim'][:-1, :]
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
plt.savefig('figs/task4_tracking_errors.png', dpi=300)
plt.show(block=False)


# --- PLOT 2: NOMINAL VS OPTIMAL TRACKING (Shown for the shoulder perturbation) ---
primary_label = 'Pert. shoulder -0.2 rad'
primary_res = results[primary_label]
x_sim_primary = primary_res['x_sim'][:-1, :]
u_sim_primary = primary_res['u_sim']

fig_traj, axs_traj = plt.subplots(ns + ni, 1, figsize=(11, 10), sharex=True)
fig_traj.suptitle(f'Task 4 — State & Constrained Input MPC Tracking ({primary_label})', fontsize=14, fontweight='bold')

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
state_colors = ['blue', 'cyan', 'green', 'purple']

for i in range(ns):
    axs_traj[i].plot(time, x_sim_primary[:, i], color=state_colors[i], lw=2, label='Actual (MPC)')
    axs_traj[i].plot(time, x_ref_traj[:, i], color='black', lw=1.5, ls='--', label='Reference')
    axs_traj[i].set_ylabel(labels_x[i])
    axs_traj[i].legend(loc='best', fontsize=9)
    axs_traj[i].grid(alpha=0.3)

# Plot control torque along with constraint boundaries
axs_traj[ns].fill_between(time, -U_MAX, U_MAX, color='tab:green', alpha=0.1, label='Feasible region')
axs_traj[ns].axhline(U_MAX, color='darkred', linestyle='--', linewidth=1.2, label=r'$u_{max}$')
axs_traj[ns].axhline(-U_MAX, color='darkred', linestyle='--', linewidth=1.2, label=r'$u_{min}$')
axs_traj[ns].plot(time, u_sim_primary[:, 0], color='red', lw=2, label='Actual (MPC)')
axs_traj[ns].plot(time, u_ref_traj[:, 0], color='orange', lw=1.5, ls='--', label='Reference')
axs_traj[ns].set_ylabel(r'$\tau$ [Nm]')
axs_traj[ns].set_xlabel('Time [s]')
axs_traj[ns].legend(loc='best', fontsize=9, ncol=2)
axs_traj[ns].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figs/task4_state_input_tracking.png', dpi=300)

# Set the last plot call to block=True so that the figures stay open on screen
plt.show(block=True)

# =============================================================================
# SECTION 7 — ANIMATION (Animate only the first perturbation if enabled)
# =============================================================================
if SHOW_ANIMATION:
    print(f"\nStarting animation for: {primary_label}")
    # Transpose arrays for animation (4, T)
    an.animate_trajectory(time, x_sim_primary.T, x_ref_traj.T, title=f"Task 4 MPC Tracking: {primary_label}")

# =============================================================================
# SECTION 8 — SAVE DATA
# =============================================================================
np.save('data/mpc_results_task4.npy', {
    'results': results,
    't_axis' : t_axis,
    'T_pred' : T_pred,
    'U_MAX'  : U_MAX,
})
print("\nTask 4 MPC results saved to 'mpc_results_task4.npy'")
