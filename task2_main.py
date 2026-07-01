import numpy as np
import matplotlib.pyplot as plt
import os
import signal

# Ensure plots don't block Ctrl+C in the terminal
signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 12})

# Import the unified physical and algorithmic parameters
import data                 
import dynamics as dyn
import reference_trajectory as ref_gen
import solver_newton
import cost as cst
import armijo

# Try to load the control library to solve the Discrete Algebraic Riccati Equation (DARE)
try:
    import control as ctrl
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False

# =============================================================================
# SECTION 1 — CONFIGURATION & INITIALIZATION
# =============================================================================
print("=" * 60)
print("   Task 2: Trajectory Generation — Smooth Reference (3-Phase)")
print("=" * 60)

# Create output directories for neatness
os.makedirs('data', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# =============================================================================
# SECTION 2 — EXTRACT PARAMETERS FROM DATA.PY
# =============================================================================
# Extract cost matrices specifically tuned for Task 2
Q_task = data.Q_task2
R_task = data.R_task2

# Load the start (downward) and goal (upward) states. Both have shape (4,)
try:
    eq_data = np.load('data/equilibrium_data.npy', allow_pickle=True).item()
    x_start = eq_data['x_eq1']
    x_goal  = eq_data['x_eq2']
    u_goal  = eq_data['u_eq2']
except FileNotFoundError:
    print("\nWARNING: 'data/equilibrium_data.npy' not found. Defaulting to standard equilibria.")
    x_start = np.zeros(data.ns)
    x_goal  = np.array([np.pi, 0.0, 0.0, 0.0])
    u_goal  = np.array([0.0])

# Task 2 specific structure: Wait for 5s, Move for 10s, Hold for 5s
t_pre, t_move, t_post = 5.0, 10.0, 5.0

# Generate a 3-phase quintic smooth reference.
# xx_ref shape: (4, TT)
# uu_ref shape: (1, TT)
xx_ref, uu_ref, TT, tf, N_pre, N_move = ref_gen.generate_extended(
    data.dt, x_start, x_goal, t_pre=t_pre, t_move=t_move, t_post=t_post
)
tt_hor = np.linspace(0, tf, TT)

# Terminal cost matrix QQT computation (Shape: 4x4)
if HAS_CONTROL:
    _, A_eq, B_eq = dyn.dynamics(x_goal, u_goal)
    QQT = ctrl.dare(A_eq, B_eq, Q_task, R_task)[0]
else:
    QQT = data.QT_task2

# =============================================================================
# SECTION 3 — INITIAL GUESS (WARM START)
# =============================================================================
# Initialize arrays to store trajectories across all iterations
# Initialize arrays to store trajectories across all iterations
xx = np.zeros((data.ns, TT, data.max_iters_task2 + 1))
uu = np.zeros((data.ni, TT, data.max_iters_task2 + 1))

# Set starting state for iteration 0
xx[:, 0, 0] = x_start

# WARM START KICK:
# Unlike Task 1's step reference, Task 2's gentle quintic reference creates very 
# weak initial gradients. The solver absolutely needs this initial hint of kinetic 
# energy during the "move" phase to escape the downward local minimum!
t_kick_local = np.linspace(0, t_move, N_move)
uu[0, N_pre:N_pre + N_move, 0] = 5.0 * np.sin(3.0 * t_kick_local)

# Forward simulate the initial guess trajectory
for t in range(TT - 1):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])

# =============================================================================
# SECTION 4 — MAIN LOOP (Newton / LTV-LQR)
# =============================================================================
print("\n" + "-"*50)
print("Starting Newton / LTV-LQR optimization (Task 2)...")
print("-"*50)

xx, uu, descent, descent_arm, JJ, converged_iter = solver_newton.newton_method(
    xx=xx, 
    uu=uu, 
    xx_ref=xx_ref, 
    uu_ref=uu_ref, 
    x0=x_start, 
    max_iters=data.max_iters_task2, 
    task_number=2, 
    armijo_plot=True, 
    armijo_plot_number=2, 
    save_path_armijo_base="figs/task2_armijo"
)

xx_star = xx[:,:, converged_iter]
uu_star = uu[:,:, converged_iter]

# =============================================================================
# SECTION 5 — FINAL PLOTS
# =============================================================================

# --- Plot 1: Convergence Metrics (Cost and Step Size) ---
fig_conv, axs_conv = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig_conv.suptitle('Task 2 — Newton/iDDP Convergence', fontsize=14)

iters_ran = np.arange(converged_iter + 1)
axs_conv[0].semilogy(iters_ran, JJ[:converged_iter+1], 'o-', color='blue', lw=2, label='Actual Cost $J(u^k)$')
axs_conv[0].set_ylabel('Cost $J$ (log)', fontsize=12)
axs_conv[0].grid(alpha=0.4)
axs_conv[0].legend()

axs_conv[1].semilogy(iters_ran, descent[:converged_iter+1], 's--', color='red', lw=2, label=r'Step Norm $\|\|\Delta u\|\|^2$')
axs_conv[1].semilogy(iters_ran, np.abs(descent_arm[:converged_iter+1]), '^-', color='green', lw=2, label=r'Expected Descent $|dJ|$')
axs_conv[1].axhline(data.term_cond, color='black', ls=':', label=f'Threshold ({data.term_cond:.0e})')
axs_conv[1].set_ylabel('Metrics (log)', fontsize=12)
axs_conv[1].set_xlabel('Iteration $k$', fontsize=12)
axs_conv[1].grid(alpha=0.4)
axs_conv[1].legend()

plt.tight_layout()
plt.savefig('figs/task2_convergence_metrics.png', dpi=300)
plt.show(block=False)

# --- Plot 2: Optimal State and Input vs Reference ---
fig_opt, axs_opt = plt.subplots(data.ns+data.ni, 1, figsize=(12, 10), sharex=True)
fig_opt.suptitle('Task 2 — Optimal Trajectory vs Smooth Reference (3-Phase)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
state_colors = ['blue', 'cyan', 'green', 'purple']

for i in range(data.ns):
    axs_opt[i].plot(tt_hor, xx_star[i,:], color=state_colors[i], lw=2, label='Optimal')
    axs_opt[i].plot(tt_hor, xx_ref[i,:TT], color='black', lw=1.5, ls='--', label='Reference')
    
    # Draw vertical lines to mark the 3 distinct movement phases
    axs_opt[i].axvline(t_pre, color='gray', ls=':', alpha=0.6)
    axs_opt[i].axvline(t_pre+t_move, color='gray', ls=':', alpha=0.6)
    
    axs_opt[i].set_ylabel(labels_x[i])
    axs_opt[i].legend(loc='best', fontsize=10)
    axs_opt[i].grid(alpha=0.4)

axs_opt[data.ns].plot(tt_hor, uu_star[0,:], color='red', lw=2, label='Optimal Torque')
axs_opt[data.ns].plot(tt_hor, uu_ref[0,:TT], color='orange', lw=1.5, ls='--', label='Reference Torque')
axs_opt[data.ns].set_ylabel(r'$\tau$ [Nm]')
axs_opt[data.ns].set_xlabel('Time [s]')
axs_opt[data.ns].legend(loc='best', fontsize=10)
axs_opt[data.ns].grid(alpha=0.4)

plt.tight_layout()
plt.savefig('figs/task2_optimal_trajectory.png', dpi=300)
plt.show(block=False)

# --- Plot 3: Evolution of Intermediate Iterations ---
fig_inter, axs_inter = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig_inter.suptitle('Task 2 — Evolution of Intermediate Trajectories', fontsize=14)

iters_to_plot = sorted(list(set([0, 1, 3, converged_iter])))
plot_colors = ['gray', 'orange', 'green', 'blue']

axs_inter[0].plot(tt_hor, xx_ref[0, :TT], color='black', ls='--', lw=2, label='Reference (Smooth)')
for i, kk_plot in enumerate(iters_to_plot):
    lbl = "Iter 0 (Warm Start)" if kk_plot == 0 else "Converged" if kk_plot == converged_iter else f"Iter {kk_plot}"
    axs_inter[0].plot(tt_hor, xx[0, :, kk_plot], color=plot_colors[i], lw=2, label=lbl, alpha=0.8)
axs_inter[0].set_ylabel(r'$\theta_1$ [rad]')
axs_inter[0].grid(alpha=0.4)
axs_inter[0].legend(loc='upper right')

axs_inter[1].plot(tt_hor, xx_ref[1, :TT], color='black', ls='--', lw=2, label='Reference')
for i, kk_plot in enumerate(iters_to_plot):
    lbl = "Iter 0 (Warm Start)" if kk_plot == 0 else "Converged" if kk_plot == converged_iter else f"Iter {kk_plot}"
    axs_inter[1].plot(tt_hor, xx[1, :, kk_plot], color=plot_colors[i], lw=2, label=lbl, alpha=0.8)
axs_inter[1].set_ylabel(r'$\theta_2$ [rad]')
axs_inter[1].set_xlabel('Time [s]')
axs_inter[1].grid(alpha=0.4)
axs_inter[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('figs/task2_intermediate_trajectories.png', dpi=300)
plt.show(block=False)

# Save the final optimal arrays for tracking in Task 3 & 4
npy_save_path = 'data/optimal_trajectory_task2.npy'
np.save(npy_save_path, {
    'x': xx_star, 
    'u': uu_star, 
    't': tt_hor, 
    'QQT': QQT, 
    'N_pre': N_pre
})
print(f"\nTask 2 trajectory safely saved to '{npy_save_path}'")