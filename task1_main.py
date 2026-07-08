import numpy as np
import matplotlib.pyplot as plt
import os
import signal
import data                 
import dynamics as dyn
import reference_trajectory as ref_gen
import solver_newton
from equilibrium_finding import find_equilibrium

import control as ctrl

# Ensure plots don't block Ctrl+C in the terminal
signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 12})


# CONFIGURATION & INITIALIZATION
print("=" * 60)
print("   Task 1: Trajectory Generation — Step Reference")
print("=" * 60)

# Create output directories for neatness
os.makedirs('data', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Problem Dimensions:
# ns = data.ns (state space), ni = data.ni (input space). time step dt = 0.01s.
tf = 10.0           
TT = int(tf / data.dt)   # Total number of time steps (3000 steps)


# EXTRACT PARAMETERS FROM DATA.PY
# Load cost matrices for stage cost: l(x,u) = 1/2*x^T*Q*x + 1/2*u^T*R*u
Q_task = data.Q_task1
R_task = data.R_task1

# Solve for exact start and goal equilibria dynamically
print("Computing exact system equilibria based on physical parameters...")
x_start, u_start = find_equilibrium(data.theta2_eq1, data.inverted_eq1, label="Equilibrium 1 (Start)")
x_goal, u_goal   = find_equilibrium(data.theta2_eq2, data.inverted_eq2, label="Equilibrium 2 (Goal)")

# Generate a discontinuous step reference. 
# Creates a pure transition boundary at T/2 that Newton's method must optimize.
xx_ref, uu_ref = ref_gen.generate_step(tf, data.dt, x_start, x_goal, u_start, u_goal)

# Terminal cost matrix QQT computation (Shape: 4x4)
# Solves the Discrete Algebraic Riccati Equation (DARE) at the target equilibrium.
# This represents the infinite-horizon quadratic cost-to-go of the linearized system.
_, A_eq, B_eq = dyn.dynamics(x_goal, u_goal.flatten())
QQT = ctrl.dare(A_eq, B_eq, Q_task, R_task)[0]

 
# INITIAL GUESS
# Trajectory dimension: state space x time horizon x iterations
xx = np.zeros((data.ns, TT + 1, data.max_iters_task1 + 1))   
uu = np.zeros((data.ni, TT, data.max_iters_task1 + 1))   

# Initialize the state trajectory at k=0 with the starting equilibrium
xx[:, 0, 0] = x_start
# the input trajectory with the step reference torque
uu[:, :, 0] = uu_ref
tt_hor = np.linspace(0, tf, TT + 1)

# Forward pass rollout for iteration 0 (open-loop simulation of the initial guess)
for t in range(TT):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])
 
# MAIN OPTIMIZATION LOOP
# Solves the optimal control problem using regularized Newton's closed-loop method (SLQ).
# The search direction is obtained at each iteration by solving an LTV LQR affine problem.
print("\n" + "-"*50)
print("Starting Newton / LTV-LQR optimization...")
print("-"*50)

xx, uu, descent, descent_arm, JJ, converged_iter = solver_newton.newton_method(
    xx=xx, 
    uu=uu, 
    xx_ref=xx_ref, 
    uu_ref=uu_ref, 
    x0=x_start,
    max_iters=data.max_iters_task1, 
    Qt=Q_task,
    Rt=R_task,
    QT=QQT,
    armijo_plot=True, 
    armijo_plot_number=2, 
    save_path_armijo_base="figs/task1_armijo",
)

# Extract the final converged state and control input trajectories
xx_star = xx[:, :, converged_iter]
uu_star = uu[:, :, converged_iter]

 
# FINAL PLOTS (Convergence and Optimal Trajectories)
# --- Plot 1: Convergence Metrics (Cost and Step Size) ---
fig_conv, axs_conv = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig_conv.suptitle('Task 1 — Newton Convergence', fontsize=14)

iters_ran = np.arange(converged_iter + 1)
axs_conv[0].semilogy(iters_ran, JJ[:converged_iter+1], 'o-', color='blue', lw=2, label='Actual Cost $J(u^k)$')
axs_conv[0].set_ylabel('Cost $J$ (log)', fontsize=12)
axs_conv[0].grid(alpha=0.4)
axs_conv[0].legend()

# Semi-logarithmic plot of the expected descent and the step norm to monitor convergence rate.
axs_conv[1].semilogy(iters_ran, descent[:converged_iter+1], 's--', color='red', lw=2, label=r'Step Norm $\|\|\Delta u\|\|^2$')
axs_conv[1].semilogy(iters_ran, np.abs(descent_arm[:converged_iter+1]), '^-', color='green', lw=2, label=r'Expected Descent $|dJ|$')
axs_conv[1].axhline(data.term_cond, color='black', ls=':', lw=1.5, label=f'Threshold ({data.term_cond:.0e})')
axs_conv[1].set_ylabel('Metrics (log)', fontsize=12)
axs_conv[1].set_xlabel('Iteration $k$', fontsize=12)
axs_conv[1].grid(alpha=0.4)
axs_conv[1].legend()

plt.tight_layout()
plt.savefig('figs/task1_convergence_metrics.png', dpi=300)


# --- Plot 2: Optimal State and Input vs Reference ---
fig_opt, axs_opt = plt.subplots(data.ns + data.ni, 1, figsize=(11, 10), sharex=True)
fig_opt.suptitle('Task 1 — Optimal Trajectory vs Reference (Step)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
state_colors = ['blue', 'cyan', 'green', 'purple']

for i in range(data.ns):
    axs_opt[i].plot(tt_hor, xx_star[i, :], color=state_colors[i], lw=2, label='Optimal')
    axs_opt[i].plot(tt_hor, xx_ref[i, :], color='black', lw=1.5, ls='--', label='Reference')
    axs_opt[i].set_ylabel(labels_x[i])
    axs_opt[i].legend(loc='best', fontsize=10)
    axs_opt[i].grid(alpha=0.4)

axs_opt[data.ns].plot(tt_hor[:-1], uu_star[0, :], color='red', lw=2, label='Optimal Torque')
axs_opt[data.ns].plot(tt_hor[:-1], uu_ref[0, :], color='orange', lw=1.5, ls='--', label='Reference Torque')
axs_opt[data.ns].set_ylabel(r'$\tau$ [Nm]')
axs_opt[data.ns].set_xlabel('Time [s]')
axs_opt[data.ns].legend(loc='best', fontsize=10)
axs_opt[data.ns].grid(alpha=0.4)

plt.tight_layout()
plt.savefig('figs/task1_optimal_trajectory.png', dpi=300)


# --- Plot 3: Evolution of Intermediate Iterations ---
fig_inter, axs_inter = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig_inter.suptitle('Task 1 — Evolution of Intermediate Trajectories', fontsize=14)

iters_to_plot = sorted(list(set([0, 1, 3, converged_iter])))
plot_colors = ['gray', 'orange', 'green', 'blue']

axs_inter[0].plot(tt_hor, xx_ref[0, :], color='black', ls='--', lw=2, label='Reference')
for i, kk_plot in enumerate(iters_to_plot):
    lbl = "Iter 0" if kk_plot == 0 else "Converged" if kk_plot == converged_iter else f"Iter {kk_plot}"
    axs_inter[0].plot(tt_hor, xx[0, :, kk_plot], color=plot_colors[i], lw=2, label=lbl, alpha=0.8)
axs_inter[0].set_ylabel(r'$\theta_1$ [rad]')
axs_inter[0].grid(alpha=0.4)
axs_inter[0].legend(loc='upper right')

axs_inter[1].plot(tt_hor, xx_ref[1, :], color='black', ls='--', lw=2, label='Reference')
for i, kk_plot in enumerate(iters_to_plot):
    lbl = "Iter 0" if kk_plot == 0 else "Converged" if kk_plot == converged_iter else f"Iter {kk_plot}"
    axs_inter[1].plot(tt_hor, xx[1, :, kk_plot], color=plot_colors[i], lw=2, label=lbl, alpha=0.8)
axs_inter[1].set_ylabel(r'$\theta_2$ [rad]')
axs_inter[1].set_xlabel('Time [s]')
axs_inter[1].grid(alpha=0.4)
axs_inter[1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('figs/task1_intermediate_trajectories.png', dpi=300)


# --- Plot 4: Distance from Goal Over Time ---
fig_dist, ax_dist = plt.subplots(figsize=(10, 5))
dist_from_goal = np.linalg.norm(xx_star - x_goal.reshape(-1, 1), axis=0)
ax_dist.plot(tt_hor, dist_from_goal, color='magenta', lw=2, label=r'$\|\mathbf{x}(t) - \mathbf{x}_{goal}\|_2$')
ax_dist.set_ylabel('Euclidean Distance from Goal', fontsize=12)
ax_dist.set_xlabel('Time [s]', fontsize=12)
ax_dist.set_title('Task 1 — Distance from Target Equilibrium State over Time', fontsize=14)
ax_dist.grid(alpha=0.4)
ax_dist.legend()
plt.tight_layout()
plt.savefig('figs/task1_distance_to_goal.png', dpi=300)
plt.show()

# Save the final optimal arrays for tracking in Task 3 & 4
npy_save_path = 'data/optimal_trajectory_task1.npy'
np.save(npy_save_path, {
    'x': xx_star, 
    'u': uu_star, 
    't': tt_hor, 
    'J': JJ[:converged_iter+1], 
    'QQT': QQT
})
print(f"\nTask 1 trajectory safely saved to {npy_save_path}")