#
# Task 2 — Trajectory Generation via Newton / iDDP
#           Reference: SMOOTH (Quintic) with 3-Phase Structure
#
# Optimal Control Project — Parameter Set 3
#

import numpy as np
import matplotlib.pyplot as plt
import os
import signal

# Ensure plots don't block Ctrl+C in the terminal
signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 12})

import dynamics             as dyn
import reference_trajectory as ref_gen
import solver_newton
import cost                 as cst
import armijo

# Try to load the control library to solve the Discrete Algebraic Riccati Equation (DARE)
# This finds the optimal terminal cost matrix QQT.
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
print("   Algorithm: Newton / iDDP")
print("=" * 60)

# Create output directories for neatness
os.makedirs('data', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Problem Dimensions:
# ns = number of states (4: theta1, theta2, omega1, omega2)
# ni = number of inputs (1: hip torque)
dt  = dyn.dt
ns  = dyn.ns
ni  = dyn.ni

max_iters       = 50
term_cond       = 1e-4
cc              = 0.5    # Armijo reduction factor
beta            = 0.7    # Armijo contraction factor
armijo_maxiters = 20

# =============================================================================
# SECTION 2 — LOAD EQUILIBRIA & REFERENCES
# =============================================================================
# Load the start (downward) and goal (upward) states. Both have shape (4,)
try:
    data    = np.load('data/equilibrium_data.npy', allow_pickle=True).item()
    x_start = data['x_eq1']
    x_goal  = data['x_eq2']
    u_goal  = data['u_eq2']
    print(f"\nEquilibria loaded from 'data/equilibrium_data.npy':")
    print(f"  x_start = {x_start.round(4)}")
    print(f"  x_goal  = {x_goal.round(4)}")
except FileNotFoundError:
    print("\nWARNING: 'data/equilibrium_data.npy' not found. Defaulting to standard equilibria.")
    x_start = np.zeros(4)
    x_goal  = np.array([np.pi, 0.0, 0.0, 0.0])
    u_goal  = np.array([0.0])

# Task 2 specific structure: Wait for 5s, Move for 10s, Hold for 5s
t_pre, t_move, t_post = 5.0, 10.0, 5.0

# Generate a 3-phase quintic smooth reference.
# xx_ref shape: (4, TT)
# uu_ref shape: (1, TT)
xx_ref, uu_ref, TT, tf, N_pre, N_move = ref_gen.generate_extended(
    dt, x_start, x_goal, t_pre=t_pre, t_move=t_move, t_post=t_post
)
tt_hor = np.linspace(0, tf, TT)

# Terminal cost matrix QQT computation (Shape: 4x4)
if HAS_CONTROL:
    try:
        _, A_eq, B_eq = dyn.dynamics(x_goal, u_goal)
        QQT = ctrl.dare(A_eq, B_eq, cst.QQt, cst.RRt)[0]
    except Exception as e:
        QQT = cst.QQT
else:
    QQT = cst.QQT

# =============================================================================
# SECTION 3 — INITIAL GUESS (WARM START)
# =============================================================================
# We store the state and input trajectories for ALL iterations to plot them later.
# xx shape: (4, TT, 50)
# uu shape: (1, TT, 50)
xx        = np.zeros((ns, TT, max_iters))
uu        = np.zeros((ni, TT, max_iters))

# Track convergence metrics over iterations
JJ          = np.zeros(max_iters)  # Total cost
descent     = np.zeros(max_iters)  # Squared norm of feedforward step
descent_arm = np.zeros(max_iters)  # Expected descent (directional derivative)

# Set starting state for iteration 0
xx[:, 0, 0] = x_start

# We can apply a heuristic "kick" (sine wave) to pump energy into the Acrobot,
# but the Newton solver is powerful enough to find the solution without it!
# (Commented out for a smoother, naturally-found trajectory)
# t_kick_local = np.linspace(0, t_move, N_move)
# uu[0, N_pre:N_pre + N_move, 0] = 5.0 * np.sin(3.0 * t_kick_local)

# Forward simulate the initial guess trajectory (all zeros if kick is commented out)
for t in range(TT - 1):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])

# =============================================================================
# SECTION 4 — MAIN LOOP (Newton / iDDP)
# =============================================================================
print("\n" + "-"*50)
print("Starting Newton / iDDP optimization (Task 2)...")
print("-"*50)

converged_iter = max_iters - 1

for kk in range(max_iters - 1):

    # 1. INITIALIZE MATRICES FOR THE BACKWARD PASS
    # These store the linearized dynamics and quadratic cost terms for the entire trajectory
    JJ[kk] = 0.0
    AA  = np.zeros((ns, ns, TT))  # Jacobian of dynamics wrt x. Shape: (4, 4, TT)
    BB  = np.zeros((ns, ni, TT))  # Jacobian of dynamics wrt u. Shape: (4, 1, TT)
    lx  = np.zeros((ns, TT))      # Gradient of cost wrt x. Shape: (4, TT)
    lu  = np.zeros((ni, TT))      # Gradient of cost wrt u. Shape: (1, TT)
    lxx = np.zeros((ns, ns, TT))  # Hessian of cost wrt x. Shape: (4, 4, TT)
    luu = np.zeros((ni, ni, TT))  # Hessian of cost wrt u. Shape: (1, 1, TT)

    # 2. EVALUATE DYNAMICS AND COST ALONG CURRENT TRAJECTORY
    for t in range(TT - 1):
        xt = xx[:, t, kk]  # Current state. Shape: (4,)
        ut = uu[:, t, kk]  # Current input. Shape: (1,)
        
        # Stage cost evaluation
        c, gx, gu = cst.stagecost(xt, ut, xx_ref[:, t], uu_ref[:, t])
        JJ[kk] += c
        
        # Store Gradients and Hessians
        lx[:, t]    = gx
        lu[:, t]    = gu
        lxx[:,:, t] = cst.QQt
        luu[:,:, t] = cst.RRt
        
        # Linearize dynamics: A_t = df/dx, B_t = df/du
        _, A_t, B_t = dyn.dynamics(xt, ut)
        AA[:,:, t]  = A_t
        BB[:,:, t]  = B_t

    # Add terminal cost at t = TT - 1
    c_T, gxT     = cst.termcost(xx[:, TT-1, kk], xx_ref[:, -1], QQT)
    JJ[kk]       += c_T
    lx[:, -1]    = gxT
    lxx[:,:, -1] = QQT

    print(f"  Iter {kk:3d}: J = {JJ[kk]:.6e}", end="")

    # 3. BACKWARD PASS (SOLVE RICCATI EQUATIONS)
    # KK     : Feedback gain matrices. Shape: (1, 4, TT)
    # kk_vec : Feedforward step vectors. Shape: (1, TT)
    # dV     : Expected cost reduction (scalar)
    KK, kk_vec, dV = solver_newton.solve_newton_step(AA, BB, lx, lu, lxx, luu, TT)
    
    descent_arm[kk] = dV

    # Compute the squared norm of the feedforward direction to check for convergence
    for t in range(TT - 1):
        descent[kk] += kk_vec[:, t].T @ kk_vec[:, t]

    print(f"  | ||Δu|| = {np.sqrt(descent[kk]):.4e}")

    # Stop if the step size is practically zero
    if descent[kk] <= term_cond:
        print(f"\n  CONVERGENCE REACHED at iter {kk}")
        converged_iter = kk
        break

    # Save Armijo line search plots for a few key iterations to avoid clutter
    save_path_armijo = f"figs/task2_armijo_iter_{kk}.png" if kk in [0, 1] or kk % 5 == 0 or kk == converged_iter else None

    # 4. FORWARD PASS (ARMIJO LINE SEARCH)
    # Find a safe step size 'alpha' between 0 and 1 that strictly decreases the nonlinear cost
    alpha = armijo.select_stepsize(
        stepsize_0=1.0, armijo_maxiters=armijo_maxiters,
        cc=cc, beta=beta, deltau=kk_vec,
        xx_ref=xx_ref, uu_ref=uu_ref, x0=x_start,
        uu=uu[:,:,kk], JJ=JJ[kk], descent_arm=descent_arm[kk],
        QQT=QQT, K_fb=KK, xx_nom=xx[:,:,kk],
        plot=False, 
        save_path=save_path_armijo
    )

    # 5. APPLY THE UPDATE (CLOSED-LOOP ROLLOUT)
    xx_new = np.zeros((ns, TT))
    uu_new = np.zeros((ni, TT))
    xx_new[:, 0] = x_start
    
    for t in range(TT - 1):
        # State error between new trajectory and previous iteration trajectory. Shape: (4,)
        dx = xx_new[:, t] - xx[:, t, kk]
        
        # Closed-loop control update: Feedforward + Feedback. Shape: (1,)
        du = alpha * kk_vec[:, t] + KK[:,:, t] @ dx
        
        uu_new[:, t] = uu[:, t, kk] + du
        xx_new[:, t+1] = dyn.step(xx_new[:, t], uu_new[:, t])

    # Save the new trajectory for the next iteration
    xx[:,:, kk+1] = xx_new
    uu[:,:, kk+1] = uu_new

else:
    print(f"\nMax iterations ({max_iters}) reached without convergence.")
    converged_iter = max_iters - 2

# Extract the final optimal trajectories
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
axs_conv[1].axhline(term_cond, color='black', ls=':', label=f'Threshold ({term_cond:.0e})')
axs_conv[1].set_ylabel('Metrics (log)', fontsize=12)
axs_conv[1].set_xlabel('Iteration $k$', fontsize=12)
axs_conv[1].grid(alpha=0.4)
axs_conv[1].legend()

plt.tight_layout()
plt.savefig('figs/task2_convergence_metrics.png', dpi=300)

# --- Plot 2: Optimal State and Input vs Reference ---
fig_opt, axs_opt = plt.subplots(ns+ni, 1, figsize=(12, 10), sharex=True)
fig_opt.suptitle('Task 2 — Optimal Trajectory vs Smooth Reference (3-Phase)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
state_colors = ['blue', 'cyan', 'green', 'purple']

for i in range(ns):
    axs_opt[i].plot(tt_hor, xx_star[i,:], color=state_colors[i], lw=2, label='Optimal')
    axs_opt[i].plot(tt_hor, xx_ref[i,:TT], color='black', lw=1.5, ls='--', label='Reference')
    
    # Draw vertical lines to mark the 3 distinct movement phases
    axs_opt[i].axvline(t_pre, color='gray', ls=':', alpha=0.6)
    axs_opt[i].axvline(t_pre+t_move, color='gray', ls=':', alpha=0.6)
    
    axs_opt[i].set_ylabel(labels_x[i])
    axs_opt[i].legend(loc='best', fontsize=10)
    axs_opt[i].grid(alpha=0.4)

axs_opt[ns].plot(tt_hor, uu_star[0,:], color='red', lw=2, label='Optimal Torque')
axs_opt[ns].plot(tt_hor, uu_ref[0,:TT], color='orange', lw=1.5, ls='--', label='Reference Torque')
axs_opt[ns].set_ylabel(r'$\tau$ [Nm]')
axs_opt[ns].set_xlabel('Time [s]')
axs_opt[ns].legend(loc='best', fontsize=10)
axs_opt[ns].grid(alpha=0.4)

plt.tight_layout()
plt.savefig('figs/task2_optimal_trajectory.png', dpi=300)

# --- Plot 3: Evolution of Intermediate Iterations ---
fig_inter, axs_inter = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig_inter.suptitle('Task 2 — Evolution of Intermediate Trajectories', fontsize=14)

# Select a few iterations to plot, ensuring we don't exceed the converged iteration count
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
plt.show(block=True)

# Save the final optimal arrays for tracking in Task 3 & 4
np.save('data/optimal_trajectory_task2.npy', {
    'x': xx_star, 
    'u': uu_star, 
    't': tt_hor, 
    'QQT': QQT, 
    'N_pre': N_pre
})
print(f"\nTask 2 trajectory safely saved to 'data/optimal_trajectory_task2.npy'")