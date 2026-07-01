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

# CONFIGURATION & INITIALIZATION
print("=" * 60)
print("   Task 1: Trajectory Generation — Step Reference")
print("   Algorithm: Newton / iDDP")
print("=" * 60)

# Create output directories for neatness
os.makedirs('data', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Problem Dimensions:
# ns = number of states (4: theta1, theta2, omega1, omega2)
# ni = number of inputs (1: hip torque)
tf = 10.0           
TT = int(tf / data.dt)   # Total number of time steps (e.g., 1000)


# EXTRACT PARAMETERS FROM DATA.PY
# Extract cost matrices specifically tuned for Task 1
Q_task = data.Q_task1
R_task = data.R_task1

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

# Generate a step reference. 
# xx_ref shape: (4, 1000)
# uu_ref shape: (1, 1000)
xx_ref, uu_ref = ref_gen.generate_step(tf, data.dt, x_start, x_goal)

# Terminal cost matrix QQT computation (Shape: 4x4)
if HAS_CONTROL:
    _, A_eq, B_eq = dyn.dynamics(x_goal, u_goal)
    QQT = ctrl.dare(A_eq, B_eq, Q_task, R_task)[0]
else:
    QQT = data.QT_task1

 
# INITIAL GUESS
# Initialize arrays to store trajectories across all iterations
# xx shape: (states, time_steps, iterations) -> (4, 1000, 50)
xx = np.zeros((data.ns, TT, data.max_iters_task1))   
# uu shape: (inputs, time_steps, iterations) -> (1, 1000, 50)
uu = np.zeros((data.ni, TT, data.max_iters_task1))   

# Track convergence metrics over iterations
JJ          = np.zeros(data.max_iters_task1)  # Total cost
descent     = np.zeros(data.max_iters_task1)  # Squared norm of feedforward step
descent_arm = np.zeros(data.max_iters_task1)  # Expected descent (directional derivative)

# Set starting state for iteration 0
xx[:, 0, 0] = x_start
tt_hor = np.linspace(0, tf, TT)

# Warm Start Kick: Sine wave injected to pump initial kinetic energy
# This helps the Newton solver escape the downward equilibrium easily.
# mask_kick = (tt_hor > 4.0) & (tt_hor < 6.0)
# uu[0, mask_kick, 0] = 5.0 * np.sin(5.0 * tt_hor[mask_kick])

# Forward simulate the initial guess trajectory
for t in range(TT - 1):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])

 
# MAIN LOOP (Newton / iDDP)
print("\n" + "-"*50)
print("Starting Newton / iDDP optimization...")
print("-"*50)

converged_iter = data.max_iters_task1 - 1

for kk in range(data.max_iters_task1 - 1):

    JJ[kk] = 0.0
    
    # PREALLOCATE MATRICES FOR THE BACKWARD PASS
    # These store the linearized dynamics and quadratic cost terms for the entire trajectory
    AA  = np.zeros((data.ns, data.ns, TT))  # State Jacobians. Shape: (4, 4, 1000)
    BB  = np.zeros((data.ns, data.ni, TT))  # Input Jacobians. Shape: (4, 1, 1000)
    lx  = np.zeros((data.ns, TT))           # State cost gradients. Shape: (4, 1000)
    lu  = np.zeros((data.ni, TT))           # Input cost gradients. Shape: (1, 1000)
    lxx = np.zeros((data.ns, data.ns, TT))  # State cost Hessians. Shape: (4, 4, 1000)
    luu = np.zeros((data.ni, data.ni, TT))  # Input cost Hessians. Shape: (1, 1, 1000)

    # EVALUATE DYNAMICS AND COST ALONG CURRENT TRAJECTORY
    for t in range(TT - 1):
        xt = xx[:, t, kk]  # Current state. Shape: (4,)
        ut = uu[:, t, kk]  # Current input. Shape: (1,)

        # Stage cost evaluation using Task 1 specific weights
        c, gx, gu = cst.stagecost(xt, ut, xx_ref[:, t], uu_ref[:, t], Q_task, R_task)
        JJ[kk] += c
        
        # Store Gradients and Hessians
        lx[:, t]    = gx
        lu[:, t]    = gu
        lxx[:,:, t] = Q_task
        luu[:,:, t] = R_task

        # Linearize dynamics at the current operating point: A_t = df/dx, B_t = df/du
        _, A_t, B_t = dyn.dynamics(xt, ut)
        AA[:,:, t]  = A_t
        BB[:,:, t]  = B_t

    # Add terminal cost at t = TT - 1
    c_T, gxT     = cst.termcost(xx[:, TT-1, kk], xx_ref[:, -1], QQT)
    JJ[kk]       += c_T
    lx[:, -1]    = gxT
    lxx[:,:, -1] = QQT

    print(f"  Iter {kk:3d}: J = {JJ[kk]:.6e}", end="")

    # BACKWARD PASS (SOLVE RICCATI EQUATIONS)
    # KK     : Feedback gain matrices. Shape: (1, 4, 1000)
    # kk_vec : Feedforward step vectors. Shape: (1, 1000)
    # dV     : Expected cost reduction (scalar)
    KK, kk_vec, dV = solver_newton.solve_newton_step(AA, BB, lx, lu, lxx, luu, TT)
    descent_arm[kk] = dV

    # Compute the squared norm of the feedforward direction to check for convergence
    for t in range(TT - 1):
        descent[kk] += kk_vec[:, t].T @ kk_vec[:, t]

    print(f"  | ||Δu|| = {np.sqrt(descent[kk]):.4e}")

    # Stop if the step size is practically zero
    if descent[kk] <= data.term_cond:
        print(f"\n  CONVERGENCE REACHED at iter {kk}")
        converged_iter = kk
        break

    # Save Armijo line search plots for a few key iterations to avoid clutter
    save_path_armijo = f"figs/task1_armijo_iter_{kk}.png" if kk in [0, 1] or kk % 5 == 0 or kk == converged_iter else None

    # FORWARD PASS (ARMIJO LINE SEARCH)
    # Find a safe step size 'alpha' between 0 and 1 that strictly decreases the nonlinear cost
    alpha = armijo.select_stepsize(
        stepsize_0     = data.armijo_stepsize0,
        armijo_maxiters= data.armijo_maxiters,
        cc             = data.armijo_c,
        beta           = data.armijo_beta,
        deltau         = kk_vec,
        xx_ref         = xx_ref,
        uu_ref         = uu_ref,
        x0             = x_start,
        uu             = uu[:, :, kk],
        JJ             = JJ[kk],
        descent_arm    = descent_arm[kk],
        Q              = Q_task,
        R              = R_task,
        QQT            = QQT,
        K_fb           = KK,
        xx_nom         = xx[:, :, kk],
        plot           = False, 
        save_path      = save_path_armijo
    )

    # 5. APPLY THE UPDATE (CLOSED-LOOP ROLLOUT)
    xx_new = np.zeros((data.ns, TT))
    uu_new = np.zeros((data.ni, TT))
    xx_new[:, 0] = x_start

    for t in range(TT - 1):
        # State error between new trajectory and previous iteration trajectory. Shape: (4,)
        dx = xx_new[:, t] - xx[:, t, kk]
        
        # Closed-loop control update: Feedforward + Feedback. Shape: (1,)
        du = alpha * kk_vec[:, t] + KK[:, :, t] @ dx
        
        uu_new[:, t]   = uu[:, t, kk] + du
        xx_new[:, t+1] = dyn.step(xx_new[:, t], uu_new[:, t])

    # Save the new trajectory for the next iteration
    xx[:, :, kk+1] = xx_new
    uu[:, :, kk+1] = uu_new

else:
    print(f"\n  Max iterations ({data.max_iters_task1}) reached without convergence.")
    converged_iter = data.max_iters_task1 - 2

# Extract the final optimal trajectories
xx_star = xx[:, :, converged_iter]
uu_star = uu[:, :, converged_iter]

 
# FINAL PLOTS
# --- Plot 1: Convergence Metrics (Cost and Step Size) ---
fig_conv, axs_conv = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig_conv.suptitle('Task 1 — Newton/iDDP Convergence', fontsize=14)

iters_ran = np.arange(converged_iter + 1)
axs_conv[0].semilogy(iters_ran, JJ[:converged_iter+1], 'o-', color='blue', lw=2, label='Actual Cost $J(u^k)$')
axs_conv[0].set_ylabel('Cost $J$ (log)', fontsize=12)
axs_conv[0].grid(alpha=0.4)
axs_conv[0].legend()

axs_conv[1].semilogy(iters_ran, descent[:converged_iter+1], 's--', color='red', lw=2, label=r'Step Norm $\|\|\Delta u\|\|^2$')
axs_conv[1].semilogy(iters_ran, np.abs(descent_arm[:converged_iter+1]), '^-', color='green', lw=2, label=r'Expected Descent $|dJ|$')
axs_conv[1].axhline(data.term_cond, color='black', ls=':', lw=1.5, label=f'Threshold ({data.term_cond:.0e})')
axs_conv[1].set_ylabel('Metrics (log)', fontsize=12)
axs_conv[1].set_xlabel('Iteration $k$', fontsize=12)
axs_conv[1].grid(alpha=0.4)
axs_conv[1].legend()

plt.tight_layout()
plt.show(block=False)
plt.savefig('figs/task1_convergence_metrics.png', dpi=300)

# --- Plot 2: Optimal State and Input vs Reference ---
fig_opt, axs_opt = plt.subplots(data.ns + data.ni, 1, figsize=(11, 10), sharex=True)
fig_opt.suptitle('Task 1 — Optimal Trajectory vs Reference (Step)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
state_colors = ['blue', 'cyan', 'green', 'purple']

for i in range(data.ns):
    axs_opt[i].plot(tt_hor, xx_star[i, :], color=state_colors[i], lw=2, label='Optimal')
    axs_opt[i].plot(tt_hor, xx_ref[i, :TT], color='black', lw=1.5, ls='--', label='Reference')
    axs_opt[i].set_ylabel(labels_x[i])
    axs_opt[i].legend(loc='best', fontsize=10)
    axs_opt[i].grid(alpha=0.4)

axs_opt[data.ns].plot(tt_hor, uu_star[0, :], color='red', lw=2, label='Optimal Torque')
axs_opt[data.ns].plot(tt_hor, uu_ref[0, :TT], color='orange', lw=1.5, ls='--', label='Reference Torque')
axs_opt[data.ns].set_ylabel(r'$\tau$ [Nm]')
axs_opt[data.ns].set_xlabel('Time [s]')
axs_opt[data.ns].legend(loc='best', fontsize=10)
axs_opt[data.ns].grid(alpha=0.4)

plt.tight_layout()
plt.show(block=False)
plt.savefig('figs/task1_optimal_trajectory.png', dpi=300)

# --- Plot 3: Evolution of Intermediate Iterations ---
fig_inter, axs_inter = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig_inter.suptitle('Task 1 — Evolution of Intermediate Trajectories', fontsize=14)

# Select a few iterations to plot, ensuring we don't exceed the converged iteration count
iters_to_plot = sorted(list(set([0, 1, 3, converged_iter])))
plot_colors = ['gray', 'orange', 'green', 'blue']

axs_inter[0].plot(tt_hor, xx_ref[0, :TT], color='black', ls='--', lw=2, label='Reference')
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
plt.show(block=False)
plt.savefig('figs/task1_intermediate_trajectories.png', dpi=300)

# Save the final optimal arrays for tracking in Task 3
npy_save_path = 'data/optimal_trajectory_task1.npy'
np.save(npy_save_path, {
    'x': xx_star, 
    'u': uu_star, 
    't': tt_hor, 
    'J': JJ[:converged_iter+1], 
    'QQT': QQT
})
print(f"\nTask 1 trajectory safely saved to {npy_save_path}")