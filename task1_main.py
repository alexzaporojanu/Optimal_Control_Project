#
# Task 1 — Trajectory Generation via Newton / iDDP
#           Reference: STEP (Downward ↦ Upward Equilibrium)
#
# Optimal Control Project — Parameter Set 3
#

import numpy as np
import matplotlib.pyplot as plt
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 13})

import dynamics           as dyn
import reference_trajectory as ref_gen
import solver_newton
import cost               as cst
import armijo

try:
    import control as ctrl
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False

# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================
print("=" * 60)
print("   Task 1: Trajectory Generation — Step Reference")
print("   Algorithm: Newton / iDDP")
print("=" * 60)

tf   = 10.0           
dt   = dyn.dt         
ns   = dyn.ns         
ni   = dyn.ni         
TT   = int(tf / dt)   

max_iters = 50         
term_cond = 1e-4       
cc               = 0.5    
beta             = 0.7    
armijo_maxiters  = 20     

# =============================================================================
# SECTION 2 — LOAD EQUILIBRIA
# =============================================================================
try:
    data    = np.load('equilibrium_data.npy', allow_pickle=True).item()
    x_start = data['x_eq1']   
    x_goal  = data['x_eq2']   
    u_goal  = data['u_eq2']
    print(f"\nEquilibria loaded from 'equilibrium_data.npy':")
    print(f"  x_start = {x_start.round(4)}")
    print(f"  x_goal  = {x_goal.round(4)}")
except FileNotFoundError:
    print("\nWARNING: 'equilibrium_data.npy' not found. Execute equilibrium_finding.py first")
    x_start = np.zeros(4)
    x_goal  = np.array([np.pi, 0.0, 0.0, 0.0])
    u_goal  = np.array([0.0])

# =============================================================================
# SECTION 3-5 — INIT & REFERENCES
# =============================================================================
xx_ref, uu_ref = ref_gen.generate_step(tf, dt, x_start, x_goal)

if HAS_CONTROL:
    try:
        _, A_eq, B_eq = dyn.dynamics(x_goal, u_goal)
        QQT = ctrl.dare(A_eq, B_eq, cst.QQt, cst.RRt)[0]
    except Exception as e:
        QQT = cst.QQT
else:
    QQT = cst.QQT

xx = np.zeros((ns, TT, max_iters))   
uu = np.zeros((ni, TT, max_iters))   
JJ       = np.zeros(max_iters)        
descent  = np.zeros(max_iters)        
descent_arm = np.zeros(max_iters)    

xx[:, 0, 0] = x_start

tt_hor  = np.linspace(0, tf, TT)
mask_kick = (tt_hor > 4.0) & (tt_hor < 6.0)
uu[0, mask_kick, 0] = 5.0 * np.sin(5.0 * tt_hor[mask_kick])

for t in range(TT - 1):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])

# =============================================================================
# SECTION 7 — MAIN LOOP Newton / iDDP
# =============================================================================
print("\n" + "-"*50)
print("Starting Newton / iDDP optimization...")
print("-"*50)

converged_iter = max_iters - 1

for kk in range(max_iters - 1):

    JJ[kk] = 0.0
    AA  = np.zeros((ns, ns, TT))
    BB  = np.zeros((ns, ni, TT))
    lx  = np.zeros((ns, TT))
    lu  = np.zeros((ni, TT))
    lxx = np.zeros((ns, ns, TT))
    luu = np.zeros((ni, ni, TT))

    for t in range(TT - 1):
        xt, ut = xx[:, t, kk], uu[:, t, kk]

        c, gx, gu   = cst.stagecost(xt, ut, xx_ref[:, t], uu_ref[:, t])
        JJ[kk]      += c
        lx[:, t]    = gx
        lu[:, t]    = gu
        lxx[:,:, t] = cst.QQt
        luu[:,:, t] = cst.RRt

        _, A_t, B_t = dyn.dynamics(xt, ut)
        AA[:,:, t]  = A_t
        BB[:,:, t]  = B_t

    c_T, gxT       = cst.termcost(xx[:, TT-1, kk], xx_ref[:, -1], QQT)
    JJ[kk]         += c_T
    lx[:, -1]      = gxT
    lxx[:,:, -1]   = QQT

    print(f"  Iter {kk:3d}: J = {JJ[kk]:.6e}", end="")

    KK, kk_vec, dV = solver_newton.solve_newton_step(AA, BB, lx, lu, lxx, luu, TT)
    
    descent_arm[kk] = dV

    for t in range(TT - 1):
        descent[kk] += kk_vec[:, t].T @ kk_vec[:, t]

    print(f"  | ||Δu|| = {np.sqrt(descent[kk]):.4e}")

    if descent[kk] <= term_cond:
        print(f"\n  CONVERGENCE REACHED at iter {kk}")
        converged_iter = kk
        break

    save_path_armijo = None
    # Save the Armijo plot for the first 2 iterations, every 5th iteration, and the last one
    if kk in [0, 1] or kk % 5 == 0 or kk == converged_iter:
        save_path_armijo = f"task1_armijo_iter_{kk}.png"

    alpha = armijo.select_stepsize(
        stepsize_0     = 1.0,
        armijo_maxiters= armijo_maxiters,
        cc             = cc,
        beta           = beta,
        deltau         = kk_vec,
        xx_ref         = xx_ref,
        uu_ref         = uu_ref,
        x0             = x_start,
        uu             = uu[:, :, kk],
        JJ             = JJ[kk],
        descent_arm    = descent_arm[kk],
        QQT            = QQT,
        K_fb           = KK,
        xx_nom         = xx[:, :, kk],
        plot           = False, 
        save_path      = save_path_armijo
    )

    xx_new = np.zeros((ns, TT))
    uu_new = np.zeros((ni, TT))
    xx_new[:, 0] = x_start

    for t in range(TT - 1):
        dx             = xx_new[:, t] - xx[:, t, kk]
        du             = alpha * kk_vec[:, t] + KK[:, :, t] @ dx
        uu_new[:, t]   = uu[:, t, kk] + du
        xx_new[:, t+1] = dyn.step(xx_new[:, t], uu_new[:, t])

    xx[:, :, kk+1] = xx_new
    uu[:, :, kk+1] = uu_new

else:
    print(f"\n  Max iterations ({max_iters}) reached without convergence.")
    converged_iter = max_iters - 2

xx_star = xx[:, :, converged_iter]
uu_star = uu[:, :, converged_iter]

# =============================================================================
# SECTION 8 — FINAL PLOTS (Assignment Requirement)
# =============================================================================

fig_conv, axs_conv = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig_conv.suptitle('Task 1 — Newton/iDDP Convergence (Log Scale)', fontsize=14)

iters_ran = np.arange(converged_iter + 1)
axs_conv[0].semilogy(iters_ran, JJ[:converged_iter+1], 'o-', color='#1f77b4', lw=2, label='$J(u^k)$')
axs_conv[0].set_ylabel('Cost $J$ (log)', fontsize=12)
axs_conv[0].grid(alpha=0.4)
axs_conv[0].legend(fontsize=11)

axs_conv[1].semilogy(iters_ran, descent[:converged_iter+1], 's--', color='#d62728', lw=2, label=r'$\|\|\Delta u\|\|^2$')
axs_conv[1].axhline(term_cond, color='k', ls=':', lw=1.5, label=f'Threshold {term_cond:.0e}')
axs_conv[1].set_ylabel(r'$\|\Delta u\|^2$ (log)', fontsize=12)
axs_conv[1].set_xlabel('Iteration $k$', fontsize=12)
axs_conv[1].grid(alpha=0.4)
axs_conv[1].legend(fontsize=11)
plt.tight_layout()
plt.savefig('task1_convergence_metrics.png', dpi=300)

fig_opt, axs_opt = plt.subplots(ns + ni, 1, figsize=(11, 10), sharex=True)
fig_opt.suptitle('Task 1 — Optimal Trajectory vs Reference (Step)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
colors_opt = ['#1f77b4', '#17becf', '#2ca02c', '#9467bd']

for i in range(ns):
    axs_opt[i].plot(tt_hor, xx_star[i, :], color=colors_opt[i], lw=2, label='Optimal')
    axs_opt[i].plot(tt_hor, xx_ref[i, :TT], color='k', lw=1.5, ls='--', label='Reference')
    axs_opt[i].set_ylabel(labels_x[i], fontsize=11)
    axs_opt[i].legend(loc='best', fontsize=9)
    axs_opt[i].grid(alpha=0.4)

axs_opt[ns].plot(tt_hor, uu_star[0, :], color='#d62728', lw=2, label=r'Optimal $\tau$ [Nm]')
axs_opt[ns].plot(tt_hor, uu_ref[0, :TT], color='#ff7f0e', lw=1.5, ls='--', label=r'Reference $\tau$')
axs_opt[ns].set_ylabel(r'$\tau$ [Nm]', fontsize=11)
axs_opt[ns].set_xlabel('Time [s]', fontsize=12)
axs_opt[ns].legend(loc='best', fontsize=9)
axs_opt[ns].grid(alpha=0.4)
plt.tight_layout()
plt.savefig('task1_optimal_trajectory.png', dpi=300)

fig_inter, axs_inter = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig_inter.suptitle('Task 1 — Evolution of Intermediate Trajectories', fontsize=14)

iters_to_plot = [0, 1, 3, converged_iter]
iters_to_plot = [i for i in iters_to_plot if i <= converged_iter]
if converged_iter not in iters_to_plot:
    iters_to_plot.append(converged_iter)

axs_inter[0].plot(tt_hor, xx_ref[0, :TT], 'k--', lw=2, label='Reference (Step)')
for kk_plot in iters_to_plot:
    lbl = "Iter 0 (Warm Start)" if kk_plot == 0 else "Optimal Iter (Converged)" if kk_plot == converged_iter else f"Iter {kk_plot}"
    axs_inter[0].plot(tt_hor, xx[0, :, kk_plot], label=lbl, alpha=0.8)
axs_inter[0].set_ylabel(r'$\theta_1$ [rad]', fontsize=12)
axs_inter[0].grid(alpha=0.4)
axs_inter[0].legend(fontsize=9, loc='upper right')

axs_inter[1].plot(tt_hor, xx_ref[1, :TT], 'k--', lw=2, label='Reference')
for kk_plot in iters_to_plot:
    lbl = "Iter 0 (Warm Start)" if kk_plot == 0 else "Optimal Iter (Converged)" if kk_plot == converged_iter else f"Iter {kk_plot}"
    axs_inter[1].plot(tt_hor, xx[1, :, kk_plot], label=lbl, alpha=0.8)
axs_inter[1].set_ylabel(r'$\theta_2$ [rad]', fontsize=12)
axs_inter[1].set_xlabel('Time [s]', fontsize=12)
axs_inter[1].grid(alpha=0.4)
axs_inter[1].legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('task1_intermediate_trajectories.png', dpi=300)
plt.show(block=True)

np.save('optimal_trajectory_task1.npy', {
    'x': xx_star, 'u': uu_star, 't': tt_hor, 'J': JJ[:converged_iter+1], 'QQT': QQT
})
print(f"\nTask 1 trajectory saved to 'optimal_trajectory_task1.npy'")