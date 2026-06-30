#
# Task 2 — Trajectory Generation via Newton / iDDP
#           Riferimento: SMOOTH (Quintico) con struttura a 3 Fasi
#
# Progetto Optimal Control — Parameter Set 3
# Autori: [inserire nomi]  — UniBO 2025/26
#
# Riferimento teorico:
#   [Slide 06] Optimal Control Shooting   — "Reference Trajectory"
#   [Slide 07] Gradient Method            — "Main Loop"
#   [Slide 08] Second-Order Methods       — "Newton / iDDP"
#   [Session4/10_main_gradient_optcon_method.py] — struttura identica
#
# DIFFERENZA RISPETTO A TASK 1
# =============================
# Task 1 usa un riferimento a gradino (discontinuo).
# Task 2 usa un riferimento smooth (polinomio quintico, C²-continuo).
# Il riferimento è strutturato in 3 fasi:
#   [Pre-wait 5s] → [Smooth Move 10s] → [Post-hold 5s]
# Questo fornisce all'ottimizzatore un hint molto migliore sulla
# dinamica attesa, migliorando tipicamente la convergenza.
#
# Il warm start è applicato SOLO durante la fase di movimento,
# centrato sulla transizione — è qui che l'energia deve essere generata.
#

import numpy as np
import matplotlib.pyplot as plt
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 13})

import dynamics             as dyn
import reference_trajectory as ref_gen
import solver_newton
import cost                 as cst
import armijo

try:
    import control as ctrl
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False

# =============================================================================
# SEZIONE 1 — CONFIGURAZIONE
# =============================================================================
print("=" * 60)
print("   Task 2: Trajectory Generation — Smooth Reference (3-Phase)")
print("   Algoritmo: Newton / iDDP")
print("=" * 60)

dt  = dyn.dt
ns  = dyn.ns
ni  = dyn.ni

# Parametri iDDP
max_iters       = 50
term_cond       = 1e-4
cc              = 0.5
beta            = 0.7
armijo_maxiters = 20
visu_iter       = False   # True = aggiorna plot ad ogni iter (lento su Windows)

# =============================================================================
# SEZIONE 2 — CARICAMENTO EQUILIBRI
# =============================================================================
try:
    data    = np.load('equilibrium_data.npy', allow_pickle=True).item()
    x_start = data['x_eq1']
    x_goal  = data['x_eq2']
    u_goal  = data['u_eq2']
    print(f"\nEquilibri: x_start={x_start.round(3)}, x_goal={x_goal.round(3)}")
except FileNotFoundError:
    print("WARNING: equilibrium_data.npy non trovato. Uso valori di default.")
    x_start = np.zeros(4)
    x_goal  = np.array([np.pi, 0.0, 0.0, 0.0])
    u_goal  = np.array([0.0])

# =============================================================================
# SEZIONE 3 — RIFERIMENTO A 3 FASI (Smooth Quintico)
# =============================================================================
t_pre, t_move, t_post = 5.0, 10.0, 5.0

xx_ref, uu_ref, TT, tf, N_pre, N_move = ref_gen.generate_extended(
    dt, x_start, x_goal,
    t_pre=t_pre, t_move=t_move, t_post=t_post
)
tt_hor = np.linspace(0, tf, TT)

print(f"\nRiferimento generato:")
print(f"  TT = {TT} passi,  tf = {tf:.1f}s")
print(f"  Fasi: Pre({t_pre}s) + Move({t_move}s) + Post({t_post}s)")
print(f"  xx_ref shape: {xx_ref.shape},  uu_ref shape: {uu_ref.shape}")

# =============================================================================
# SEZIONE 4 — COSTO TERMINALE Q_T via DARE
# =============================================================================
if HAS_CONTROL:
    try:
        _, A_eq, B_eq = dyn.dynamics(x_goal, u_goal)
        QQT = ctrl.dare(A_eq, B_eq, cst.QQt, cst.RRt)[0]
        print(f"\nQ_T dalla DARE: diag ≈ {np.diag(QQT).round(1)}")
    except Exception as e:
        print(f"\nWARNING DARE: {e}. Uso Q_T di default.")
        QQT = cst.QQT
else:
    QQT = cst.QQT

# =============================================================================
# SEZIONE 5 — INIZIALIZZAZIONE + WARM START
# =============================================================================
xx        = np.zeros((ns, TT, max_iters))
uu        = np.zeros((ni, TT, max_iters))
JJ        = np.zeros(max_iters)
descent   = np.zeros(max_iters)
descent_arm = np.zeros(max_iters)

xx[:, 0, 0] = x_start

# WARM START MIRATO
t_kick_local = np.linspace(0, t_move, N_move)
uu[0, N_pre:N_pre + N_move, 0] = 5.0 * np.sin(3.0 * t_kick_local)

# Primo rollout
for t in range(TT - 1):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])

# =============================================================================
# SEZIONE 6 — PLOT ITERATIVO
# =============================================================================
if visu_iter:
    plt.ion()
    fig_iter, axs_iter = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig_iter.suptitle('Task 2 — Newton/iDDP: Evoluzione Traiettoria', fontsize=14)

def _ylim_safe(a, b):
    mn = min(np.nanmin(a), np.nanmin(b))
    mx = max(np.nanmax(a), np.nanmax(b))
    if np.isfinite(mn) and np.isfinite(mx) and mx - mn > 0:
        m = 0.1 * (mx - mn)
        return mn - m, mx + m
    return mn - 1.0, mx + 1.0

def _update_iter_plot(kk):
    if not visu_iter:
        return
    for ax in axs_iter:
        ax.cla()

    t_trans_start = t_pre
    t_trans_end   = t_pre + t_move

    axs_iter[0].plot(tt_hor, xx[0,:,kk], color='#1f77b4', lw=2, label=r'$\theta_1$ ottimo')
    axs_iter[0].plot(tt_hor, xx_ref[0,:TT], color='#2ca02c', lw=2, ls='--', label=r'$\theta_1$ ref')
    axs_iter[0].axvline(t_trans_start, color='gray', ls=':', lw=1, alpha=0.7)
    axs_iter[0].axvline(t_trans_end,   color='gray', ls=':', lw=1, alpha=0.7)
    axs_iter[0].set_ylabel(r'$\theta_1$ [rad]'); axs_iter[0].legend(fontsize=9)
    axs_iter[0].grid(alpha=0.4)
    axs_iter[0].set_ylim(_ylim_safe(xx[0,:,kk], xx_ref[0,:TT]))
    axs_iter[0].set_title(
        f'Iter {kk} — J={JJ[kk]:.4e} | ||Δu||={np.sqrt(descent[kk]):.3e}', fontsize=11)

    axs_iter[1].plot(tt_hor, xx[1,:,kk], color='#17becf', lw=2, label=r'$\theta_2$ ottimo')
    axs_iter[1].plot(tt_hor, xx_ref[1,:TT], color='#9467bd', lw=2, ls='--', label=r'$\theta_2$ ref')
    axs_iter[1].axvline(t_trans_start, color='gray', ls=':', lw=1, alpha=0.7)
    axs_iter[1].axvline(t_trans_end,   color='gray', ls=':', lw=1, alpha=0.7)
    axs_iter[1].set_ylabel(r'$\theta_2$ [rad]'); axs_iter[1].legend(fontsize=9)
    axs_iter[1].grid(alpha=0.4)

    axs_iter[2].plot(tt_hor, uu[0,:,kk], color='#d62728', lw=2, label=r'$\tau$ [Nm]')
    axs_iter[2].set_ylabel(r'$\tau$ [Nm]'); axs_iter[2].set_xlabel('Tempo [s]')
    axs_iter[2].legend(fontsize=9); axs_iter[2].grid(alpha=0.4)

    plt.tight_layout(); plt.pause(1e-4)

# =============================================================================
# SEZIONE 7 — LOOP PRINCIPALE Newton / iDDP
# =============================================================================
print("\n" + "-"*50)
print("Avvio ottimizzazione Newton / iDDP (Task 2)...")
print("-"*50)

converged_iter = max_iters - 1

for kk in range(max_iters - 1):

    # A. Calcolo costo e gradienti
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

    c_T, gxT     = cst.termcost(xx[:, TT-1, kk], xx_ref[:, -1], QQT)
    JJ[kk]       += c_T
    lx[:, -1]    = gxT
    lxx[:,:, -1] = QQT

    print(f"  Iter {kk:3d}: J = {JJ[kk]:.6e}", end="")

    # B. Backward Pass
    KK, kk_vec = solver_newton.solve_newton_step(AA, BB, lx, lu, lxx, luu, TT)
    for t in range(TT - 1):
        descent[kk]     += kk_vec[:, t].T @ kk_vec[:, t]
        descent_arm[kk] += lu[:, t].T    @ kk_vec[:, t]

    print(f"  | ||Δu|| = {np.sqrt(descent[kk]):.4e}")
    _update_iter_plot(kk)

    # Controllo convergenza
    if descent[kk] <= term_cond:
        print(f"\n  CONVERGENZA a iter {kk}: ||Δu||² = {descent[kk]:.2e}")
        converged_iter = kk
        break

    # C. Armijo + Forward Pass
    save_path_armijo = None
    if kk in [0, 1] or kk == converged_iter:
        save_path_armijo = f"task2_armijo_iter_{kk}.png"

    alpha = armijo.select_stepsize(
        stepsize_0=1.0, armijo_maxiters=armijo_maxiters,
        cc=cc, beta=beta, deltau=kk_vec,
        xx_ref=xx_ref, uu_ref=uu_ref, x0=x_start,
        uu=uu[:,:,kk], JJ=JJ[kk], descent_arm=descent_arm[kk],
        QQT=QQT, K_fb=KK, xx_nom=xx[:,:,kk],
        plot=(kk < 3), save_path=save_path_armijo
    )

    xx_new = np.zeros((ns, TT)); uu_new = np.zeros((ni, TT))
    xx_new[:, 0] = x_start
    for t in range(TT - 1):
        dx           = xx_new[:, t] - xx[:, t, kk]
        du           = alpha * kk_vec[:, t] + KK[:,:, t] @ dx
        uu_new[:, t] = uu[:, t, kk] + du
        xx_new[:, t+1] = dyn.step(xx_new[:, t], uu_new[:, t])

    xx[:,:, kk+1] = xx_new
    uu[:,:, kk+1] = uu_new

else:
    print(f"\nIterazioni massime ({max_iters}) raggiunte.")
    converged_iter = max_iters - 2

xx_star = xx[:,:, converged_iter]
uu_star = uu[:,:, converged_iter]

# =============================================================================
# SEZIONE 8 — PLOT FINALI (Richiesta Assignment)
# =============================================================================

# --- 1. Costo e norma del gradiente (semi-logaritmica) ---
fig_conv, axs_conv = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig_conv.suptitle('Task 2 — Convergenza Newton/iDDP (Scala Log)', fontsize=14)
iters_ran = np.arange(converged_iter + 1)
axs_conv[0].semilogy(iters_ran, JJ[:converged_iter+1], 'o-', color='#1f77b4', lw=2, label='$J(u^k)$')
axs_conv[0].set_ylabel('Costo $J$'); axs_conv[0].grid(alpha=0.4); axs_conv[0].legend()
axs_conv[1].semilogy(iters_ran, descent[:converged_iter+1], 's--', color='#d62728', lw=2, label='$||\\Delta u||^2$')
axs_conv[1].axhline(term_cond, color='k', ls=':', label=f'Soglia {term_cond:.0e}')
axs_conv[1].set_ylabel('$||\\Delta u||^2$'); axs_conv[1].set_xlabel('Iterazione'); axs_conv[1].grid(alpha=0.4); axs_conv[1].legend()
plt.tight_layout()
plt.savefig('task2_convergence_metrics.png', dpi=300)

# --- 2. Traiettoria Ottima vs Riferimento Smooth ---
fig_opt, axs_opt = plt.subplots(ns+ni, 1, figsize=(12, 10), sharex=True)
fig_opt.suptitle('Task 2 — Traiettoria Ottima vs Riferimento Smooth (3-Fasi)', fontsize=14)
labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]', r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
colors_opt = ['#1f77b4','#17becf','#2ca02c','#9467bd']
for i in range(ns):
    axs_opt[i].plot(tt_hor, xx_star[i,:], color=colors_opt[i], lw=2, label='Ottimo')
    axs_opt[i].plot(tt_hor, xx_ref[i,:TT], 'k--', lw=1.5, label='Riferimento')
    axs_opt[i].axvline(t_pre, color='gray', ls=':', alpha=0.6)
    axs_opt[i].axvline(t_pre+t_move, color='gray', ls=':', alpha=0.6)
    axs_opt[i].set_ylabel(labels_x[i]); axs_opt[i].legend(fontsize=9); axs_opt[i].grid(alpha=0.4)
axs_opt[ns].plot(tt_hor, uu_star[0,:], color='#d62728', lw=2, label=r'$\tau$ [Nm]')
axs_opt[ns].set_ylabel(r'$\tau$ [Nm]'); axs_opt[ns].set_xlabel('Tempo [s]')
axs_opt[ns].legend(fontsize=9); axs_opt[ns].grid(alpha=0.4)
plt.tight_layout()
plt.savefig('task2_optimal_trajectory.png', dpi=300)

# --- 3. Traiettorie Intermedie (Richiesta Assignment) ---
fig_inter, axs_inter = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
fig_inter.suptitle('Task 2 — Evoluzione delle Traiettorie Intermedie', fontsize=14)

iters_to_plot = [0, 1, 3, converged_iter]
iters_to_plot = [i for i in iters_to_plot if i <= converged_iter]
if converged_iter not in iters_to_plot:
    iters_to_plot.append(converged_iter)

# Theta 1
axs_inter[0].plot(tt_hor, xx_ref[0, :TT], 'k--', lw=2, label='Reference (Smooth)')
for kk_plot in iters_to_plot:
    lbl = f"Iter {kk_plot}"
    if kk_plot == 0:
        lbl = "Iter 0 (Warm Start)"
    elif kk_plot == converged_iter:
        lbl = "Iter Ottima (Converged)"
    axs_inter[0].plot(tt_hor, xx[0, :, kk_plot], label=lbl, alpha=0.8)
axs_inter[0].set_ylabel(r'$\theta_1$ [rad]', fontsize=12)
axs_inter[0].grid(alpha=0.4)
axs_inter[0].legend(fontsize=9, loc='upper right')

# Theta 2
axs_inter[1].plot(tt_hor, xx_ref[1, :TT], 'k--', lw=2, label='Reference')
for kk_plot in iters_to_plot:
    lbl = f"Iter {kk_plot}"
    if kk_plot == 0:
        lbl = "Iter 0 (Warm Start)"
    elif kk_plot == converged_iter:
        lbl = "Iter Ottima (Converged)"
    axs_inter[1].plot(tt_hor, xx[1, :, kk_plot], label=lbl, alpha=0.8)
axs_inter[1].set_ylabel(r'$\theta_2$ [rad]', fontsize=12)
axs_inter[1].set_xlabel('Tempo [s]', fontsize=12)
axs_inter[1].grid(alpha=0.4)
axs_inter[1].legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('task2_intermediate_trajectories.png', dpi=300)

plt.show(block=True)

# =============================================================================
# SEZIONE 9 — SALVATAGGIO
# =============================================================================
np.save('optimal_trajectory_task2.npy', {
    'x'    : xx_star,
    'u'    : uu_star,
    't'    : tt_hor,
    'QQT'  : QQT,
    'N_pre': N_pre,
})
print(f"\nTraiettoria Task 2 salvata in 'optimal_trajectory_task2.npy'")
print(f"  Costo finale: J = {JJ[converged_iter]:.4e}")
print(f"  Shape x: {xx_star.shape},  u: {uu_star.shape}")
