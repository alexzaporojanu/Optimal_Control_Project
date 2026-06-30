#
# Task 1 — Generazione Traiettoria via Newton / iDDP
#           Riferimento: GRADINO (Downward ↦ Upward Equilibrium)
#
# Progetto Optimal Control — Parameter Set 3
# Autori: [inserire nomi]  — UniBO 2025/26
#
# Riferimento teorico:
#   [Slide 06] Optimal Control — Shooting Methods
#   [Slide 07] Gradient Method for Optimal Control
#   [Slide 08] Second-Order Closed-Loop Methods (Newton / iDDP)
#   [Session4/10_main_gradient_optcon_method.py] — struttura main identica
#
# PROBLEMA
# ========
# Trovare la sequenza di coppie {u_t}_{t=0}^{T-1} che minimizza:
#
#   J(x₀, u) = Σ_{t=0}^{T-1} l(x_t, u_t) + l_T(x_T) (costo)
#
#   s.t.  x_{t+1} = F(x_t, u_t)   (dinamica discreta RK4)
#         x_0     = x_start        (condizione iniziale)
#
# con costo quadratico attorno al riferimento a gradino x_ref(t).
#
# ALGORITMO: iDDP (iterative Differential Dynamic Programming)
# ============================================================
# Ogni iterazione k consiste in 3 fasi:
#   A. FORWARD PASS INIZIALE: simula la traiettoria con u^k
#   B. BACKWARD PASS (Newton):  calcola k_t, K_t via Q-function expansion
#   C. ARMIJO + FORWARD PASS: aggiorna u^{k+1} = u^k + α k + K(x-x̄)
#
# Il costo terminale Q_T è calcolato come soluzione della DARE,
# fornendo un'approssimazione del costo infinito-orizzonte all'equilibrio.
# [Rif.: Session4/10_main..., righe 83-88 — ctrl.dare()]
#

import numpy as np
import matplotlib.pyplot as plt
import signal

# Permette Ctrl-C nonostante i plot (pattern del professore)
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Stile plot coerente con gli esempi del professore
plt.rcParams.update({'font.size': 13})

import dynamics           as dyn
import reference_trajectory as ref_gen
import solver_newton
import cost               as cst
import armijo

# Tentiamo di importare 'control' per la DARE (opzionale — fallback su default)
try:
    import control as ctrl
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False
    print("INFO [Task 1]: pacchetto 'control' non trovato. "
          "Uso Q_T di default (non-DARE). Installa con: pip install control")

# =============================================================================
# SEZIONE 1 — CONFIGURAZIONE
# =============================================================================
print("=" * 60)
print("   Task 1: Trajectory Generation — Step Reference")
print("   Algoritmo: Newton / iDDP")
print("=" * 60)

tf   = 10.0           # Orizzonte temporale [s]
dt   = dyn.dt         # Passo di integrazione (da dynamics.py = 0.01 s)
ns   = dyn.ns         # 4 stati
ni   = dyn.ni         # 1 ingresso
TT   = int(tf / dt)   # Numero di passi = 1000


# Parametri algoritmo iDDP
max_iters = 50         # Max iterazioni (aumentato rispetto a V4 per convergenza)
term_cond = 1e-4       # Tolleranza convergenza su ||Δu|| (norma gradiente)
                       # [Rif.: Session4/10_main..., riga 53 — term_cond = 1e-4]

# Parametri Armijo — identici al professore [Session4/3_armijo.py]
cc               = 0.5    # Costante Armijo c ∈ (0,1)
beta             = 0.7    # Contrazione β ∈ (0,1)
armijo_maxiters  = 20     # Max iterazioni backtracking

# Visualizzazione iterativa (stile professore — Session4/10_main..., righe 162-303)
# NOTA PERFORMANCE: su Windows il plot iterativo può rallentare significativamente.
# Impostare True solo per la demo finale o su sistemi veloci.
visu_iter = False    # True = aggiorna plot ad ogni iter (bello ma lento)

# =============================================================================
# SEZIONE 2 — CARICAMENTO EQUILIBRI
# =============================================================================
# Gli equilibri (x_eq1, x_eq2) sono pre-calcolati da equilibrium_finding.py
# tramite SQP con condizioni KKT [Slide 03].
# Se il file non esiste, usa valori analitici approssimati.

try:
    data    = np.load('equilibrium_data.npy', allow_pickle=True).item()
    x_start = data['x_eq1']   # Equilibrio giù ≈ [0,0,0,0]
    x_goal  = data['x_eq2']   # Equilibrio su  ≈ [π,0,0,0]
    u_goal  = data['u_eq2']   # Ingresso equil. su ≈ [0]
    print(f"\nEquilibri caricati da 'equilibrium_data.npy':")
    print(f"  x_start = {x_start.round(4)}")
    print(f"  x_goal  = {x_goal.round(4)}")
except FileNotFoundError:
    print("\nWARNING: 'equilibrium_data.npy' non trovato. "
          "Esegui prima equilibrium_finding.py")
    print("Uso valori analitici approssimati.")
    x_start = np.zeros(4)
    x_goal  = np.array([np.pi, 0.0, 0.0, 0.0])
    u_goal  = np.array([0.0])

# =============================================================================
# SEZIONE 3 — RIFERIMENTO A GRADINO
# =============================================================================
# [Rif.: Slide 06 — "Step Reference Curve"]
# [Rif.: Session4/9_reference_trajectory.py — gen(step_reference=True)]
xx_ref, uu_ref = ref_gen.generate_step(tf, dt, x_start, x_goal)

# =============================================================================
# SEZIONE 4 — COSTO TERMINALE Q_T via DARE
# =============================================================================
# MIGLIORAMENTO rispetto a V4: invece di usare un Q_T ad-hoc (20000 I),
# calcoliamo Q_T come soluzione della DARE (Discrete Algebraic Riccati Equation)
# linearizzando all'equilibrio obiettivo.
#
# La DARE fornisce la soluzione del problema LQR a orizzonte INFINITO all'equilibrio:
#   P∞ = Q + A^T P∞ A - A^T P∞ B (R + B^T P∞ B)^{-1} B^T P∞ A
#
# Usare QQT = P∞ come costo terminale rende il costo a orizzonte finito
# una buona approssimazione del costo infinito (principio del "bang-bang" a fine orizzonte).
# [Rif.: Session4/10_main..., righe 83-88]

if HAS_CONTROL:
    try:
        _, A_eq, B_eq = dyn.dynamics(x_goal, u_goal)
        QQT = ctrl.dare(A_eq, B_eq, cst.QQt, cst.RRt)[0]
        print("\nQ_T calcolata dalla DARE all'equilibrio obiettivo.")
        print(f"  QQT diagonale ≈ {np.diag(QQT).round(1)}")
    except Exception as e:
        print(f"\nWARNING DARE: {e}. Uso Q_T di default.")
        QQT = cst.QQT
else:
    QQT = cst.QQT
    print("\nUSO Q_T di default (non-DARE).")

# =============================================================================
# SEZIONE 5 — INIZIALIZZAZIONE
# =============================================================================
# STRUTTURA DATI (pattern del professore — Session4/10_main..., righe 139-149):
# xx[:, :, k] = traiettoria stati all'iterazione k
# uu[:, :, k] = sequenza ingressi all'iterazione k

xx = np.zeros((ns, TT, max_iters))   # (4, 1000, 50)
uu = np.zeros((ni, TT, max_iters))   # (1, 1000, 50)
JJ       = np.zeros(max_iters)        # storico del costo
descent  = np.zeros(max_iters)        # storico ||Δu||²
descent_arm = np.zeros(max_iters)    # storico ∇Jᵀ Δu (per Armijo)

xx[:, 0, 0] = x_start   # Condizione iniziale fissata

# WARM START — "Kick" sinusoidale vicino al gradino per rompere la simmetria
# Il gradino avviene a T/2 = 5s. Applichiamo la perturbazione in [4s, 6s].
# Senza questo, l'ingresso nullo porta il robot a rimanere vicino all'equilibrio giù.
# [Rif.: Session4/10_main..., riga 100-101 — "Starting at low position"]
tt_hor  = np.linspace(0, tf, TT)
mask_kick = (tt_hor > 4.0) & (tt_hor < 6.0)
uu[0, mask_kick, 0] = 5.0 * np.sin(5.0 * tt_hor[mask_kick])

# Primo rollout con warm start
for t in range(TT - 1):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])

# =============================================================================
# SEZIONE 6 — PLOT ITERATIVO (stile professore)
# =============================================================================
# Replica la struttura di Session4/10_main..., righe 164-303
# La figura viene aggiornata ad ogni iterazione con plt.pause()

if visu_iter:
    plt.ion()   # Modalità interattiva NON-BLOCCANTE — evita il freeze su Windows
    fig_iter, axs_iter = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    fig_iter.suptitle('Task 1 — Newton/iDDP: Evoluzione Traiettoria', fontsize=14)

def _ylim_safe(a, b):
    """Calcola limiti y con margine — da Session4/10_main..., righe 252-258."""
    mn = min(np.nanmin(a), np.nanmin(b))
    mx = max(np.nanmax(a), np.nanmax(b))
    if np.isfinite(mn) and np.isfinite(mx) and mx - mn > 0:
        m = 0.1 * (mx - mn)
        return mn - m, mx + m
    return mn - 1.0, mx + 1.0

def _update_iter_plot(kk):
    """Aggiorna il plot iterativo — [Rif.: Session4/10_main..., righe 241-303]."""
    if not visu_iter:
        return
    for ax in axs_iter:
        ax.cla()

    # θ₁ — angolo link 1
    axs_iter[0].plot(tt_hor, xx[0, :, kk], color='#1f77b4', lw=2, label=r'$\theta_1$ ottimo')
    axs_iter[0].plot(tt_hor, xx_ref[0, :TT], color='#2ca02c', lw=2, ls='--', label=r'$\theta_1$ ref')
    axs_iter[0].set_ylabel(r'$\theta_1$ [rad]', fontsize=12)
    axs_iter[0].legend(loc='upper right', fontsize=10)
    axs_iter[0].grid(alpha=0.4)
    axs_iter[0].set_ylim(_ylim_safe(xx[0, :, kk], xx_ref[0, :TT]))
    axs_iter[0].set_title(
        f'Iter {kk} — J = {JJ[kk]:.4e} | ||Δu|| = {np.sqrt(descent[kk]):.3e}',
        fontsize=12)

    # θ₂ — angolo link 2
    axs_iter[1].plot(tt_hor, xx[1, :, kk], color='#17becf', lw=2, label=r'$\theta_2$ ottimo')
    axs_iter[1].plot(tt_hor, xx_ref[1, :TT], color='#9467bd', lw=2, ls='--', label=r'$\theta_2$ ref')
    axs_iter[1].set_ylabel(r'$\theta_2$ [rad]', fontsize=12)
    axs_iter[1].legend(loc='upper right', fontsize=10)
    axs_iter[1].grid(alpha=0.4)

    # Ingresso u (coppia al gomito)
    axs_iter[2].plot(tt_hor, uu[0, :, kk], color='#d62728', lw=2, label=r'$\tau$ [Nm]')
    axs_iter[2].plot(tt_hor, uu_ref[0, :TT], color='#ff7f0e', lw=2, ls='--', label=r'$\tau_{ref}$')
    axs_iter[2].set_ylabel(r'$\tau$ [Nm]', fontsize=12)
    axs_iter[2].set_xlabel('Tempo [s]', fontsize=12)
    axs_iter[2].legend(loc='upper right', fontsize=10)
    axs_iter[2].grid(alpha=0.4)

    plt.tight_layout()
    plt.pause(1e-4)

# =============================================================================
# SEZIONE 7 — LOOP PRINCIPALE Newton / iDDP
# =============================================================================
print("\n" + "-"*50)
print("Avvio ottimizzazione Newton / iDDP...")
print("-"*50)

converged_iter = max_iters - 1

for kk in range(max_iters - 1):

    # ------------------------------------------------------------------
    # A. CALCOLO COSTO e GRADIENTI (Forward Evaluation)
    # ------------------------------------------------------------------
    # Calcola J^k e i termini per il backward pass:
    #   lx_t  = ∂l/∂x,   lu_t  = ∂l/∂u
    #   lxx_t = Q = ∂²l/∂x²,   luu_t = R = ∂²l/∂u²
    #   A_t   = ∂F/∂x,   B_t   = ∂F/∂u
    # [Rif.: Session4/10_main..., righe 170-203]

    JJ[kk] = 0.0
    AA  = np.zeros((ns, ns, TT))
    BB  = np.zeros((ns, ni, TT))
    lx  = np.zeros((ns, TT))
    lu  = np.zeros((ni, TT))
    lxx = np.zeros((ns, ns, TT))
    luu = np.zeros((ni, ni, TT))

    for t in range(TT - 1):
        xt, ut = xx[:, t, kk], uu[:, t, kk]

        # Costo di stadio e gradienti [cost.py — stagecost()]
        c, gx, gu   = cst.stagecost(xt, ut, xx_ref[:, t], uu_ref[:, t])
        JJ[kk]      += c
        lx[:, t]    = gx
        lu[:, t]    = gu
        lxx[:,:, t] = cst.QQt    # Hessiana costante per costo quadratico
        luu[:,:, t] = cst.RRt

        # Linearizzazione dinamica [dynamics.py — dynamics()]
        _, A_t, B_t = dyn.dynamics(xt, ut)
        AA[:,:, t]  = A_t
        BB[:,:, t]  = B_t

    # Costo terminale [cost.py — termcost()]
    # Usa QQT calcolata dalla DARE (miglioramento rispetto a V4)
    c_T, gxT       = cst.termcost(xx[:, TT-1, kk], xx_ref[:, -1], QQT)
    JJ[kk]         += c_T
    lx[:, -1]      = gxT
    lxx[:,:, -1]   = QQT    # Q_T da DARE all'equilibrio obiettivo

    print(f"  Iter {kk:3d}: J = {JJ[kk]:.6e}", end="")

    # ------------------------------------------------------------------
    # B. BACKWARD PASS — Newton Step (Q-function expansion)
    # ------------------------------------------------------------------
    # Calcola k_t (feedforward) e K_t (feedback) risolvendo il
    # sottoproblema LQR locale [Slide 08 — Q-function expansion]
    # [solver_newton.py — solve_newton_step()]

    KK, kk_vec = solver_newton.solve_newton_step(AA, BB, lx, lu, lxx, luu, TT)

    # Calcolo della norma del descent (criterio di convergenza)
    # [Session4/10_main..., righe 205-206]
    for t in range(TT - 1):
        descent[kk]     += kk_vec[:, t].T @ kk_vec[:, t]       # ||Δu||²
        descent_arm[kk] += lu[:, t].T    @ kk_vec[:, t]        # ∇Jᵀ Δu

    print(f"  | ||Δu|| = {np.sqrt(descent[kk]):.4e}")

    # Visualizzazione iterativa
    _update_iter_plot(kk)

    # ------------------------------------------------------------------
    # CONDIZIONE DI TERMINAZIONE
    # ------------------------------------------------------------------
    # Usiamo la norma del descent come criterio (come il professore):
    #   ||Δu||² ≤ term_cond
    # [Session4/10_main..., righe 311-315]
    if descent[kk] <= term_cond:
        print(f"\n  CONVERGENZA RAGGIUNTA a iter {kk}: "
              f"||Δu||² = {descent[kk]:.2e} ≤ {term_cond:.0e}")
        converged_iter = kk
        break

    # ------------------------------------------------------------------
    # C. ARMIJO LINE SEARCH + FORWARD PASS (Closed-Loop Update)
    # ------------------------------------------------------------------
    # [armijo.py — select_stepsize()] con K_fb=KK (Newton closed-loop)
    # [Rif.: Slide 08 — "Closed-Loop Forward Pass"]
    # [Rif.: Session4/10_main..., righe 215-238]

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
        K_fb           = KK,              # Closed-loop update (Newton)
        xx_nom         = xx[:, :, kk],
        plot           = (kk < 3)         # Mostra plot Armijo per prime 3 iter
    )

    # Update della traiettoria con legge closed-loop:
    # u_t^{k+1} = u_t^k + α · k_t + K_t (x_t^{k+1} - x_t^k)
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
    print(f"\n  Iterazioni massime ({max_iters}) raggiunte senza convergenza.")
    converged_iter = max_iters - 2

# Traiettoria ottima
xx_star = xx[:, :, converged_iter]
uu_star = uu[:, :, converged_iter]

# =============================================================================
# SEZIONE 8 — PLOT FINALI
# =============================================================================
# Stile coerente con Session4/10_main..., righe 322-379

# --- Plot convergenza ---
fig_conv, axs_conv = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig_conv.suptitle('Task 1 — Convergenza Newton/iDDP', fontsize=14)

iters_ran = np.arange(converged_iter + 1)

axs_conv[0].semilogy(iters_ran, JJ[:converged_iter+1],
                     'o-', color='#1f77b4', lw=2, label='$J(u^k)$')
axs_conv[0].set_ylabel('Costo $J$ (log)', fontsize=12)
axs_conv[0].grid(alpha=0.4)
axs_conv[0].legend(fontsize=11)

axs_conv[1].semilogy(iters_ran, descent[:converged_iter+1],
                     's--', color='#d62728', lw=2, label=r'$\|\|\Delta u\|\|^2$')
axs_conv[1].axhline(term_cond, color='k', ls=':', lw=1.5,
                    label=f'Soglia {term_cond:.0e}')
axs_conv[1].set_ylabel(r'$\|\Delta u\|^2$ (log)', fontsize=12)
axs_conv[1].set_xlabel('Iterazione $k$', fontsize=12)
axs_conv[1].grid(alpha=0.4)
axs_conv[1].legend(fontsize=11)

plt.tight_layout()

# --- Plot traiettoria ottima vs riferimento ---
fig_opt, axs_opt = plt.subplots(ns + ni, 1, figsize=(11, 10),
                                 sharex=True)
fig_opt.suptitle('Task 1 — Traiettoria Ottima vs Riferimento (Gradino)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]',
            r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']
colors_opt = ['#1f77b4', '#17becf', '#2ca02c', '#9467bd']

for i in range(ns):
    axs_opt[i].plot(tt_hor, xx_star[i, :],
                    color=colors_opt[i], lw=2, label='Ottimo')
    axs_opt[i].plot(tt_hor, xx_ref[i, :TT],
                    color='k', lw=1.5, ls='--', label='Riferimento')
    axs_opt[i].set_ylabel(labels_x[i], fontsize=11)
    axs_opt[i].legend(loc='best', fontsize=9)
    axs_opt[i].grid(alpha=0.4)

axs_opt[ns].plot(tt_hor, uu_star[0, :],
                  color='#d62728', lw=2, label=r'$\tau$ ottimo [Nm]')
axs_opt[ns].plot(tt_hor, uu_ref[0, :TT],
                  color='#ff7f0e', lw=1.5, ls='--', label=r'$\tau_{ref}$')
axs_opt[ns].set_ylabel(r'$\tau$ [Nm]', fontsize=11)
axs_opt[ns].set_xlabel('Tempo [s]', fontsize=12)
axs_opt[ns].legend(loc='best', fontsize=9)
axs_opt[ns].grid(alpha=0.4)

plt.tight_layout()
plt.show(block=True)

# =============================================================================
# SEZIONE 9 — SALVATAGGIO
# =============================================================================
# La traiettoria ottima (x*, u*) è salvata per uso in Task 2, 3, 4.
# Convenzione shape: x (ns, TT), u (ni, TT) — colonne temporali

np.save('optimal_trajectory_task1.npy', {
    'x': xx_star,      # (4, TT)
    'u': uu_star,      # (1, TT)
    't': tt_hor,
    'J': JJ[:converged_iter+1],
    'QQT': QQT         # salva anche la DARE matrix per eventuale riferimento
})
print(f"\nTraiettoria Task 1 salvata in 'optimal_trajectory_task1.npy'")
print(f"  Costo finale:   J = {JJ[converged_iter]:.4e}")
print(f"  Convergenza a iter {converged_iter}")
