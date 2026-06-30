#
# Task 4 — Trajectory Tracking via MPC (Model Predictive Control)
#           con Dinamica Linearizzata Tempo-Variante (LTV)
#
# Progetto Optimal Control — Parameter Set 3
# Autori: [inserire nomi]  — UniBO 2025/26
#
# Riferimento teorico:
#   [Slide 11] Model Predictive Control    — sezione "MPC for Tracking"
#   [Slide 10] Optimal Control Tracking    — sezione "Receding Horizon"
#   [Slide 04] LQ Optimal Control         — sezione "Terminal Cost / DARE"
#   [Session6/solver.py]                   — CasADi Opti solver (base)
#   [Session6/main_MPC.py]                 — pattern MPC loop
#
# IDEA CENTRALE DI MPC
# =====================
# MPC applica ad ogni istante t un'ottimizzazione a ORIZZONTE FINITO T_pred:
#
#   min_{δu_t,...,δu_{t+T_pred-1}}
#       Σ_{k=0}^{T_pred-1} [δx_{t+k}^T Q δx_{t+k} + δu_{t+k}^T R δu_{t+k}]
#       + δx_{t+T_pred}^T P_{t+T_pred} δx_{t+T_pred}
#
#   s.t. δx_{k+1} = A_{t+k} δx_k + B_{t+k} δu_k    (LTV prediction model)
#        |u*_{t+k} + δu_{t+k}| ≤ U_MAX              (vincolo ingresso assoluto)
#        δx_0 = x_t - x*_t                           (stato attuale misurato)
#
# Solo il PRIMO ingresso u_t = u*_t + δu_t è applicato (Receding Horizon).
# Al passo successivo il problema viene ri-ottimizzato con il nuovo stato.
#
# VANTAGGI rispetto a TV-LQR (Task 3):
# - Gestisce esplicitamente i VINCOLI sull'ingresso
# - Può compensare disturbi non modelati più efficacemente
# - Orizzonte predittivo finito → meno conservativo del LQR infinito
#
# COSTO TERMINALE P_t dalla Riccati:
# Il terminal cost P_{t+T_pred} è preso dalla soluzione Riccati di Task 3,
# garantendo stabilità ricorsiva (sub-optimality bound teorico).
# [Rif.: Slide 11 — "Terminal Cost for Stability", Slide 04 — DARE]
#

import numpy as np
import matplotlib.pyplot as plt
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 13})

from dynamics       import dynamics, step
from solver_ltv_lqr import backward_riccati

# Import CasADi per l'ottimizzatore MPC
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    print("ERRORE: CasADi non trovato. Installa con: pip install casadi")
    exit()

# =============================================================================
# SEZIONE 1 — CONFIGURAZIONE MPC
# =============================================================================
print("=" * 60)
print("   Task 4: Trajectory Tracking — LTV-MPC")
print("   Solver: CasADi IPOPT")
print("=" * 60)

ns, ni = 4, 1

# Parametri MPC — [Rif.: Session6/main_MPC.py]
T_pred  = 10       # Orizzonte predittivo [passi] — bilancio computazione/ottimalità
U_MAX   = 30.0     # Limite coppia assoluta [Nm] — vincolo ingresso
DELTA_U_MAX = 20.0 # Limite DELTA coppia [Nm] — per robustezza numerica

# Pesi — possono essere diversi da Task 3 per sfruttare l'orizzonte finito
Q_mpc  = np.diag([1000.0, 1000.0, 100.0, 100.0])
R_mpc  = np.eye(ni) * 0.1

# Opzioni IPOPT (silenzioso per non sporcare l'output)
ipopt_opts = {
    "ipopt.print_level": 0,
    "print_time"       : 0,
    "ipopt.max_iter"   : 2000,
    "ipopt.tol"        : 1e-7,
}

# =============================================================================
# SEZIONE 2 — CARICAMENTO DATI (Traiettoria + Guadagni Riccati)
# =============================================================================
print("\nCaricamento dati...")

try:
    data2       = np.load('optimal_trajectory_task2.npy', allow_pickle=True).item()
    x_ref_raw   = data2['x']
    u_ref_raw   = data2['u']
    t_axis      = data2['t']

    if x_ref_raw.shape[0] == ns and x_ref_raw.shape[1] != ns:
        x_ref_traj = x_ref_raw.T
        u_ref_traj = u_ref_raw.T if u_ref_raw.shape[0] == ni else u_ref_raw
    else:
        x_ref_traj = x_ref_raw
        u_ref_traj = u_ref_raw

    if u_ref_traj.ndim == 1:
        u_ref_traj = u_ref_traj.reshape(-1, 1)

    steps = x_ref_traj.shape[0]
    print(f"  Traiettoria ref: {steps} passi, T={t_axis[-1]:.1f}s")

except FileNotFoundError:
    print("ERRORE: 'optimal_trajectory_task2.npy' non trovato. Esegui task2_main.py")
    exit()

try:
    data3   = np.load('lqr_data_task3.npy', allow_pickle=True).item()
    P_list  = data3['P_list']
    print(f"  Dati LQR Task 3 caricati: {len(P_list)} Riccati matrices")
    USE_RICCATI_TERMINAL = True

except FileNotFoundError:
    print("  WARNING: 'lqr_data_task3.npy' non trovato.")
    print("  Uso Q_T = Q_mpc come terminal cost (sub-ottimale).")
    USE_RICCATI_TERMINAL = False
    P_list = [Q_mpc] * (steps + T_pred + 10)

# =============================================================================
# SEZIONE 3 — LINEARIZZAZIONE LUNGO LA TRAIETTORIA DI RIFERIMENTO
# =============================================================================
print("\nLinearizzazione lungo la traiettoria di riferimento...")
A_list, B_list = [], []
for t in range(steps):
    _, A, B = dynamics(x_ref_traj[t], u_ref_traj[t].flatten())
    A_list.append(A)
    B_list.append(B)
print(f"  Linearizzazione completata: {steps} coppie (A_t, B_t)")

# =============================================================================
# SEZIONE 4 — FUNZIONE SOLVER MPC (CasADi Opti)
# =============================================================================
def solve_mpc_step(delta_x0, t_idx, A_list, B_list, P_list,
                   Q, R, x_ref_traj, u_ref_traj, steps, T_pred, U_MAX):
    """
    Risolve il sottoproblema MPC a orizzonte T_pred dal tempo t_idx.

    Formulazione in VARIABILI DI DEVIAZIONE δx, δu [Slide 11]:
        min  Σ_{k=0}^{T_pred-1} (δx_k^T Q δx_k + δu_k^T R δu_k) + δx_T^T P_T δx_T
        s.t. δx_{k+1} = A_{t+k} δx_k + B_{t+k} δu_k   (LTV dynamics)
             δx_0 = delta_x0
             |u*_{t+k} + δu_k| ≤ U_MAX                  (vincolo coppia assoluta)
    """
    horizon = min(T_pred, steps - t_idx)

    opti = ca.Opti()
    DX   = opti.variable(ns, horizon + 1)
    DU   = opti.variable(ni, horizon)

    cost = 0.0

    for k in range(horizon):
        t_k = min(t_idx + k, steps - 1)
        A_k = ca.DM(A_list[t_k])
        B_k = ca.DM(B_list[t_k])

        cost += ca.mtimes([DX[:, k].T, ca.DM(Q), DX[:, k]])
        cost += ca.mtimes([DU[:, k].T, ca.DM(R), DU[:, k]])

        opti.subject_to(DX[:, k+1] == A_k @ DX[:, k] + B_k @ DU[:, k])

        t_k_abs  = min(t_idx + k, steps - 1)
        u_ref_k  = float(u_ref_traj[t_k_abs, 0])
        opti.subject_to(opti.bounded(-U_MAX - u_ref_k, DU[:, k], U_MAX - u_ref_k))

    t_term = min(t_idx + horizon, len(P_list) - 1)
    P_term = P_list[t_term] if USE_RICCATI_TERMINAL else Q
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
        print(f"  [MPC] IPOPT fallito a t={t_idx}: {str(e)[:60]}...")
        return np.zeros(ni), np.zeros((ns, horizon + 1)), 'failed'


# =============================================================================
# SEZIONE 5 — LOOP MPC (Receding Horizon)
# =============================================================================
perturbations = {
    'Pert. spalla -0.2 rad': np.array([-0.2,  0.0, 0.0, 0.0]),
    'Pert. gomito +0.3 rad': np.array([ 0.0,  0.3, 0.0, 0.0]),
}

results = {}

for label, pert in perturbations.items():
    print(f"\n{'='*50}")
    print(f"  Simulazione MPC: {label}")
    print(f"{'='*50}")

    x_sim = np.zeros((steps + 1, ns))
    u_sim = np.zeros((steps, ni))
    x_sim[0] = x_ref_traj[0] + pert

    failed_steps = 0

    for t in range(steps):
        if t % 100 == 0:
            err_t = np.linalg.norm(x_sim[t] - x_ref_traj[t])
            print(f"  t={t:4d}/{steps}: ||δx|| = {err_t:.3e}", end="")

        delta_x0 = x_sim[t] - x_ref_traj[t]

        delta_u0, _, status = solve_mpc_step(
            delta_x0, t, A_list, B_list, P_list,
            Q_mpc, R_mpc, x_ref_traj, u_ref_traj,
            steps, T_pred, U_MAX
        )

        if status == 'failed':
            failed_steps += 1

        u_applied   = u_ref_traj[t].flatten() + delta_u0
        u_sim[t]    = u_applied

        if t % 100 == 0:
            print(f"  | u_applied = {u_applied[0]:.2f}")

        x_sim[t+1] = step(x_sim[t], u_applied)

    err_final = np.linalg.norm(x_sim[-2] - x_ref_traj[-1])
    print(f"\n  Errore finale: ||x_T - x*_T|| = {err_final:.4e}")
    print(f"  Passi falliti IPOPT: {failed_steps}/{steps}")
    results[label] = {'x_sim': x_sim, 'u_sim': u_sim}

# =============================================================================
# SEZIONE 6 — PLOT RISULTATI (Richiesta Assignment)
# =============================================================================
colors_pert = ['#d62728', '#1f77b4', '#2ca02c']

fig, axs = plt.subplots(ns + ni, 1, figsize=(12, 12), sharex=True)
fig.suptitle(f'Task 4 — LTV-MPC Tracking (T_pred={T_pred}, U_MAX={U_MAX}Nm)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]',
            r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']

for i, (label, res) in enumerate(results.items()):
    x_sim = res['x_sim']
    u_sim = res['u_sim']
    c = colors_pert[i]
    for j in range(ns):
        axs[j].plot(t_axis, x_sim[:-1, j], color=c, lw=1.8, label=label, alpha=0.85)
    axs[ns].plot(t_axis, u_sim[:, 0], color=c, lw=1.8, label=label, alpha=0.85)

for j in range(ns):
    axs[j].plot(t_axis, x_ref_traj[:, j], 'k--', lw=2, label='Riferimento x*', zorder=5)
    axs[j].axhline(0, color='gray', ls=':', lw=0.8, alpha=0.5)
    axs[j].set_ylabel(labels_x[j], fontsize=11)
    axs[j].legend(fontsize=8, loc='best')
    axs[j].grid(alpha=0.4)

axs[ns].plot(t_axis, u_ref_traj[:, 0], 'k--', lw=2, label='Riferimento u*', zorder=5)
axs[ns].axhline( U_MAX, color='red', ls=':', lw=1.5, alpha=0.7, label=f'U_MAX={U_MAX}')
axs[ns].axhline(-U_MAX, color='red', ls=':', lw=1.5, alpha=0.7)
axs[ns].set_ylabel(r'$\tau$ [Nm]', fontsize=11)
axs[ns].set_xlabel('Tempo [s]', fontsize=12)
axs[ns].legend(fontsize=8, loc='best')
axs[ns].grid(alpha=0.4)
plt.tight_layout()
plt.savefig('task4_tracking_results.png', dpi=300)

# Plot errore di tracking (Richiesta Assignment)
fig_err, ax_err = plt.subplots(figsize=(10, 5))
fig_err.suptitle('Task 4 — Errore di Tracking MPC per Diverse Condizioni Iniziali', fontsize=14)
for i, (label, res) in enumerate(results.items()):
    err_t = np.linalg.norm(res['x_sim'][:-1] - x_ref_traj, axis=1)
    ax_err.plot(t_axis, err_t, color=colors_pert[i], lw=2, label=label)
ax_err.set_xlabel('Tempo [s]'); ax_err.set_ylabel(r'$\|x_t - x^*_t\|$')
ax_err.legend(fontsize=10); ax_err.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('task4_tracking_error.png', dpi=300)

plt.show(block=True)

# =============================================================================
# SEZIONE 7 — SALVATAGGIO
# =============================================================================
np.save('mpc_results_task4.npy', {
    'results': results,
    't_axis' : t_axis,
    'T_pred' : T_pred,
    'U_MAX'  : U_MAX,
})
print("\nRisultati Task 4 MPC salvati in 'mpc_results_task4.npy'")
