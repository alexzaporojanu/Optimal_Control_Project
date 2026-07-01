#
# Task 3 — Trajectory Tracking via TV-LQR (Time-Varying LQR)
#
# Progetto Optimal Control — Parameter Set 3
# Autori: [inserire nomi]  — UniBO 2025/26
#
# Riferimento teorico:
#   [Slide 10] Optimal Control Based Tracking — sezione "TV-LQR Design"
#   [Slide 04] LQ Optimal Control             — sezione "Discrete Riccati Eq."
#   [Session5/3_main_dlqr_tracking.py]        — struttura main identica
#   [Session5/4_solver_ltv_LQR.py]            — backward Riccati (adattato)
#
# STRATEGIA DI TRACKING (Linearizzazione + LQR)
# ==============================================
# Data la traiettoria ottima (x*, u*) calcolata in Task 2,
# si linearizza il sistema Acrobot lungo di essa:
#
#   x_{t+1} = F(x*_t, u*_t) + A_t (x_t - x*_t) + B_t (u_t - u*_t)
#
# Definendo l'errore: δx_t = x_t - x*_t,  δu_t = u_t - u*_t
# si ottiene il sistema LTV (Linearizzato Tempo-Variante):
#
#   δx_{t+1} = A_t δx_t + B_t δu_t
#
# Il problema di ottimo per l'errore è il classico TV-LQR:
#   min Σ δx_t^T Q δx_t + δu_t^T R δu_t + δx_T^T Q_T δx_T
#
# Soluzione: Backward Riccati Equation (FORM 1 — stabile numericamente)
#   S_t = R + B_t^T P_{t+1} B_t
#   K_t = S_t^{-1} B_t^T P_{t+1} A_t
#   P_t = Q + A_t^T P_{t+1} (A_t - B_t K_t)
#
# Legge di controllo:
#   u_t = u*_t - K_t (x_t - x*_t)
#
# ROBUSTEZZA: il TV-LQR è testato con MULTIPLE condizioni iniziali perturbate
# per mostrare il bacino di attrazione del controllore.
# [Rif.: Session5/3_main_dlqr_tracking.py]
#

import numpy as np
import matplotlib.pyplot as plt
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)
plt.rcParams.update({'font.size': 13})

from dynamics        import dynamics, step
from solver_ltv_lqr  import backward_riccati   # [NUOVO] — modulo dedicato

# =============================================================================
# SEZIONE 1 — CONFIGURAZIONE
# =============================================================================
print("=" * 60)
print("   Task 3: Trajectory Tracking — TV-LQR")
print("=" * 60)

ns, ni = 4, 1

# Pesi TV-LQR
# [Rif.: Session5/3_main_dlqr_tracking.py — QQ e QQ_f (da DARE)]
Q_lqr = np.diag([1000.0, 1000.0, 100.0, 100.0])   # peso stato (stage)
R_lqr = np.eye(ni) * 0.1                             # peso ingresso

# =============================================================================
# SEZIONE 2 — CARICAMENTO TRAIETTORIA DI RIFERIMENTO
# =============================================================================
print("\nCaricamento traiettoria di riferimento (Task 2)...")
try:
    data_dict         = np.load('data/optimal_trajectory_task2.npy', allow_pickle=True).item()
    x_ref_raw    = data_dict['x']    # (4, TT) o (TT, 4) — gestiamo entrambi
    u_ref_raw    = data_dict['u']    # (1, TT) o (TT, 1)
    t_axis       = data_dict['t']    # (TT,)
    QQT_dare     = data_dict.get('QQT', None)   # DARE matrix (se salvata da Task 2)

    # ---- FIX ROBUSTO DIMENSIONI ----
    if x_ref_raw.ndim == 3:
        x_ref_raw = x_ref_raw[:,:,-1]    # Prende ultima iterazione se 3D
    if x_ref_raw.shape[0] == ns and x_ref_raw.shape[1] != ns:
        x_ref_traj = x_ref_raw.T         # (TT, 4)
        u_ref_traj = u_ref_raw.T if u_ref_raw.shape[0] == ni else u_ref_raw
    else:
        x_ref_traj = x_ref_raw           # già (TT, 4)
        u_ref_traj = u_ref_raw

    if u_ref_traj.ndim == 1:
        u_ref_traj = u_ref_traj.reshape(-1, 1)

    steps = x_ref_traj.shape[0]

    if x_ref_traj.shape[1] != 4:
        raise ValueError(f"x_ref ha shape {x_ref_traj.shape}, atteso (TT, 4)")

    print(f"  Traiettoria caricata: {steps} passi,  T = {t_axis[-1]:.1f}s")

except FileNotFoundError:
    print("ERRORE: 'optimal_trajectory_task2.npy' non trovato.")
    print("Esegui prima task2_main.py")
    exit()
except Exception as e:
    print(f"ERRORE nel caricamento: {e}")
    exit()

# =============================================================================
# SEZIONE 3 — LINEARIZZAZIONE LUNGO LA TRAIETTORIA DI RIFERIMENTO
# =============================================================================
print("\nLinearizzazione del sistema lungo la traiettoria di riferimento...")
A_list, B_list = [], []
for t in range(steps):
    x_t = x_ref_traj[t]
    u_t = u_ref_traj[t]
    _, A, B = dynamics(x_t, u_t)
    A_list.append(A)
    B_list.append(B)
print(f"  Linearizzazione completata: {steps} coppie (A_t, B_t)")

# =============================================================================
# SEZIONE 4 — BACKWARD RICCATI (TV-LQR Design)
# =============================================================================
if QQT_dare is not None:
    QQf = QQT_dare
    print("\nUsando Q_T dalla DARE (caricata da Task 2).")
else:
    QQf = Q_lqr
    print("\nUsando Q_T = Q_lqr (DARE non disponibile).")

print("Calcolo Riccati backward (TV-LQR)...")
K_gains, P_list = backward_riccati(A_list, B_list, Q_lqr, R_lqr, QQf, steps)
print(f"  Backward Riccati completato.")
print(f"  K_gains[0] = {K_gains[0].round(3)}")

# =============================================================================
# SEZIONE 5 — SIMULAZIONE TRACKING con MULTIPLE PERTURBAZIONI
# =============================================================================
perturbations = {
    'Pert. spalla -0.2 rad': np.array([-0.2,  0.0, 0.0, 0.0]),
    'Pert. gomito +0.3 rad': np.array([ 0.0,  0.3, 0.0, 0.0]),
    'Pert. vel. spalla'    : np.array([ 0.0,  0.0, 0.3, 0.0]),
}

results = {}

print("\nSimulazione tracking LQR con perturbazioni multiple...")

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
    print(f"  [{label}]: errore finale ||x_T - x*_T|| = {err_final:.4e}")
    results[label] = {'x_sim': x_sim, 'u_sim': u_sim}

# =============================================================================
# SEZIONE 6 — PLOT RISULTATI (Richiesta Assignment)
# =============================================================================
n_pert = len(perturbations)
colors_pert = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']

# --- Plot 1: Traiettorie degli stati e dell'ingresso per ciascuna perturbazione ---
fig, axs = plt.subplots(ns + ni, 1, figsize=(12, 12), sharex=True)
fig.suptitle('Task 3 — TV-LQR Tracking (Multiple Perturbazioni)', fontsize=14)

labels_x = [r'$\theta_1$ [rad]', r'$\theta_2$ [rad]',
            r'$\dot\theta_1$ [rad/s]', r'$\dot\theta_2$ [rad/s]']

for i, (label, res) in enumerate(results.items()):
    x_sim = res['x_sim']
    u_sim = res['u_sim']
    c = colors_pert[i]

    for j in range(ns):
        axs[j].plot(t_axis, x_sim[:-1, j], color=c, lw=1.8,
                    label=label, alpha=0.85)

    axs[ns].plot(t_axis, u_sim[:, 0], color=c, lw=1.8, label=label, alpha=0.85)

for j in range(ns):
    axs[j].plot(t_axis, x_ref_traj[:, j], 'k--', lw=2, label='Riferimento x*', zorder=5)
    axs[j].set_ylabel(labels_x[j], fontsize=11)
    axs[j].legend(fontsize=8, loc='best')
    axs[j].grid(alpha=0.4)

axs[ns].plot(t_axis, u_ref_traj[:, 0], 'k--', lw=2, label='Riferimento u*', zorder=5)
axs[ns].set_ylabel(r'$\tau$ [Nm]', fontsize=11)
axs[ns].set_xlabel('Tempo [s]', fontsize=12)
axs[ns].legend(fontsize=8, loc='best')
axs[ns].grid(alpha=0.4)

plt.tight_layout()
plt.savefig('task3_tracking_results.png', dpi=300)

# --- Plot 2: Errore di tracking per diverse condizioni iniziali (Richiesta Assignment) ---
fig_err, ax_err = plt.subplots(figsize=(10, 5))
fig_err.suptitle('Task 3 — Errore di Tracking TV-LQR per Diverse Condizioni Iniziali', fontsize=14)

for i, (label, res) in enumerate(results.items()):
    x_sim = res['x_sim']
    err_t = np.linalg.norm(x_sim[:-1] - x_ref_traj, axis=1)
    ax_err.plot(t_axis, err_t, color=colors_pert[i], lw=2, label=label)

ax_err.set_xlabel('Tempo [s]', fontsize=12)
ax_err.set_ylabel(r'$\|x_t - x^*_t\|$', fontsize=12)
ax_err.legend(fontsize=10, loc='best')
ax_err.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('task3_tracking_error.png', dpi=300)

plt.show(block=True)

# =============================================================================
# SEZIONE 7 — SALVATAGGIO
# =============================================================================
np.save('data/lqr_data_task3.npy', {
    'K_gains': K_gains,
    'P_list' : P_list,
    'Q_lqr'  : Q_lqr,
    'R_lqr'  : R_lqr,
    'QQf'    : QQf
})
print("\nGuadagni LQR e Riccati P_t salvati in 'lqr_data_task3.npy'")
