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
import animation as an

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
    data_dict2       = np.load('data/optimal_trajectory_task2.npy', allow_pickle=True).item()
    x_ref_raw   = data_dict2['x']
    u_ref_raw   = data_dict2['u']
    t_axis      = data_dict2['t']

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
    data_dict3   = np.load('data/lqr_data_task3.npy', allow_pickle=True).item()
    P_list  = data_dict3['P_list']
    K_gains = data_dict3['K_gains']
    print(f"  Dati LQR Task 3 caricati: {len(P_list)} Riccati matrices")
    USE_RICCATI_TERMINAL = True

except FileNotFoundError:
    print("  WARNING: 'data/lqr_data_task3.npy' non trovato.")
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

        # Safe boundary condition: if remaining steps is less than 1, bypass solver
        current_horizon = min(T_pred, steps - 1 - t)
        if current_horizon < 1:
            delta_u0 = np.zeros(ni)
            status = 'ok'
        else:
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

def plot_results_task4(time, xx_ref, uu_ref, xx_sim, uu_sim, label_name, pert_idx, U_MAX):
    """
    Plots comparison Reference vs MPC Tracking.
    """
    # --- FIGURE 1: STATES AND VELOCITY ---
    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle(f'Task 4: MPC Tracking Performance - {label_name}', fontsize=16, fontweight='bold')

    # --- 1.1 Theta 1 ---
    ax = axs[0, 0]
    ax.plot(time, np.degrees(xx_ref[:, 0]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 0]), 'b-', linewidth=2, label='MPC Link 1')
    ax.set_ylabel(r'Pos $\theta_1$ [deg]')
    ax.set_title('Link 1 Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 1.2 Theta 2 ---
    ax = axs[0, 1]
    ax.plot(time, np.degrees(xx_ref[:, 1]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 1]), 'r-', linewidth=2, label='MPC Link 2')
    ax.set_ylabel(r'Pos $\theta_2$ [deg]')
    ax.set_title('Link 2 Position', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 2.1 Error Theta 1 ---
    ax = axs[1, 0]
    err_pos1 = np.degrees(xx_sim[:, 0] - xx_ref[:, 0])
    ax.plot(time, err_pos1, color='dodgerblue', linewidth=1.5, label=r'Error $\theta_1$')
    ax.set_ylabel(r'Err $\theta_1$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 2.2 Error Theta 2 ---
    ax = axs[1, 1]
    err_pos2 = np.degrees(xx_sim[:, 1] - xx_ref[:, 1])
    ax.plot(time, err_pos2, color='tomato', linewidth=1.5, label=r'Error $\theta_2$')
    ax.set_ylabel(r'Err $\theta_2$ [deg]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 3.1 Velocity Theta 1 ---
    ax = axs[2, 0]
    ax.plot(time, np.degrees(xx_ref[:, 2]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 2]), 'b-', linewidth=2, label='MPC Vel 1')
    ax.set_ylabel(r'Vel $\dot{\theta}_1$ [deg/s]')
    ax.set_title('Link 1 Velocity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 3.2 Velocity Theta 2 ---
    ax = axs[2, 1]
    ax.plot(time, np.degrees(xx_ref[:, 3]), 'k--', label='Ref', alpha=0.7)
    ax.plot(time, np.degrees(xx_sim[:, 3]), 'r-', linewidth=2, label='MPC Vel 2')
    ax.set_ylabel(r'Vel $\dot{\theta}_2$ [deg/s]')
    ax.set_title('Link 2 Velocity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')

    # --- 4.1 Error Vel 1 ---
    ax = axs[3, 0]
    err_vel1 = np.degrees(xx_sim[:, 2] - xx_ref[:, 2])
    ax.plot(time, err_vel1, color='dodgerblue', linewidth=1.5, label=r'Error $\dot{\theta}_1$')
    ax.set_ylabel(r'Err $\dot{\theta}_1$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

    # --- 4.2 Error Vel 2 ---
    ax = axs[3, 1]
    err_vel2 = np.degrees(xx_sim[:, 3] - xx_ref[:, 3])
    ax.plot(time, err_vel2, color='tomato', linewidth=1.5, label=r'Error $\dot{\theta}_2$')
    ax.set_ylabel(r'Err $\dot{\theta}_2$ [deg/s]')
    ax.set_xlabel('Time [s]')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'figs/task4_states_pert_{pert_idx}.png', dpi=300)
    plt.show(block=False)

    # --- FIGURE 2: INPUT AND INPUT ERROR ---
    fig2, axs2 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle(f'Task 4: Control Input ({label_name})', fontsize=14, fontweight='bold')
    
    uu_ref_plot = np.copy(uu_ref); uu_ref_plot[-1, :] = uu_ref_plot[-2, :]
    uu_sim_plot = np.copy(uu_sim); uu_sim_plot[-1, :] = uu_sim_plot[-2, :]

    # 1. Input Comparison
    ax = axs2[0]
    ax.fill_between(time, -U_MAX, U_MAX, color='tab:green', alpha=0.1, label='Feasible')
    ax.axhline(U_MAX, color='darkred', linestyle='--', linewidth=1.5, label=r'$u_{max}$')
    ax.axhline(-U_MAX, color='lightcoral', linestyle='--', linewidth=1.5, label=r'$u_{min}$')

    ax.step(time, uu_ref_plot[:, 0], 'k--', where='post', label='Ref', alpha=0.6)
    ax.step(time, uu_sim_plot[:, 0], 'g-', where='post', linewidth=2, label='MPC Input')
    
    ax.set_ylabel(r'Torque [Nm]')
    ax.set_title('Control Input (Constrained)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=3)

    # 2. Input Error
    ax = axs2[1]
    err_u = uu_sim_plot[:, 0] - uu_ref_plot[:, 0]
    ax.step(time, err_u, 'seagreen', where='post', label=r'$\Delta u$')
    ax.set_ylabel(r'Err $\tau$ [Nm]')
    ax.set_xlabel('Time [s]')
    ax.set_title(r'Input Deviation ($\Delta u$)', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'figs/task4_input_pert_{pert_idx}.png', dpi=300)
    plt.show(block=False)

import os
os.makedirs('figs', exist_ok=True)

for i, (label, res) in enumerate(results.items()):
    x_sim_plot = res['x_sim'][:-1, :]
    u_sim_plot = res['u_sim']
    plot_results_task4(t_axis, x_ref_traj, u_ref_traj, x_sim_plot, u_sim_plot, label, i+1, U_MAX)

    # Animate the trajectory vs the reference
    print(f"\nAvvio animazione per: {label}")
    # Traspone gli array per l'animazione (4, T)
    an.animate_trajectory(t_axis, x_sim_plot.T, x_ref_traj.T, title=f"Task 4 MPC Tracking: {label}")

# =============================================================================
# SEZIONE 7 — SALVATAGGIO
# =============================================================================
np.save('data/mpc_results_task4.npy', {
    'results': results,
    't_axis' : t_axis,
    'T_pred' : T_pred,
    'U_MAX'  : U_MAX,
})
print("\nRisultati Task 4 MPC salvati in 'mpc_results_task4.npy'")
