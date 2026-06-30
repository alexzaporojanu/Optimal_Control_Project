#
# Task 5 — Animazione Acrobot (Cinematica Diretta + matplotlib.animation)
#
# Progetto Optimal Control — Parameter Set 3
# Autori: [inserire nomi]  — UniBO 2025/26
#
# Riferimento teorico:
#   [Slide 05] Dynamics of Mechanical Systems — sezione "Forward Kinematics"
#   [Session4/animation_acrobot.py]           — base dell'implementazione
#
# CINEMATICA DIRETTA DELL'ACROBOT
# ================================
# Dato lo stato x = [θ₁, θ₂, θ̇₁, θ̇₂], le posizioni cartesiane dei
# giunti e dell'end-effector sono (perno fisso in [0, 0]):
#
# Link 1 (spalla → gomito):
#   x_joint1 = l₁ sin(θ₁)
#   y_joint1 = -l₁ cos(θ₁)
#   Convenzione: θ₁=0 → link 1 verso il BASSO (y negativo)
#
# Link 2 (gomito → end-effector):
#   L'angolo ASSOLUTO del link 2 è θ₁ + θ₂ (θ₂ è RELATIVO)
#   x_ee = x_joint1 + l₂ sin(θ₁ + θ₂)
#   y_ee = y_joint1 - l₂ cos(θ₁ + θ₂)
#
# Per l'animazione, visualizziamo la sequenza di stati ottimi da Task 1/2
# e, opzionalmente, la traiettoria tracking da Task 3/4.
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)

# FIX CRITICO rispetto a V4:
# Importa da 'dynamics' (V5) e non da 'dynamics_acrobot' (V1 legacy)
from dynamics import dynamics, l1, l2, dt

# =============================================================================
# SEZIONE 1 — CINEMATICA DIRETTA
# =============================================================================
def forward_kinematics(theta1, theta2):
    """
    Calcola le posizioni cartesiane dell'Acrobot (cinematica diretta).

    Convenzione angolare (coerente con dynamics.py):
        θ₁ = 0    → link 1 verso il BASSO (equilibrio stabile)
        θ₁ = π    → link 1 verso l'ALTO   (equilibrio instabile)
        θ₂ = 0    → link 2 allineato con link 1 (gomito disteso)

    Perno fisso in [0, 0].
    [Rif.: Slide 05 — "Forward Kinematics of Double Pendulum"]

    Args:
        theta1 : float — angolo assoluto link 1 [rad]
        theta2 : float — angolo relativo link 2 rispetto a link 1 [rad]

    Returns:
        (x0, y0) : (0, 0)             — perno fisso
        (x1, y1) : posizione gomito
        (x2, y2) : posizione end-effector
    """
    # Wrap angles per evitare salti di visualizzazione > 2π
    # (non influenza la dinamica, solo la rappresentazione grafica)
    theta1 = theta1 % (2 * np.pi)
    theta2 = theta2 % (2 * np.pi)

    x0, y0 = 0.0, 0.0

    # Posizione del gomito (fine link 1)
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    # Posizione dell'end-effector (fine link 2)
    # θ_abs_2 = θ₁ + θ₂  (angolo assoluto del link 2)
    theta_abs_2 = theta1 + theta2
    x2 = x1 + l2 * np.sin(theta_abs_2)
    y2 = y1 - l2 * np.cos(theta_abs_2)

    return (x0, y0), (x1, y1), (x2, y2)


# =============================================================================
# SEZIONE 2 — CARICAMENTO DATI
# =============================================================================
print("=" * 60)
print("   Task 5: Animazione Acrobot — Cinematica Diretta")
print("=" * 60)

# Carica la traiettoria di Task 2 (smooth) come traiettoria principale
# Fallback: traiettoria di Task 1 se Task 2 non disponibile

loaded_traj = None
traj_label  = ""

for fname, label in [
    ('optimal_trajectory_task2.npy', 'Task 2 (Smooth Reference)'),
    ('optimal_trajectory_task1.npy', 'Task 1 (Step Reference)'),
]:
    try:
        data = np.load(fname, allow_pickle=True).item()
        x_traj_raw = data['x']
        t_axis     = data['t']

        # Fix dimensioni: converti in (TT, 4) per indicizzazione [t, :]
        if x_traj_raw.shape[0] == 4 and x_traj_raw.shape[1] != 4:
            x_traj = x_traj_raw.T
        else:
            x_traj = x_traj_raw

        loaded_traj = x_traj
        traj_label  = label
        print(f"\nTraiettoria caricata da '{fname}':")
        print(f"  {label}: {x_traj.shape[0]} passi, T = {t_axis[-1]:.1f}s")
        break

    except FileNotFoundError:
        print(f"  '{fname}' non trovato, provo il successivo...")

if loaded_traj is None:
    print("\nNessuna traiettoria trovata. Esegui prima task1_main.py o task2_main.py")
    exit()

x_traj = loaded_traj
steps  = x_traj.shape[0]

# Carica opzionalmente la traiettoria di tracking da Task 3 o Task 4
# per comparazione visiva nell'animazione
x_tracking = None

for fname, label in [
    ('mpc_results_task4.npy',  'Task 4 MPC Tracking'),
]:
    try:
        data_track = np.load(fname, allow_pickle=True).item()
        # Prende il primo risultato disponibile (prima perturbazione)
        first_key  = list(data_track['results'].keys())[0]
        x_track_raw = data_track['results'][first_key]['x_sim']
        # x_track_raw shape: (steps+1, ns)
        x_tracking  = x_track_raw[:-1, :]   # (steps, 4)
        tracking_label = f"{label} — {first_key}"
        print(f"  Traiettoria tracking caricata: {tracking_label}")
        break
    except (FileNotFoundError, KeyError):
        pass

# =============================================================================
# SEZIONE 3 — CALCOLO POSIZIONI CARTESIANE
# =============================================================================
# Pre-calcola le posizioni in tutti i passi temporali per efficienza
print("\nCalcolo cinematica diretta...")

pos_joint0 = np.zeros((steps, 2))   # perno fisso (sempre 0)
pos_joint1 = np.zeros((steps, 2))   # posizione gomito
pos_ee     = np.zeros((steps, 2))   # posizione end-effector

for t in range(steps):
    th1, th2 = x_traj[t, 0], x_traj[t, 1]
    p0, p1, p2 = forward_kinematics(th1, th2)
    pos_joint0[t] = p0
    pos_joint1[t] = p1
    pos_ee[t]     = p2

# Analoghi per la traiettoria di tracking (se disponibile)
if x_tracking is not None:
    steps_tr = min(x_tracking.shape[0], steps)
    pos_j1_tr = np.zeros((steps_tr, 2))
    pos_ee_tr = np.zeros((steps_tr, 2))
    for t in range(steps_tr):
        th1, th2 = x_tracking[t, 0], x_tracking[t, 1]
        _, p1, p2 = forward_kinematics(th1, th2)
        pos_j1_tr[t] = p1
        pos_ee_tr[t] = p2

# =============================================================================
# SEZIONE 4 — ANIMAZIONE
# =============================================================================
fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
ax_anim.set_aspect('equal')
margin = l1 + l2 + 0.5
ax_anim.set_xlim(-margin, margin)
ax_anim.set_ylim(-margin, margin)
ax_anim.set_title(f'Acrobot Animation — {traj_label}', fontsize=13)
ax_anim.set_xlabel('x [m]', fontsize=11)
ax_anim.set_ylabel('y [m]', fontsize=11)
ax_anim.grid(alpha=0.3)

# Elementi grafici dell'Acrobot (traiettoria ottima)
link1_line,   = ax_anim.plot([], [], 'o-', color='#1f77b4', lw=4,
                              markersize=8, label='Link 1 (passivo)')
link2_line,   = ax_anim.plot([], [], 'o-', color='#d62728', lw=4,
                              markersize=8, label='Link 2 (attuato)')
ee_trace,     = ax_anim.plot([], [], '-', color='#2ca02c', lw=1,
                              alpha=0.5, label='Traccia EE (ottimo)')

# Traiettoria tracking (se disponibile)
if x_tracking is not None:
    link1_tr, = ax_anim.plot([], [], '--', color='#9467bd', lw=2, alpha=0.7,
                               label='Link 1 tracking')
    link2_tr, = ax_anim.plot([], [], '--', color='#ff7f0e', lw=2, alpha=0.7,
                               label='Link 2 tracking')

# Perno fisso
ax_anim.plot(0, 0, 'k^', markersize=12, zorder=10)

# Traccia progressiva dell'end-effector
trace_x, trace_y = [], []

# Testo con info temporale
time_text = ax_anim.text(0.02, 0.96, '', transform=ax_anim.transAxes,
                          fontsize=11, verticalalignment='top')

ax_anim.legend(fontsize=9, loc='upper right')

# Frame skip per velocizzare l'animazione (mostra 1 frame ogni skip)
# dt=0.01s → con skip=5 l'animazione scorre a 50ms/frame ≈ 20fps
FRAME_SKIP = 5
frame_indices = range(0, steps, FRAME_SKIP)


def _init_anim():
    """Inizializzazione animazione matplotlib."""
    link1_line.set_data([], [])
    link2_line.set_data([], [])
    ee_trace.set_data([], [])
    time_text.set_text('')
    if x_tracking is not None:
        link1_tr.set_data([], [])
        link2_tr.set_data([], [])
    return (link1_line, link2_line, ee_trace, time_text) + \
           ((link1_tr, link2_tr) if x_tracking is not None else ())


def _update_anim(frame):
    """Aggiorna un frame dell'animazione."""
    t = frame

    # Traiettoria ottima
    p0 = pos_joint0[t]
    p1 = pos_joint1[t]
    p2 = pos_ee[t]

    link1_line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
    link2_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])

    trace_x.append(p2[0])
    trace_y.append(p2[1])
    ee_trace.set_data(trace_x, trace_y)

    time_text.set_text(f't = {t * dt:.2f}s  |  θ₁={x_traj[t,0]:.2f}rad')

    # Tracking
    if x_tracking is not None and t < steps_tr:
        p1_tr = pos_j1_tr[t]
        p2_tr = pos_ee_tr[t]
        link1_tr.set_data([0, p1_tr[0]], [0, p1_tr[1]])
        link2_tr.set_data([p1_tr[0], p2_tr[0]], [p1_tr[1], p2_tr[1]])
        return (link1_line, link2_line, ee_trace, time_text, link1_tr, link2_tr)

    return (link1_line, link2_line, ee_trace, time_text)


anim = animation.FuncAnimation(
    fig_anim, _update_anim, frames=frame_indices,
    init_func=_init_anim,
    interval=FRAME_SKIP * dt * 1000,   # [ms] — velocità animazione
    blit=True
)

plt.tight_layout()
plt.show(block=True)

# Opzionale: salva l'animazione come GIF (richiede Pillow o ffmpeg)
try:
    save_path = 'acrobot_animation.gif'
    print(f"\nSalvataggio animazione in '{save_path}'...")
    anim.save(save_path, writer='pillow', fps=int(1/(FRAME_SKIP*dt)))
    print("  Salvata.")
except Exception as e:
    print(f"  Salvataggio GIF non riuscito (ffmpeg/pillow non disponibili): {e}")
