#
# Acrobot — TV-LQR Solver (Time-Varying Linear Quadratic Regulator)
# Progetto Optimal Control — Parameter Set 3
#
# Riferimento teorico:
#   [Slide 10] Optimal Control Based Tracking — sezione "TV-LQR"
#   [Slide 04] LQ Optimal Control             — sezione "Discrete Riccati Equation"
#   [Session5/4_solver_ltv_LQR.py]            — implementazione del professore (base)
#   [Session5/3_main_dlqr_tracking.py]        — pattern di utilizzo
#
# PROBLEMA TV-LQR PER TRACKING
# ==============================
# Dato un sistema LTV (tempo variante) ottenuto linearizzando l'Acrobot
# lungo la traiettoria di riferimento (x*, u*):
#
#   δx_{t+1} = A_t δx_t + B_t δu_t
#
# con δx_t = x_t - x*_t,  δu_t = u_t - u*_t (dinamica dell'errore)
#
# Si vuole minimizzare il costo di tracking:
#
#   J = Σ_{t=0}^{T-1} [δx_tᵀ Q δx_t + δu_tᵀ R δu_t] + δx_Tᵀ Q_T δx_T
#
# SOLUZIONE: Backward Riccati Equation [Slide 04, Slide 10]
#
#   P_T = Q_T
#   Per t = T-1, ..., 0:
#       S_t   = R + B_tᵀ P_{t+1} B_t              ← denominatore
#       K_t   = S_t⁻¹ (B_tᵀ P_{t+1} A_t)         ← guadagno ottimo
#       P_t   = Q + A_tᵀ P_{t+1} A_t - (A_tᵀ P_{t+1} B_t) K_t   ← Riccati
#
# LEGGE DI CONTROLLO (tracking):
#   u_t = u*_t - K_t (x_t - x*_t) = u*_t + K_t δx_t
#
# NOTA: questa formulazione in deviazione è ESATTA per sistemi lineari
# e APPROSSIMATA per sistemi nonlineari (validità locale attorno a x*).
# [Rif.: Slide 10 — "LQR Tracking via Linearization"]
#

import numpy as np


def backward_riccati(A_list, B_list, QQ, RR, QQf, steps):
    """
    Backward Riccati Equation per TV-LQR.

    Calcola la sequenza di guadagni ottimi K_t e la matrice di Riccati P_t
    risolvendo l'equazione di Riccati discreta all'indietro nel tempo.

    Forma standard dell'equazione di Riccati (FORM 1 — da [Session5/4_solver_ltv_LQR.py]):
        S_t = R + B_tᵀ P_{t+1} B_t
        K_t = S_t⁻¹ B_tᵀ P_{t+1} A_t
        P_t = Q + A_tᵀ P_{t+1} A_t - (A_tᵀ P_{t+1} B_t) K_t

    Equivalentemente (FORM 2 — usata anche in V4):
        P_t = Q + (A_t - B_t K_t)ᵀ P_{t+1} (A_t - B_t K_t) + K_tᵀ R K_t
    Le due forme sono MATEMATICAMENTE EQUIVALENTI ma la FORM 1 è più
    numericamente stabile (evita la sottrazione tra matrici grandi).
    Per questo motivo usiamo FORM 1, in accordo con [Session5/4_solver_ltv_LQR.py].

    [Rif.: Slide 04 — "Discrete-time Riccati Equation"]
    [Rif.: Slide 10 — "TV-LQR Backward Pass"]
    [Rif.: Session5/4_solver_ltv_LQR.py — ltv_LQR(), righe 121-141]

    Args:
        A_list : list di ndarray (ns, ns) — Jacobiani ∂F/∂x sulla traiettoria ref.
        B_list : list di ndarray (ns, ni) — Jacobiani ∂F/∂u sulla traiettoria ref.
        QQ     : ndarray (ns, ns) — matrice di peso stato (stage)
        RR     : ndarray (ni, ni) — matrice di peso ingresso
        QQf    : ndarray (ns, ns) — matrice di peso terminale (idealmente = DARE)
        steps  : int — numero di passi temporali (= len(A_list))

    Returns:
        K_gains : list di ndarray (ni, ns) — guadagni K_t per t=0,...,steps-1
        P_list  : list di ndarray (ns, ns) — Riccati P_t (usata in Task 4 come terminale)
    """
    K_gains = [None] * steps   # K_t per t = 0, ..., steps-1
    P_list  = [None] * steps   # P_t per riferimento e terminal cost MPC

    # Condizione al contorno: P_T = Q_T
    P = QQf.copy()

    # Ricorsione backward: da t = steps-1 a t = 0
    for t in reversed(range(steps)):
        A = A_list[t]   # (ns×ns)
        B = B_list[t]   # (ns×ni)

        # S_t = R + B_tᵀ P_{t+1} B_t   (ni×ni) — "denominatore" di Riccati
        S = RR + B.T @ P @ B

        # K_t = S_t⁻¹ B_tᵀ P_{t+1} A_t — guadagno feedback ottimo
        # Usa np.linalg.solve invece di inv per maggiore stabilità numerica
        K_t = np.linalg.solve(S, B.T @ P @ A)   # (ni×ns)

        K_gains[t] = K_t
        P_list[t]  = P.copy()   # salva P_{t+1} associato a questo step

        # Aggiornamento Riccati: FORM 1 (numericamente stabile)
        # P_t = Q + A_tᵀ P_{t+1} A_t - (A_tᵀ P_{t+1} B_t) K_t
        #     = Q + A_tᵀ P_{t+1} (A_t - B_t K_t)
        P = QQ + A.T @ P @ (A - B @ K_t)

    return K_gains, P_list
