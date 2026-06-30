#
# Acrobot — Generazione Traiettoria di Riferimento
# Progetto Optimal Control — Parameter Set 3
#
# Riferimento teorico:
#   [Slide 06] Optimal Control Shooting — sezione "Reference Trajectory"
#   [Session4/9_reference_trajectory.py] — pattern del professore
#
# Due tipologie di riferimento:
#
# 1. GRADINO (Step) — Task 1
#    x_ref(t) = x_start  per t < T/2
#    x_ref(t) = x_goal   per t ≥ T/2
#    Il gradino crea una discontinuità che l'ottimizzatore deve bridgare.
#    Fisicamente: "il robot deve star giù poi in su".
#
# 2. SMOOTH QUINTICO — Task 2
#    Usa un polinomio di 5° grado (smooth step) per la transizione:
#    s(τ) = 10τ³ - 15τ⁴ + 6τ⁵  (τ ∈ [0,1])
#    Proprietà: s(0)=s'(0)=s''(0)=0 e s(1)=1, s'(1)=s''(1)=0
#    → Continua con prima e seconda derivata nulla agli estremi (C²)
#    Utile perché fornisce all'ottimizzatore un riferimento più "realistico"
#    e una traiettoria iniziale meno discontinua.
#

import numpy as np


def generate_step(tf, dt, x_start, x_goal):
    """
    Genera il riferimento a GRADINO per Task 1.

    Il riferimento cambia istantaneamente da x_start a x_goal a metà tempo (T/2).
    L'input di riferimento è zero (equilibri passivi dell'Acrobot).

    [Rif.: Session4/9_reference_trajectory.py — step_reference=True]

    Args:
        tf      : float    — orizzonte temporale [s]
        dt      : float    — passo di campionamento [s]
        x_start : ndarray (ns,) — stato iniziale (equilibrio giù)
        x_goal  : ndarray (ns,) — stato obiettivo (equilibrio su)

    Returns:
        xx_ref : ndarray (ns, TT+1) — sequenza di riferimento stati
        uu_ref : ndarray (ni, TT)   — sequenza di riferimento ingressi (zero)
    """
    steps = int(tf / dt)
    ns    = len(x_start)
    ni    = 1

    xx_ref = np.zeros((ns, steps + 1))
    uu_ref = np.zeros((ni, steps))

    half = steps // 2

    # Prima metà: equilibrio giù
    xx_ref[:, :half] = x_start.reshape(-1, 1)
    # Seconda metà (incluso T): equilibrio su
    xx_ref[:, half:] = x_goal.reshape(-1, 1)

    # Input di riferimento sempre zero (entrambi gli equilibri sono passivi)
    uu_ref[:, :] = 0.0

    return xx_ref, uu_ref


def generate_smooth(tf, dt, x_start, x_goal,
                    t_start_frac=0.2, t_end_frac=0.8):
    """
    Genera il riferimento SMOOTH (polinomio quintico) per Task 2.

    Usa un polinomio di quintico di Hermite per la transizione:
        s(τ) = 10τ³ - 15τ⁴ + 6τ⁵   con τ ∈ [0, 1] normalizzato
    Proprietà:
        s(0)=0, s'(0)=0, s''(0)=0   ← inizia con vel. e acc. nulle
        s(1)=1, s'(1)=0, s''(1)=0   ← finisce con vel. e acc. nulle
    → Interpolazione C² (continua con prima e seconda derivata) — ideale per
    traiettorie di riferimento da fornire all'ottimizzatore.

    Struttura temporale:
        [0, t_start_frac·tf]         : costante a x_start
        [t_start_frac·tf, t_end·tf]  : transizione smooth
        [t_end_frac·tf, tf]          : costante a x_goal

    [Rif.: Slide 06 — "Smooth Reference Trajectory"]

    Args:
        tf           : float — orizzonte temporale [s]
        dt           : float — passo [s]
        x_start      : ndarray (ns,) — stato iniziale
        x_goal       : ndarray (ns,) — stato finale
        t_start_frac : float — frazione di tf per inizio transizione (default 0.2)
        t_end_frac   : float — frazione di tf per fine transizione (default 0.8)

    Returns:
        xx_ref : ndarray (ns, TT+1) — riferimento stati
        uu_ref : ndarray (ni, TT)   — riferimento ingressi (zero)
    """
    steps = int(tf / dt)
    ns    = len(x_start)
    ni    = 1

    xx_ref = np.zeros((ns, steps + 1))
    uu_ref = np.zeros((ni, steps))

    t_start = t_start_frac * tf
    t_end   = t_end_frac   * tf
    duration = t_end - t_start

    for k in range(steps + 1):
        t = k * dt
        if t <= t_start:
            xx_ref[:, k] = x_start
        elif t >= t_end:
            xx_ref[:, k] = x_goal
        else:
            # Polinomio quintico di Hermite
            tau = (t - t_start) / duration   # τ ∈ [0, 1]
            s   = 10*tau**3 - 15*tau**4 + 6*tau**5
            xx_ref[:, k] = x_start + (x_goal - x_start) * s

    uu_ref[:, :] = 0.0

    return xx_ref, uu_ref


def generate_extended(dt, x_start, x_goal,
                      t_pre=5.0, t_move=10.0, t_post=5.0):
    """
    Genera il riferimento ESTESO (3 fasi) per Task 2.

    Struttura:
        [0, t_pre]              : Pre-wait — costante a x_start
        [t_pre, t_pre+t_move]   : Smooth Move — transizione quintica
        [t_pre+t_move, tf]      : Post-hold — costante a x_goal

    Questo approccio a 3 fasi:
      1. Dà al sistema tempo per "stabilizzarsi" all'equilibrio iniziale
      2. Esegue la transizione smooth (ottimizzata con polinomio quintico)
      3. Dà al sistema tempo per "assorbire" l'arrivo al nuovo equilibrio

    Returns:
        xx_ref : ndarray (ns, TT)  — riferimento stati (senza +1 — gestito internamente)
        uu_ref : ndarray (ni, TT)  — riferimento ingressi
        TT     : int               — numero totale di passi
        tf     : float             — orizzonte totale [s]
        N_pre  : int               — passi della fase pre-wait
        N_move : int               — passi della fase smooth-move
    """
    N_pre  = int(t_pre  / dt)
    N_move = int(t_move / dt)
    N_post = int(t_post / dt)
    tf     = t_pre + t_move + t_post

    ns = len(x_start)
    ni = 1

    # Fase Pre: costante a x_start
    xx_pre  = np.tile(x_start.reshape(-1, 1), (1, N_pre))
    uu_pre  = np.zeros((ni, N_pre))

    # Fase Move: smooth quintico
    xx_move, uu_move = generate_smooth(t_move, dt, x_start, x_goal)
    # xx_move ha N_move+1 colonne — usiamo le prime N_move

    # Fase Post: costante a x_goal
    xx_post = np.tile(x_goal.reshape(-1, 1), (1, N_post))
    uu_post = np.zeros((ni, N_post))

    # Concatenazione:
    # [Pre(N_pre)] + [Move(N_move)] + [Post(N_post)] = TT totale
    # xx_move[:, :-1] evita di duplicare il punto di giunzione tra Move e Post
    xx_ref = np.hstack([xx_pre, xx_move[:, :-1], xx_post])
    uu_ref = np.hstack([uu_pre, uu_move,          uu_post])

    TT = uu_ref.shape[1]

    return xx_ref, uu_ref, TT, tf, N_pre, N_move
