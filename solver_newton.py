#
# Acrobot — Backward Pass Newton / iDDP (Iterative DDP)
# Progetto Optimal Control — Parameter Set 3
#
# Riferimento teorico:
#   [Slide 08] Second-Order Closed-Loop Methods — sezione "iDDP / Newton"
#   [Slide 07] Gradient Method — sezione "Backward Pass (costate equations)"
#   [Jacobson & Mayne, 1970] — Differential Dynamic Programming
#   [Session4/10_main_gradient_optcon_method.py] — struttura backward pass
#
# ALGORITMO iDDP — BACKWARD PASS
# ================================
# Il backward pass risolve, partendo da t=T e andando indietro fino a t=0,
# il sottoproblema LQR locale ottenuto LINEARIZZANDO la dinamica e
# ESPANDENDO il costo al 2° ordine intorno alla traiettoria corrente.
#
# TEOREMA (DDP — Differential Dynamic Programming):
# La funzione valore ottima V_t(δx) può essere approssimata localmente come:
#
#   V_t(δx) ≈ ½ δxᵀ P_t δx + p_tᵀ δx + cost_to_go
#
# dove (P_t, p_t) soddisfano la ricorsione di Riccati:
#   P_t = Q_{xx,t} + K_tᵀ Q_{uu,t} K_t + K_tᵀ Q_{ux,t} + Q_{ux,t}ᵀ K_t
#   p_t = Q_{x,t}  + K_tᵀ Q_{uu,t} k_t + K_tᵀ Q_{u,t}  + Q_{ux,t}ᵀ k_t
#
# con i termini della Q-FUNCTION EXPANSION [Slide 08, Sec. 3]:
#   Q_{x}   = lx_t  + A_tᵀ p_{t+1}
#   Q_{u}   = lu_t  + B_tᵀ p_{t+1}
#   Q_{xx}  = lxx_t + A_tᵀ P_{t+1} A_t
#   Q_{uu}  = luu_t + B_tᵀ P_{t+1} B_t
#   Q_{ux}  = B_tᵀ P_{t+1} A_t
#
# MINIMIZZAZIONE LOCALE rispetto a δu:
#   ∂Q/∂δu = Q_{uu} δu + Q_{ux} δx + Q_{u} = 0
#   δu* = -Q_{uu}⁻¹ Q_{u}  -  Q_{uu}⁻¹ Q_{ux} δx
#       = k_t               +  K_t δx
#
# Il passo feedforward k_t = -Q_{uu}⁻¹ Q_{u}  è la DIREZIONE DI NEWTON
# (equivale al gradiente precondizionato dalla curvatura locale).
#

import numpy as np


def solve_newton_step(AA, BB, lx, lu, lxx, luu, TT):
    """
    Backward Pass iDDP — calcola il Newton step per Optimal Control.

    Propaga le equazioni della Q-Function all'indietro nel tempo
    per calcolare i guadagni ottimi (feedback K_t e feedforward k_t).

    Questi vengono poi usati nel Forward Pass (armijo.py):
        u_t^{k+1} = u_t^k + α · k_t + K_t (x_t^{k+1} - x_t^k)

    [Rif.: Slide 08 — Algorithm 1 "iDDP Backward Pass"]
    [Rif.: Session4/10_main_gradient_optcon_method.py — struttura analoga]

    REGOLARIZZAZIONE LEVENBERG-MARQUARDT:
    Q_{uu} viene regolarizzata con ε·I per garantire invertibilità:
        Q_{uu,reg} = Q_{uu} + ε·I  (con ε = 1e-2)
    Questo è necessario perché R = RRt = 1e-3·I è molto piccolo, quindi
    Q_{uu} può essere mal condizionata nelle prime iterazioni.
    La regolarizzazione agisce come il metodo di Marquardt: se ε è grande
    il passo diventa simile al gradiente (LM damping), se ε→0 è Newton puro.
    [Rif.: Slide 08 — "Regularization / Levenberg-Marquardt"]

    Args:
        AA  : ndarray (ns, ns, TT) — Jacobiani ∂F/∂x lungo la traiettoria
        BB  : ndarray (ns, ni, TT) — Jacobiani ∂F/∂u lungo la traiettoria
        lx  : ndarray (ns, TT)    — gradienti ∂l/∂x (e ∂l_T/∂x all'ultimo step)
        lu  : ndarray (ni, TT)    — gradienti ∂l/∂u
        lxx : ndarray (ns,ns,TT)  — Hessiane ∂²l/∂x² (costanti = Q per cost. quad.)
        luu : ndarray (ni,ni,TT)  — Hessiane ∂²l/∂u² (costanti = R per cost. quad.)
        TT  : int                 — numero di passi temporali

    Returns:
        KK     : ndarray (ni, ns, TT) — guadagni feedback K_t per il forward pass
        kk_vec : ndarray (ni, TT)    — direzioni feedforward k_t (Newton steps)
    """
    ns_loc = AA.shape[0]
    ni_loc = BB.shape[1]

    KK     = np.zeros((ni_loc, ns_loc, TT))   # guadagni feedback
    kk_vec = np.zeros((ni_loc, TT))            # passi feedforward

    # ---- Inizializzazione con il Costo Terminale ----
    # P_T = Q_T = lxx[:,:,-1]  (matrice della Value Function al tempo T)
    # p_T = ∂l_T/∂x = lx[:,-1]  (gradiente lineare della Value Function a T)
    # [Rif.: Slide 08 — "Boundary conditions of the backward pass"]
    P = lxx[:, :, -1].copy()   # (ns×ns) — inizializzato con Q_T (o DARE)
    p = lx[:, -1].copy()       # (ns,)   — inizializzato con gradiente terminale

    # ---- Backward Recursion: da t=T-1 a t=0 ----
    for t in reversed(range(TT - 1)):

        At = AA[:, :, t]   # A_t = ∂F/∂x  (4×4)
        Bt = BB[:, :, t]   # B_t = ∂F/∂u  (4×1)

        # Termini del costo locale al tempo t
        q_x  = lx[:, t]       # ∂l/∂x   (4,)
        q_u  = lu[:, t]       # ∂l/∂u   (1,)
        q_xx = lxx[:, :, t]   # Q = ∂²l/∂x²  (4×4)
        q_uu = luu[:, :, t]   # R = ∂²l/∂u²  (1×1)

        # ---- Q-Function Expansion [Slide 08, eq. Q-function] ----
        # Propaga la Value Function (P_{t+1}, p_{t+1}) indietro di un passo
        Qx  = q_x + At.T @ p           # gradiente Q rispetto a x   (4,)
        Qu  = q_u + Bt.T @ p           # gradiente Q rispetto a u   (1,)
        Qxx = q_xx + At.T @ P @ At     # Hessiana Q rispetto a x²   (4×4)
        Quu = q_uu + Bt.T @ P @ Bt     # Hessiana Q rispetto a u²   (1×1)
        Qux = Bt.T @ P @ At            # termine misto u-x           (1×4)

        # ---- Regolarizzazione Levenberg-Marquardt [Slide 08] ----
        # Aggiunge ε·I per garantire Q_uu > 0 (passo Newton valido)
        # Usata SOLO per la soluzione del sistema lineare, non per l'update di P
        Quu_reg = Quu + 1e-2 * np.eye(ni_loc)

        # ---- Calcolo Guadagni Ottimi ----
        # Risolve il sistema lineare (più stabile di inv):
        #   Q_{uu} k_t = -Q_{u}    →  k_t = -Q_{uu}⁻¹ Q_{u}
        #   Q_{uu} K_t = -Q_{ux}   →  K_t = -Q_{uu}⁻¹ Q_{ux}
        try:
            k_t = -np.linalg.solve(Quu_reg, Qu)    # feedforward step (ni,)
            K_t = -np.linalg.solve(Quu_reg, Qux)   # feedback gain    (ni×ns)
        except np.linalg.LinAlgError:
            # Fallback con regolarizzazione più forte (raro con ε=1e-2)
            print(f"  [solver_newton] WARNING: Hessiana singolare a t={t}. "
                  "Aumento regolarizzazione a 1e-1.")
            Quu_extra = Quu_reg + 1e-1 * np.eye(ni_loc)
            k_t = -np.linalg.solve(Quu_extra, Qu)
            K_t = -np.linalg.solve(Quu_extra, Qux)

        KK[:, :, t]  = K_t
        kk_vec[:, t] = k_t

        # ---- Aggiornamento Value Function (Bellman) ----
        # Sostituisce il controllo ottimo nell'equazione di Bellman.
        # NOTA: usiamo Quu (NON regolarizzato) per l'update di P
        # perché la regolarizzazione serve solo per la stabilità numerica
        # della soluzione, non deve distorcere la stima del valore ottimo.
        #
        # P_t = Qxx + K_tᵀ Quu K_t + K_tᵀ Qux + Qux.T K_t
        # p_t = Qx  + K_tᵀ Quu k_t + K_tᵀ Qu  + Qux.T k_t
        P = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
        p = Qx  + K_t.T @ Quu @ k_t + K_t.T @ Qu  + Qux.T @ k_t

    return KK, kk_vec
