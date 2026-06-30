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
#
# Acrobot — Backward Pass Newton / iDDP (Iterative DDP)
# Progetto Optimal Control — Parameter Set 3
#

import numpy as np

def solve_newton_step(AA, BB, lx, lu, lxx, luu, TT):
    """
    Backward Pass iDDP — Computes the optimal feedback gains (K_t) and feedforward steps (k_t)
    for a given trajectory using the iDDP (iterative Differential Dynamic Programming) method.
    """
    ns_loc = AA.shape[0]
    ni_loc = BB.shape[1]

    KK     = np.zeros((ni_loc, ns_loc, TT))   # guadagni feedback
    kk_vec = np.zeros((ni_loc, TT))            # passi feedforward
    
    # Initialize expected descent for Armijo condition
    expected_descent = 0.0

    # Inizializzazione con il Costo Terminale
    P = lxx[:, :, -1].copy()   
    p = lx[:, -1].copy()       

    # Backward Recursion: da t=T-1 a t=0 
    for t in reversed(range(TT - 1)):

        At = AA[:, :, t]   
        Bt = BB[:, :, t]   

        q_x  = lx[:, t]       
        q_u  = lu[:, t]       
        q_xx = lxx[:, :, t]   
        q_uu = luu[:, :, t]   

        # Q-Function Expansion 
        Qx  = q_x + At.T @ p           
        Qu  = q_u + Bt.T @ p           
        Qxx = q_xx + At.T @ P @ At     
        Quu = q_uu + Bt.T @ P @ Bt     
        Qux = Bt.T @ P @ At            

        # Regolarizzazione Levenberg-Marquardt
        Quu_reg = Quu + 1e-2 * np.eye(ni_loc)

        try:
            k_t = -np.linalg.solve(Quu_reg, Qu)    
            K_t = -np.linalg.solve(Quu_reg, Qux)   
        except np.linalg.LinAlgError:
            print(f"  [solver_newton] WARNING: Hessiana singolare a t={t}. Aumento regolarizzazione a 1e-1.")
            Quu_extra = Quu_reg + 1e-1 * np.eye(ni_loc)
            k_t = -np.linalg.solve(Quu_extra, Qu)
            K_t = -np.linalg.solve(Quu_extra, Qux)

        KK[:, :, t]  = K_t
        kk_vec[:, t] = k_t
        
        # [FIX]: Calcolo esatto della derivata direzionale: Qu^T * k_t
        expected_descent += (Qu.T @ k_t).item()

        # Aggiornamento Value Function (Bellman) 
        P = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
        p = Qx  + K_t.T @ Quu @ k_t + K_t.T @ Qu  + Qux.T @ k_t

    # Restituiamo anche expected_descent per la regola di Armijo
    return KK, kk_vec, expected_descent