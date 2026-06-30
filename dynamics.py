#
# Acrobot — Modello Dinamico Continuo e Discreto
# Progetto Optimal Control — Parameter Set 3
#
# Autori: [inserire nomi]
# Corso:  Optimal Control & Reinforcement Learning — UniBO 2025/26
#
# Riferimento teorico:
#   [Slide 05] Continuous-Time Optimal Control  — Sezione "Mechanical Systems"
#   [Slide 07] Gradient Method — sezione "System Linearization"
#   [Slide 08] Second-Order Methods (iDDP)      — sezione "Jacobians"
#   [Session4/8_dynamics_symb.py]               — pattern simbolico del professore
#
# DESCRIZIONE DEL SISTEMA
# =======================
# L'Acrobot è un doppio pendolo planare con:
#   - Link 1 (spalla): passivo, nessuna coppia diretta
#   - Link 2 (gomito): attuato con coppia τ
#
# Stato:  x = [θ₁, θ₂, θ̇₁, θ̇₂] ∈ ℝ⁴
#   θ₁  : angolo assoluto link 1, misurato dal basso (θ₁=0 → link giù)
#   θ₂  : angolo RELATIVO link 2 rispetto a link 1
#   θ̇₁, θ̇₂ : velocità angolari
#
# Ingresso: u = [τ] ∈ ℝ¹ (coppia al solo giunto 2 — Acrobot underactuated)
#

import numpy as np
import sympy as sy

# =============================================================================
# 0. DIMENSIONI GLOBALI DEL SISTEMA
# =============================================================================
ns = 4   # numero di stati  [θ₁, θ₂, θ̇₁, θ̇₂]
ni = 1   # numero di ingressi [τ]

# =============================================================================
# 1. PARAMETRI FISICI — Set 3 dell'Assignment
# =============================================================================
#
# Ricavati dal documento "Acrobot_Assignment.pdf" — Parameter Set 3
#
# m1, m2  [kg]    : masse dei link
# l1, l2  [m]     : lunghezze totali dei link
# lc1,lc2 [m]     : distanza centro di massa dal giunto prossimale (link uniforme → lc = l/2)
# I1, I2  [kg·m²] : momenti di inerzia rispetto al centro di massa (baricentro)
# g       [m/s²]  : accelerazione di gravità
# f1, f2  [N·m·s] : coefficienti di smorzamento viscoso ai giunti
# dt      [s]     : passo di discretizzazione RK4 — identico in tutto il progetto

m1,  m2  = 1.5, 1.5
l1,  l2  = 2.0, 2.0
lc1, lc2 = 1.0, 1.0
I1,  I2  = 2.0, 2.0
g        = 9.81
f1,  f2  = 1.0, 1.0
dt       = 0.01   # [s]

# =============================================================================
# 2. MODELLO DINAMICO SIMBOLICO — Equazioni di Eulero-Lagrange
# =============================================================================
#
# L'equazione del moto in forma Lagrangiana è (vedi [Slide 05]):
#
#   M(q) q̈ + C(q, q̇) q̇ + G(q) + F q̇ = B_u τ
#
# dove:
#   q = [θ₁, θ₂]ᵀ         — vettore di configurazione
#   M(q)  ∈ ℝ²ˣ²          — matrice di massa (simmetrica, def. positiva)
#   C(q,q̇) ∈ ℝ²ˣ²         — matrice di Coriolis/centrifuga (Simboli di Christoffel)
#   G(q)   ∈ ℝ²            — vettore gravitazionale (= ∂V/∂q)
#   F = diag(f₁,f₂) ∈ ℝ²ˣ² — attrito viscoso
#   B_u = [0; 1] ∈ ℝ²      — matrice di ingresso (solo giunto 2 attuato)
#
# Forma esplicita per la simulazione (isolata q̈):
#   q̈ = M(q)⁻¹ [B_u τ - C(q,q̇) q̇ - G(q) - F q̇]
#
# Stato completo: ẋ = f(x,u) = [q̇; q̈]

print("Acrobot dynamics: compiling symbolic matrices (one-time cost)...")

# Simboli SymPy (pattern adottato da [Session4/8_dynamics_symb.py])
th1, th2, dth1, dth2 = sy.symbols('th1 th2 dth1 dth2')
tau_sym = sy.symbols('tau')
state_sym = [th1, th2, dth1, dth2]

# ---- Matrice di Massa M(q) ----
# M₁₁ = I₁ + I₂ + m₁ lc₁² + m₂(l₁² + 2l₁lc₂cos(θ₂) + lc₂²)
# M₁₂ = M₂₁ = I₂ + m₂ lc₂ (l₁ cos(θ₂) + lc₂)
# M₂₂ = I₂ + m₂ lc₂²
M_sym = sy.Matrix([
    [I1 + I2 + lc1**2*m1 + m2*(l1**2 + 2*l1*lc2*sy.cos(th2) + lc2**2),
     I2 + lc2*m2*(l1*sy.cos(th2) + lc2)],
    [I2 + lc2*m2*(l1*sy.cos(th2) + lc2),
     I2 + lc2**2*m2]
])

# ---- Matrice di Coriolis/Centrifuga C(q,q̇) ----
# Calcolata dalla formula dei Simboli di Christoffel: C_ij = Σ_k Γᵢⱼₖ q̇ₖ
# dove Γᵢⱼₖ = ½(∂Mᵢⱼ/∂qₖ + ∂Mᵢₖ/∂qⱼ - ∂Mⱼₖ/∂qᵢ)
# Il termine h = m₂ l₁ lc₂ sin(θ₂) appare ripetutamente
C_sym = sy.Matrix([
    [-l1*lc2*m2*dth2*sy.sin(th2),  -l1*lc2*m2*(dth1+dth2)*sy.sin(th2)],
    [ l1*lc2*m2*dth1*sy.sin(th2),   0]
])

# ---- Vettore Gravitazionale G(q) = ∂V/∂q ----
# V(q) = m₁g lc₁(1-cosθ₁) + m₂g[l₁(1-cosθ₁) + lc₂(1-cos(θ₁+θ₂))]
# G₁ = ∂V/∂θ₁ = g lc₁ m₁ sinθ₁ + g m₂(l₁ sinθ₁ + lc₂ sin(θ₁+θ₂))
# G₂ = ∂V/∂θ₂ = g m₂ lc₂ sin(θ₁+θ₂)
G_sym = sy.Matrix([
    [g*lc1*m1*sy.sin(th1) + g*m2*(l1*sy.sin(th1) + lc2*sy.sin(th1+th2))],
    [g*m2*lc2*sy.sin(th1+th2)]
])

# ---- Matrice di Attrito Viscoso F ----
F_sym = sy.Matrix([[f1, 0], [0, f2]])

# Compilazione in funzioni NumPy efficienti (lambdify — vedi [Session4/8_dynamics_symb.py])
# Eseguita UNA sola volta all'import del modulo
get_M = sy.lambdify(state_sym, M_sym, 'numpy')
get_C = sy.lambdify(state_sym, C_sym, 'numpy')
get_G = sy.lambdify(state_sym, G_sym, 'numpy')
get_F = sy.lambdify(state_sym, F_sym, 'numpy')

print("Acrobot dynamics: symbolic compilation complete.")


# =============================================================================
# 3. DINAMICA CONTINUA NUMERICA: ẋ = f(x, u)
# =============================================================================
def continuous_dynamics(xx, uu):
    """
    Calcola la derivata dello stato ẋ = f(x, u).

    Implementa l'equazione di Eulero-Lagrange:
        M(q) q̈ = B_u τ - C(q,q̇) q̇ - G(q) - F q̇

    dove B_u = [0; 1] (coppia solo al giunto 2 — Acrobot).
    Il sistema lineare M q̈ = rhs è risolto con np.linalg.solve
    (fattorizzazione LU — più stabile e veloce di inv(M)).

    [Rif.: Slide 05 — "Dynamics of Mechanical Systems"]

    Args:
        xx : array-like (4,) — stato [θ₁, θ₂, θ̇₁, θ̇₂]
        uu : array-like (1,) — ingresso [τ]

    Returns:
        x_dot : ndarray (4,) — [θ̇₁, θ̇₂, θ̈₁, θ̈₂]
    """
    xx = np.array(xx, dtype=float).flatten()
    uu = np.array(uu, dtype=float).flatten()

    if xx.size != 4:
        raise ValueError(
            f"[dynamics] Attesi 4 stati, ricevuti {xx.size}. "
            "Assicurarsi di passare un singolo step, non l'intera traiettoria."
        )

    th1_v, th2_v, dth1_v, dth2_v = xx

    # Valutazione numerica delle matrici dinamiche
    M_n = get_M(th1_v, th2_v, dth1_v, dth2_v)  # (2×2)
    C_n = get_C(th1_v, th2_v, dth1_v, dth2_v)  # (2×2)
    G_n = get_G(th1_v, th2_v, dth1_v, dth2_v)  # (2×1)
    F_n = get_F(th1_v, th2_v, dth1_v, dth2_v)  # (2×2)

    dth_vec = np.array([[dth1_v], [dth2_v]])      # q̇ ∈ ℝ²
    u_val   = uu[0] if uu.size > 0 else 0.0
    tau_vec = np.array([[0.0], [u_val]])           # B_u τ (giunto 1 passivo)

    # rhs = B_u τ - C q̇ - G - F q̇
    rhs = tau_vec - C_n @ dth_vec - G_n - F_n @ dth_vec

    # Risoluzione: M q̈ = rhs  →  q̈ = M⁻¹ rhs
    ddth = np.linalg.solve(M_n, rhs).flatten()

    return np.array([dth1_v, dth2_v, ddth[0], ddth[1]])


# =============================================================================
# 4. MAPPA DISCRETA RK4: x_{t+1} = F(x_t, u_t)
# =============================================================================
def step(xx, uu):
    """
    Singolo passo RK4 senza Jacobiani — versione pubblica veloce.

    Usare questa funzione invece di dynamics() quando NON servono
    i Jacobiani (es. rollout in armijo.py, simulazione forward pass).
    È ~11x più veloce di dynamics() perché evita le 10 chiamate FD.
    """
    return _step_only(xx, uu)


def _step_only(xx, uu):
    """
    Singolo passo RK4 — funzione helper PRIVATA.

    Separata da dynamics() per prevenire ricorsione infinita
    durante il calcolo dei Jacobiani per differenze finite:
    dynamics() chiama _step_only(), non se stessa.

    Integrazione Runge-Kutta del 4° ordine (errore locale O(dt⁵)):
        k₁ = f(x, u)
        k₂ = f(x + dt/2·k₁, u)
        k₃ = f(x + dt/2·k₂, u)
        k₄ = f(x + dt·k₃, u)
        x⁺ = x + dt/6·(k₁ + 2k₂ + 2k₃ + k₄)

    [Rif.: Slide 05 — "Discretization of Continuous-Time Systems"]
    """
    k1 = continuous_dynamics(xx, uu)
    k2 = continuous_dynamics(xx + 0.5*dt*k1, uu)
    k3 = continuous_dynamics(xx + 0.5*dt*k2, uu)
    k4 = continuous_dynamics(xx + dt*k3,     uu)
    return xx + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def dynamics(xx, uu):
    """
    Mappa discreta x_{t+1} = F(x_t, u_t) con Jacobiani linearizzati.

    Restituisce lo stato successivo e le matrici di linearizzazione:
        A_t = ∂F/∂x |_{x_t, u_t}   ∈ ℝ⁴ˣ⁴
        B_t = ∂F/∂u |_{x_t, u_t}   ∈ ℝ⁴ˣ¹

    I Jacobiani sono calcolati con differenze finite CENTRALI (errore O(ε²)):
        ∂F/∂x_i ≈ [F(x+ε eᵢ, u) - F(x-ε eᵢ, u)] / (2ε)
    più accurate delle differenze unilaterali O(ε).
    [Rif.: Slide 07 — "Numerical Computation of Jacobians"]

    Questi Jacobiani sono usati in:
        - solver_newton.py       → Q-function expansion (iDDP backward pass)
        - solver_ltv_lqr.py     → TV-LQR design (Task 3)
        - task4_main.py         → LTV model per MPC (Task 4)

    Args:
        xx : array (4,) — stato corrente
        uu : array (1,) — ingresso corrente

    Returns:
        xxp : ndarray (4,)   — stato successivo
        A   : ndarray (4,4)  — Jacobiano discreto ∂F/∂x
        B   : ndarray (4,1)  — Jacobiano discreto ∂F/∂u
    """
    xx = np.array(xx, dtype=float).flatten()
    uu = np.array(uu, dtype=float).flatten()

    xxp = _step_only(xx, uu)

    eps = 1e-5   # passo di perturbazione: bilanciamento errore troncamento/arrotondamento

    # Jacobiano A = ∂F/∂x (differenze finite centrali per ogni componente di stato)
    A = np.zeros((ns, ns))
    for i in range(ns):
        xi = xx.copy(); xi[i] += eps
        xd = xx.copy(); xd[i] -= eps
        A[:, i] = (_step_only(xi, uu) - _step_only(xd, uu)) / (2*eps)

    # Jacobiano B = ∂F/∂u (differenze finite centrali per ogni componente di ingresso)
    B = np.zeros((ns, ni))
    for i in range(ni):
        ui = uu.copy(); ui[i] += eps
        ud = uu.copy(); ud[i] -= eps
        B[:, i] = (_step_only(xx, ui) - _step_only(xx, ud)) / (2*eps)

    return xxp, A, B
