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
import numpy as np
import sympy as sy
from data import * 

# =============================================================================
# MODELLO DINAMICO SIMBOLICO — Equazioni di Eulero-Lagrange
# =============================================================================
print("Acrobot dynamics: compiling symbolic matrices (one-time cost)...")

th1, th2, dth1, dth2 = sy.symbols('th1 th2 dth1 dth2')
tau_sym = sy.symbols('tau')
state_sym = [th1, th2, dth1, dth2]

M_sym = sy.Matrix([
    [I1 + I2 + lc1**2*m1 + m2*(l1**2 + 2*l1*lc2*sy.cos(th2) + lc2**2),
     I2 + lc2*m2*(l1*sy.cos(th2) + lc2)],
    [I2 + lc2*m2*(l1*sy.cos(th2) + lc2),
     I2 + lc2**2*m2]
])

C_sym = sy.Matrix([
    [-l1*lc2*m2*dth2*sy.sin(th2),  -l1*lc2*m2*(dth1+dth2)*sy.sin(th2)],
    [ l1*lc2*m2*dth1*sy.sin(th2),   0]
])

G_sym = sy.Matrix([
    [g*lc1*m1*sy.sin(th1) + g*m2*(l1*sy.sin(th1) + lc2*sy.sin(th1+th2))],
    [g*m2*lc2*sy.sin(th1+th2)]
])

F_sym = sy.Matrix([[f1, 0], [0, f2]])

get_M = sy.lambdify(state_sym, M_sym, 'numpy')
get_C = sy.lambdify(state_sym, C_sym, 'numpy')
get_G = sy.lambdify(state_sym, G_sym, 'numpy')
get_F = sy.lambdify(state_sym, F_sym, 'numpy')

print("Acrobot dynamics: symbolic compilation complete.")

def continuous_dynamics(xx, uu):
    xx = np.array(xx, dtype=float).flatten()
    uu = np.array(uu, dtype=float).flatten()

    th1_v, th2_v, dth1_v, dth2_v = xx

    M_n = get_M(th1_v, th2_v, dth1_v, dth2_v)
    C_n = get_C(th1_v, th2_v, dth1_v, dth2_v)
    G_n = get_G(th1_v, th2_v, dth1_v, dth2_v)
    F_n = get_F(th1_v, th2_v, dth1_v, dth2_v)

    dth_vec = np.array([[dth1_v], [dth2_v]])
    u_val   = uu[0] if uu.size > 0 else 0.0
    
    # MATRICE B_u (Attuazione solo sul giunto 2!)
    tau_vec = np.array([[0.0], [u_val]])

    rhs = tau_vec - C_n @ dth_vec - G_n - F_n @ dth_vec
    ddth = np.linalg.solve(M_n, rhs).flatten()

    return np.array([dth1_v, dth2_v, ddth[0], ddth[1]])

def _step_only(xx, uu):
    k1 = continuous_dynamics(xx, uu)
    k2 = continuous_dynamics(xx + 0.5*dt*k1, uu)
    k3 = continuous_dynamics(xx + 0.5*dt*k2, uu)
    k4 = continuous_dynamics(xx + dt*k3,     uu)
    return xx + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)

def step(xx, uu):
    return _step_only(xx, uu)

def dynamics(xx, uu):
    xx = np.array(xx, dtype=float).flatten()
    uu = np.array(uu, dtype=float).flatten()
    xxp = _step_only(xx, uu)
    eps = 1e-5

    A = np.zeros((ns, ns))
    for i in range(ns):
        xi = xx.copy(); xi[i] += eps
        xd = xx.copy(); xd[i] -= eps
        A[:, i] = (_step_only(xi, uu) - _step_only(xd, uu)) / (2*eps)

    B = np.zeros((ns, ni))
    for i in range(ni):
        ui = uu.copy(); ui[i] += eps
        ud = uu.copy(); ud[i] -= eps
        B[:, i] = (_step_only(xx, ui) - _step_only(xx, ud)) / (2*eps)

    return xxp, A, B