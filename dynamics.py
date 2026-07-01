#
# Optimal control of an Acrobot
# Semi-Analytical Dynamic script
#

import numpy as np
import sympy as sy
import data

# ==========================================
# 1. PARAMETERS SETUP
# ==========================================
m1, m2 = data.m1, data.m2
l1, l2 = data.l1, data.l2
lc1, lc2 = data.lc1, data.lc2
I1, I2 = data.I1, data.I2
f1, f2 = data.f1, data.f2
g_gravity = data.g

dt = data.dt 
ns = data.ns 
ni = data.ni 

# ==========================================
# 2. SYMBOLIC CONTINUOUS DEFINITION
# ==========================================
print(" Compiling Symbolic Continuous Dynamics...")

x_sym = sy.Matrix(sy.symbols(f'x0:{ns}'))   
u_sym = sy.Matrix(sy.symbols(f'u0:{ni}'))   

th1, th2, dth1, dth2 = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
tau = u_sym[0]

s1, s2, c2, s12 = sy.sin(th1), sy.sin(th2), sy.cos(th2), sy.sin(th1 + th2)

# Mass Matrix M(q)
m11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
m12 = I2 + m2 * lc2 * (l1 * c2 + lc2)
m21 = m12
m22 = I2 + m2 * lc2**2
M_mat = sy.Matrix([[m11, m12], [m21, m22]])

# Coriolis matrix C(q, dq)
h_cor = -m2 * l1 * lc2 * s2 
C_mat = sy.Matrix([
    [h_cor * dth2, h_cor * (dth1 + dth2)], 
    [-h_cor * dth1, 0]
])

# Gravity G(q)
G_vec = sy.Matrix([
    [m1 * lc1 * g_gravity * s1 + m2 * g_gravity * (l1 * s1 + lc2 * s12)], 
    [m2 * lc2 * g_gravity * s12]
])

# Friction matrix F(dq)
F_mat = sy.Matrix([[f1, 0], [0, f2]])

# Underactuated Input (Hip only)
Tau_vec = sy.Matrix([[0], [tau]])
dq = sy.Matrix([dth1, dth2])

# ddq = M^-1 * (Tau - C*dq - F*dq - G)
ddq = M_mat.inv() * (Tau_vec - C_mat * dq - F_mat * dq - G_vec)

f_continuous_sym = sy.Matrix([dth1, dth2, ddq[0], ddq[1]])

# We only lambdify the continuous dynamics. No RK4 algebraic explosion
f_continuous_func = sy.lambdify([x_sym, u_sym], f_continuous_sym, 'numpy', cse=True)

def continuous_dynamics(xx, uu):
    """Evaluates the compiled continuous physics."""
    return f_continuous_func(xx.reshape(ns, 1), uu.reshape(ni, 1)).flatten()

# ==========================================
# 3. NUMERICAL RK4 & JACOBIANS
# ==========================================
print(" Dynamics module ready! (Instant compilation achieved)")

def step(xx, uu):
    """Numerical RK4 using the compiled continuous dynamics."""
    xx = np.array(xx, dtype=float).flatten()
    uu = np.array(uu, dtype=float).flatten()
    
    k1 = continuous_dynamics(xx, uu)
    k2 = continuous_dynamics(xx + 0.5 * dt * k1, uu)
    k3 = continuous_dynamics(xx + 0.5 * dt * k2, uu)
    k4 = continuous_dynamics(xx + dt * k3, uu)
    
    return xx + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def dynamics(xx, uu):
    """
    Returns the next state and the Jacobians evaluated via rapid finite differences.
    Because `step` calls a lambdified C-level function, this takes microseconds.
    """
    xxp = step(xx, uu)
    eps = 1e-5

    A_d = np.zeros((ns, ns))
    for i in range(ns):
        x_plus = xx.copy(); x_plus[i] += eps
        x_minus = xx.copy(); x_minus[i] -= eps
        A_d[:, i] = (step(x_plus, uu) - step(x_minus, uu)) / (2 * eps)

    B_d = np.zeros((ns, ni))
    for i in range(ni):
        u_plus = uu.copy(); u_plus[i] += eps
        u_minus = uu.copy(); u_minus[i] -= eps
        B_d[:, i] = (step(xx, u_plus) - step(xx, u_minus)) / (2 * eps)

    return xxp, A_d, B_d