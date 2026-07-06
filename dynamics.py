#
# Acrobot — Dynamics and Analytical Linearization Module (CasADi)
# Optimal Control Project — Parameter Set 3
#

import casadi as ca
import numpy as np
import data

# =============================================================================
# 1. LOAD PHYSICAL AND TEMPORAL PARAMETERS
# =============================================================================
ns = data.ns  # 4 states (theta1, theta2, dtheta1, dtheta2)
ni = data.ni  # 1 control input (torque tau at elbow)
dt = data.dt  # 0.01s discretization step

m1, m2 = data.m1, data.m2
l1, l2 = data.l1, data.l2
lc1, lc2 = data.lc1, data.lc2
I1, I2 = data.I1, data.I2
f1, f2 = data.f1, data.f2
g_gravity = data.g

# =============================================================================
# 2. SYMBOLIC CONTINUOUS DYNAMICS (CasADi SX)
# =============================================================================
# Symbolic variables (SX represents fast sparse symbolic variables)
x_sym = ca.SX.sym('x', ns)
u_sym = ca.SX.sym('u', ni)

th1, th2, dth1, dth2 = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
tau = u_sym[0]

s1, s2, c2, s12 = ca.sin(th1), ca.sin(th2), ca.cos(th2), ca.sin(th1 + th2)

# Mass Matrix M(q)
m11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
m12 = I2 + m2 * lc2 * (l1 * c2 + lc2)
m21 = m12
m22 = I2 + m2 * lc2**2

M_mat = ca.SX(2, 2)
M_mat[0, 0] = m11
M_mat[0, 1] = m12
M_mat[1, 0] = m21
M_mat[1, 1] = m22

# Coriolis Matrix C(q, dq)
h_cor = -m2 * l1 * lc2 * s2
C_mat = ca.SX(2, 2)
C_mat[0, 0] = h_cor * dth2
C_mat[0, 1] = h_cor * (dth1 + dth2)
C_mat[1, 0] = -h_cor * dth1
C_mat[1, 1] = 0

# Gravity Vector G(q)
G_vec = ca.SX(2, 1)
G_vec[0] = m1 * lc1 * g_gravity * s1 + m2 * g_gravity * (l1 * s1 + lc2 * s12)
G_vec[1] = m2 * lc2 * g_gravity * s12

# Viscous Friction Matrix F(dq)
F_mat = ca.SX(2, 2)
F_mat[0, 0] = f1
F_mat[0, 1] = 0
F_mat[1, 0] = 0
F_mat[1, 1] = f2

# Input and velocity vectors
Tau_vec = ca.SX(2, 1)
Tau_vec[0] = 0
Tau_vec[1] = tau

dq = ca.SX(2, 1)
dq[0] = dth1
dq[1] = dth2

# Compute joint accelerations: ddq = M^-1 * (Tau - C*dq - F*dq - G)
# Using ca.solve(A, b) is numerically more stable than inv(A) @ b
rhs = Tau_vec - ca.mtimes(C_mat, dq) - ca.mtimes(F_mat, dq) - G_vec
ddq = ca.solve(M_mat, rhs)

# Continuous-time state derivative x_dot = f_c(x, u)
f_continuous_sym = ca.vertcat(dth1, dth2, ddq[0], ddq[1])

# Create continuous-time dynamics function
f_continuous_func = ca.Function('f_c', [x_sym, u_sym], [f_continuous_sym])

# =============================================================================
# 3. SYMBOLIC RK4 INTEGRATOR & ALGORITHMIC DIFFERENTIATION (AD)
# =============================================================================
# Define RK4 steps symbolically
k1 = f_continuous_func(x_sym, u_sym)
k2 = f_continuous_func(x_sym + 0.5 * dt * k1, u_sym)
k3 = f_continuous_func(x_sym + 0.5 * dt * k2, u_sym)
k4 = f_continuous_func(x_sym + dt * k3, u_sym)

# Next state symbolic expression x_next
x_next_sym = x_sym + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
step_func  = ca.Function('step_func', [x_sym, u_sym], [x_next_sym])

# Compute exact analytical Jacobians via Algorithmic Differentiation (AD)
A_d_sym = ca.jacobian(x_next_sym, x_sym)  # Discrete-time state Jacobian A_d (4x4)
B_d_sym = ca.jacobian(x_next_sym, u_sym)  # Discrete-time input Jacobian B_d (4x1)

# Combined function evaluating step and exact Jacobians in a single call
dynamics_func = ca.Function('dynamics', [x_sym, u_sym], [x_next_sym, A_d_sym, B_d_sym])

# =============================================================================
# 4. PUBLIC API INTERFACE (NumPy Compatible)
# =============================================================================
def step(xx, uu):
    """Computes next state by evaluating the symbolic RK4 step."""
    return np.array(step_func(xx, uu)).flatten()

def dynamics(xx, uu):
    """
    Returns the next state and the exact discrete Jacobians (A_d, B_d)
    computed via Algorithmic Differentiation.
    """
    xxp, A_d, B_d = dynamics_func(xx, uu)
    
    # Convert to NumPy arrays for compatibility with external scripts
    return np.array(xxp).flatten(), np.array(A_d), np.array(B_d)