#
# Acrobot — Physical Data, Hyperparameters and Costs
# Optimal Control Project — Parameter Set 3
#

import numpy as np

# TIME AND DIMENSIONS
ns = 4     # Number of states [theta1, theta2, omega1, omega2]
ni = 1     # Number of inputs [tau]
dt = 0.01  # Discretization step RK4 [s]

# PHYSICAL PARAMETERS (Set 3)
m1  = 1.5   # Mass link 1 [kg]
m2  = 1.5   # Mass link 2 [kg]
l1  = 2.0   # Length link 1 [m]
l2  = 2.0   # Length link 2 [m]
lc1 = 1.0   # Distance center of mass link 1 [m]
lc2 = 1.0   # Distance center of mass link 2 [m]
I1  = 2.0   # Inertia link 1 [kg*m^2]
I2  = 2.0   # Inertia link 2 [kg*m^2]
g   = 9.81  # Gravity [m/s^2]
f1  = 1.0   # Viscous friction joint 1 [N*m*s]
f2  = 1.0   # Viscous friction joint 2 [N*m*s]

# EQUILIBRIUM DEFINITIONS
# Specify only the desired elbow angle (theta2) and whether the
# configuration is inverted (upright) or hanging (downward).
theta2_eq1   = np.radians(-180.0)# Elbow angle for equilibrium 1 [rad]
inverted_eq1 = False            # False = hanging down

theta2_eq2   = np.radians(180.0) # Elbow angle for equilibrium 2 [rad]
inverted_eq2 = False            # False = hanging down

# SOLVER PARAMETERS (Newton / Armijo)
max_iters_task1 = 100      # Maximum iteration for step reference (faster to converge)
max_iters_task2 = 100      # Maximum iterations for smooth reference (slower to converge)
term_cond       = 1e-4     # Tolerance on the norm of the gradient ||Du||^2

# Armijo Line Search Parameters
armijo_c        = 0.5    # Reduction factor (sufficient decrease)
armijo_beta     = 0.7      # Contraction factor of the step
armijo_maxiters = 20       # Maximum number of bisections per step
armijo_stepsize0 = 1       # Initial Armijo Step Size for k=0
armijo_term_cond = 1e-6     # Terminal Condition to stop the search
armijo_plot_resolution = 51 # Number of steps for Armijo plots

# TASK 1 (Step Reference)
Q_task1  = np.diag([200.0, 20.0, 10.0, 1.0])
R_task1  = np.array([[1e-3]])
QT_task1 = np.diag([1e10, 1e2, 1e4, 1e4]) # not actually used, we use the DARE solution instead

# TASK 2 (Smooth Quintic Reference)
# Winner from sweep (C08): heavy θ₁ in generation keeps the shoulder rigid
# during the quintic move; tracking uses balanced equal weights.
# Result: swing_θ₂ ≈ 10.4° vs 28.8° with baseline — 64% reduction.
Q_task2  = np.diag([20.0, 20.0, 1.0, 1.0])   # HIGH weight on θ₂ tracking!
R_task2  = np.array([[0.001]])                       # Slightly higher control cost
QT_task2 = np.diag([1000.0, 5000.0, 100.0, 100.0]) # not actually used, we use the DARE solution instead


# TASK 3 & 4 (LQR / MPC Tracking)
# Winner from sweep (C08): balanced equal weights on θ₁ and θ₂ give the best
# trade-off — err_T_mean=0.037, swing_θ₂=10.4°, swing_θ₁=8.2°.
Q_track  = np.diag([100.0, 1.0, 100, 10.0])   
R_track  = np.array([[0.001]])
QT_track = np.diag([1000.0, 1000.0, 100.0, 100.0]) # not actually used, we use the DARE solution instead