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

# SOLVER PARAMETERS (Newton / iDDP / Armijo)
max_iters_task1 = 100       # Maximum iteration for step reference (faster to converge)
max_iters_task2 = 100      # Maximum iterations for smooth reference (slower to converge)
term_cond       = 1e-4     # Tolerance on the norm of the gradient ||Du||^2

# Armijo Line Search Parameters
armijo_c        = 0.5     # Reduction factor (sufficient decrease)
armijo_beta     = 0.7      # Contraction factor of the step
armijo_maxiters = 20       # Maximum number of bisections per step
armijo_stepsize0 = 1       # Initial Armijo Step Size for k=0
armijo_term_cond = 1e-6     # Terminal Condition to stop the search
armijo_plot_resolution = 51 # Number of steps for Armijo plots

# TASK 1 (Step Reference)
# Discontinuous reference. The acrobot should do aggressive maneuvers.
# Low value on states to build up inertia.
Q_task1  = np.diag([1.0, 1.0, 0.00001, 0.00001])
R_task1  = np.array([[1e-2]])
QT_task1 = np.diag([1e7, 1e7, 1000.0, 1000.0]) 

# TASK 2 (Smooth Quintic Reference)
# More feasible reference, state weights can be higher.
Q_task2  = np.diag([2.0, 0.2, 0.1, 0.1])
R_task2  = np.array([[0.01]])
QT_task2 = np.diag([10000.0, 80000.0, 1.0, 1.0]) # Fallback without DARE

# TASK 3 & 4 (LQR / MPC Tracking)
# Online Tracking: rigid control for initial perturbations.
# State weights are higher than Task 1 & 2.
Q_track  = np.diag([1000.0, 1000.0, 100.0, 100.0])
R_track  = np.array([[0.1]])
QT_track = np.diag([1000.0, 1000.0, 100.0, 100.0]) # Terminal cost (often replaced by Riccati matrices)