#
# Acrobot — Physical Data, Horizons, Cost Weights, and Solver Settings
# Optimal Control Project — Parameter Set 3
#

import numpy as np

# =============================================================================
# GENERAL SYSTEM PARAMETERS
# =============================================================================
ns = 4     # Number of states [theta1, theta2, omega1, omega2]
ni = 1     # Number of inputs [tau]
dt = 0.01  # Discretization step RK4 [s]

# Physical Parameters (Set 3)
m1  = 1.5   # Mass link 1 [kg]
m2  = 1.5   # Mass link 2 [kg]
l1  = 2.0   # Length link 1 [m]
l2  = 2.0   # Length link 2 [m]
lc1 = 1.0   # Center of mass distance link 1 [m]
lc2 = 1.0   # Center of mass distance link 2 [m]
I1  = 2.0   # Inertia link 1 [kg*m^2]
I2  = 2.0   # Inertia link 2 [kg*m^2]
g   = 9.81  # Gravity [m/s^2]
f1  = 1.0   # Viscous friction joint 1 [N*m*s]
f2  = 1.0   # Viscous friction joint 2 [N*m*s]

# Nominal Equilibrium Targets (used to find start and goal states)
# Specify only the desired elbow angle (theta2) and whether the
# configuration is inverted (upright) or hanging (downward).
theta2_eq1   = np.radians(-90.0) # Elbow angle for equilibrium 1 [rad]
inverted_eq1 = False             # False = hanging down

theta2_eq2   = np.radians(45.0)  # Elbow angle for equilibrium 2 [rad]
inverted_eq2 = False             # False = hanging down

# Newton & Armijo Solver Shared Settings (used by trajectory generators)
term_cond       = 1e-4     # Convergence tolerance on norm of gradient ||Du||^2
armijo_c        = 0.5      # Reduction factor (sufficient decrease parameter)
armijo_beta     = 0.7      # Contraction factor of the step size
armijo_maxiters = 20       # Maximum number of backtracking bisections
armijo_stepsize0 = 1       # Initial step size guess at k=0
armijo_term_cond = 1e-6    # Terminal condition to stop line search
armijo_plot_resolution = 51 # Number of grid points for Armijo cost plots


# =============================================================================
# TASK 0 — PASSIVE DYNAMICS SANITY CHECK
# =============================================================================
tf_task0 = 10.0  # Simulation time horizon [s]
x0_task0 = np.array([np.pi/2, 0.0, 0.0, 0.0]) # Initial state (horizontal drop)


# =============================================================================
# TASK 1 — TRAJECTORY OPTIMIZATION (STEP REFERENCE)
# =============================================================================
tf_task1        = 10.0  # Simulation time horizon [s]
max_iters_task1 = 100   # Maximum iterations for step reference convergence
Q_task1  = np.diag([200.0, 20.0, 10.0, 1.0]) # State running cost weights
R_task1  = np.array([[1e-3]])                # Input running cost weight
QT_task1 = np.diag([1e10, 1e2, 1e4, 1e4])    # Terminal cost weights (DARE default fallback)


# =============================================================================
# TASK 2 — TRAJECTORY OPTIMIZATION (SMOOTH QUINTIC REFERENCE)
# =============================================================================
# Temporal partition of the 3-phase reference profile:
t_pre_task2  = 10.0  # Pre-wait: hold at start equilibrium [s]
t_move_task2 = 15.0  # Move: quintic transition to final equilibrium [s]
t_post_task2 = 10.0  # Post-hold: stabilize at final equilibrium [s]

max_iters_task2 = 100   # Maximum iterations for smooth reference convergence
Q_task2  = np.diag([20.0, 20.0, 1.0, 1.0])          # State running cost weights (heavy elbow weight)
R_task2  = np.array([[0.001]])                       # Input running cost weight
QT_task2 = np.diag([1000.0, 5000.0, 100.0, 100.0]) # Terminal cost weights (DARE default fallback)


# =============================================================================
# TASK 3 — TRAJECTORY TRACKING VIA TV-LQR
# =============================================================================
# Closed-loop tracking cost weights
Q_track  = np.diag([1000.0, 10.0, 1000.0, 100.0])   
R_track  = np.array([[0.001]])
QT_track = np.diag([1000.0, 1000.0, 100.0, 100.0]) # Terminal weights (DARE default fallback)

# Initial state perturbations for LQR tracking check
perturbations_task3 = {
    'Pert. shoulder -0.4 rad': np.array([-0.4,  0.0, 0.0, 0.0]),
    'Pert. elbow +0.3 rad'   : np.array([ 0.0,  0.3, 0.0, 0.0]),
    'Pert. vel. shoulder'    : np.array([ 0.0,  0.0, 0.3, 0.0]),
}
show_animation_task3 = True  # Render the tracking animation at the end


# =============================================================================
# TASK 4 — TRAJECTORY TRACKING VIA LTV-MPC
# =============================================================================
T_pred_mpc = 100    # Prediction horizon [steps]
U_MAX_mpc  = 6.0    # Absolute physical torque saturation limit [Nm]

# Initial state perturbations for MPC tracking check
perturbations_task4 = {
    'Pert. shoulder -0.2 rad': np.array([-0.2,  0.0, 0.0, 0.0]),
    #'Pert. elbow +0.3 rad'   : np.array([ 0.0,  0.3, 0.0, 0.0]),
}
show_animation_task4 = True # Render the tracking animation at the end