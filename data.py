#
# Acrobot — Dati Fisici, Hyperparameters e Costi
# Optimal Control Project — Parameter Set 3
#

import numpy as np

# TIME AND DIMENSIONS
ns = 4     # Numero di stati [θ₁, θ₂, θ̇₁, θ̇₂]
ni = 1     # Numero di ingressi [τ]
dt = 0.01  # Passo di discretizzazione RK4 [s]


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
term_cond       = 1e-4     # Tolerance on the norm of the gradient ||Δu||^2

# Parametri Armijo Line Search
armijo_c        = 0.5      # Reduction factor (sufficient decrease)
armijo_beta     = 0.7      # Contraction factor of the step
armijo_maxiters = 20       # Maximum number of bisections per step
armijo_stepsize0 = 1       # Initial Armijo Step Size for k=0
armijo_term_cond = 1e-6     # Terminal Condition to stop the search
armijo_plot_resolution = 51 # Number of steps for Armijo plots

# TASK 1 (Step Reference)
# Discontinous reference. The acrobot should do aggressive manouvres
# Low value to the states in order to build up the inertia
Q_task1  = np.diag([0.20, 0.20, 0.001, 0.001])
R_task1  = np.array([[1e-5]])

# QT_task1 
QT_task1 = np.diag([200.0, 2000000.0, 1.0, 1.0]) 


# TASK 2 (Smooth Quintic Reference)
# More feasible reference we can lift the values of the states.
# If it goes in overflow raise the values of R and reduce Q
Q_task2  = np.diag([2.0, 2.0, 0.1, 0.1])
R_task2  = np.array([[0.01]])

# QT_task2 (Fallback senza DARE)
QT_task2 = np.diag([10000.0, 10000.0, 1.0, 1.0]) 


# TASK 3 & 4 (LQR / MPC Tracking)
# Tracking Online: we want a more rigid control to guarantee control over initial perturbances
# The weights on the states are of a higher order of magnitude compared to Task 1 and 2.
Q_track  = np.diag([1000.0, 1000.0, 100.0, 100.0])
R_track  = np.array([[0.1]])

# Terminal cost (spesso sostituito dalle matrici di Riccati)
QT_track = np.diag([1000.0, 1000.0, 100.0, 100.0])