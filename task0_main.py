#
# Optimal control of an Acrobot
# Task 0: Free Fall / Passive Dynamics Sanity Check
#
# Technical Context:
#   Validates the physical correctness of the Euler-Lagrange equations
#   integrated via a 4th-order Runge-Kutta (RK4) scheme. The state vector is
#   defined in configuration space as x = [q^T, dq^T]^T in R^4, where q = [theta1, theta2]^T.
#

import numpy as np
import matplotlib.pyplot as plt
import os
import dynamics as dyn
import data

print("=" * 60)
print("   Task 0: Passive Dynamics Sanity Check (Free Fall)")
print("=" * 60)

# Create output directories if they don't exist
os.makedirs('data', exist_ok=True)

# 1. SIMULATION PARAMETERS
tf = data.tf_task0                 # Time horizon in seconds
TT = int(tf / data.dt)             # Discrete steps based on RK4 step dt
tt_hor = np.linspace(0, tf, TT)    # Time grid

# 2. INITIALIZATION
# State trajectory xx is propagated passively. Input uu is set to zero (passive dynamics).
xx = np.zeros((data.ns, TT))       # State trajectory array
uu = np.zeros((data.ni, TT))       # Control input trajectory (zero torque)

# Set initial state: drop the robot from a horizontal position
# x0 = [theta1, theta2, omega1, omega2]^T
x0 = data.x0_task0
xx[:, 0] = x0

# 3. FORWARD SIMULATION
# Open-loop simulation of passive dynamics using RK4 integration.
print("Simulating forward (0 torque applied)...")
for t in range(TT - 1):
    xx[:, t+1] = dyn.step(xx[:, t], uu[:, t])

print("Simulation complete! Generating plots...")

# 4. PLOTTING
fig, axs = plt.subplots(data.ns, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Task 0 — Acrobot Free Swing (Zero Torque)', fontsize=14)

labels = [
    r'$\theta_1$ (Shoulder) [rad]', 
    r'$\theta_2$ (Elbow) [rad]', 
    r'$\dot{\theta}_1$ [rad/s]', 
    r'$\dot{\theta}_2$ [rad/s]'
]
colors = ['blue', 'cyan', 'green', 'purple']

for i in range(data.ns):
    axs[i].plot(tt_hor, xx[i, :], color=colors[i], lw=2)
    axs[i].set_ylabel(labels[i])
    axs[i].grid(alpha=0.4)

axs[-1].set_xlabel('Time [s]')
plt.tight_layout()

# 5. STANDARDIZED TRAJECTORY SAVING
npy_save_path = 'data/optimal_trajectory_task0.npy'
np.save(npy_save_path, {
    'x': xx, 
    'u': uu, 
    't': tt_hor
})
print(f"\nTask 0 trajectory safely saved to '{npy_save_path}'")

# Block=True ensures the window stays open until manually closed
plt.show(block=True)