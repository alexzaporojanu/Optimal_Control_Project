#
# Optimal control of an Acrobot
# Task 0: Free Fall / Passive Dynamics Sanity Check
#

import numpy as np
import matplotlib.pyplot as plt
import os
import dynamics as dyn
import data
import animation as an

print("=" * 60)
print("   Task 0: Passive Dynamics Sanity Check (Free Fall)")
print("=" * 60)

# Create output directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# 1. SIMULATION PARAMETERS
tf = 30.0                          # Simulate for 10 seconds
TT = int(tf / data.dt)             # Number of time steps
tt_hor = np.linspace(0, tf, TT)    # Time vector

# 2. INITIALIZATION
xx = np.zeros((data.ns, TT))       # State trajectory array
uu = np.zeros((data.ni, TT))       # Input trajectory array (All zeros!)

# Set initial state: Let's drop it from exactly horizontal (90 degrees)
# [theta1, theta2, omega1, omega2]
x0 = np.array([np.pi/2, 0.0, 0.0, 0.0])
xx[:, 0] = x0

# 3. FORWARD SIMULATION
print("Simulating forward (0 torque applied)...")
for t in range(TT - 1):
    # Pass the current state and zero-torque input to the dynamics step
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
plt.savefig('figs/task0_passive_dynamics.png', dpi=300)

# =============================================================================
# 5. STANDARDIZED TRAJECTORY SAVING
# =============================================================================
npy_save_path = 'data/optimal_trajectory_task0.npy'
np.save(npy_save_path, {
    'x': xx, 
    'u': uu, 
    't': tt_hor
})
print(f"\nTask 0 trajectory safely saved to '{npy_save_path}'")

# Block=False so we can proceed to animation
plt.show(block=False)

print("\nGenerating animation...")
an.animate_trajectory(tt_hor, xx, title="Task 0: Passive Dynamics (Free Fall)")