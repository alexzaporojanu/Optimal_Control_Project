import numpy as np
try:
    data = np.load('/home/alex/OPCON_Proj/Project/data/optimal_trajectory_task1.npy', allow_pickle=True).item()
    x = data['x']
    u = data['u']
    t = data['t']
    print(f"Loaded Task 1 trajectory:")
    print(f"  Time horizon: {t[-1]:.2f} seconds")
    print(f"  Steps: {x.shape[1]}")
    print(f"  Max torque: {np.max(np.abs(u)):.2f} Nm")
    
    # Calculate some statistics about the settling/wiggle phase
    # In the second half of the trajectory, what are the oscillations of theta1 and theta2?
    t_mid_idx = x.shape[1] // 2
    theta1_second_half = x[0, t_mid_idx:]
    theta2_second_half = x[1, t_mid_idx:]
    print("Second half (settling phase) stats:")
    print(f"  Theta 1 range: [{np.min(theta1_second_half):.4f}, {np.max(theta1_second_half):.4f}] (mean: {np.mean(theta1_second_half):.4f})")
    print(f"  Theta 2 range: [{np.min(theta2_second_half):.4f}, {np.max(theta2_second_half):.4f}] (mean: {np.mean(theta2_second_half):.4f})")
except Exception as e:
    print(f"Error: {e}")
