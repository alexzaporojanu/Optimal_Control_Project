import numpy as np
data = np.load('/home/alex/OPCON_Proj/Project/data/optimal_trajectory_task1.npy', allow_pickle=True).item()
x = data['x']
print("Final 20 states:")
for t in range(-20, 0):
    print(f"t={1000+t}: theta1={x[0, t]:.4f}, theta2={x[1, t]:.4f}, omega1={x[2, t]:.4f}, omega2={x[3, t]:.4f}")
