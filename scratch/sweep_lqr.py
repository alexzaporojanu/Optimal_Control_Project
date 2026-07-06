import numpy as np
import sys
sys.path.append('/home/alex/OPCON_Proj/Project')
from dynamics import dynamics, step
from solver_ltv_lqr import backward_riccati

# Load Task 2 reference
data_dict = np.load('/home/alex/OPCON_Proj/Project/data/optimal_trajectory_task2.npy', allow_pickle=True).item()
x_ref_traj = data_dict['x'].T
u_ref_traj = data_dict['u'].T
QQT_dare = data_dict.get('QQT', None)
steps = x_ref_traj.shape[0]

# Linearize
A_list, B_list = [], []
for t in range(steps):
    _, A, B = dynamics(x_ref_traj[t], u_ref_traj[t].flatten())
    A_list.append(A)
    B_list.append(B)

# Sweep R_track
R_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
Q_values = [
    np.diag([1000.0, 1000.0, 100.0, 100.0]),
    np.diag([10000.0, 10000.0, 1000.0, 1000.0]),
]

pert = np.array([-0.01, 0.0, 0.0, 0.0])

for Q in Q_values:
    print(f"\nEvaluating Q_track = {np.diag(Q)}")
    for R_val in R_values:
        R = np.array([[R_val]])
        QQf = QQT_dare if QQT_dare is not None else Q
        
        K_gains, _ = backward_riccati(A_list, B_list, Q, R, QQf, steps)
        
        x_sim = np.zeros((steps + 1, 4))
        x_sim[0] = x_ref_traj[0] + pert
        
        # Max torque used
        max_torque_applied = 0.0
        
        for t in range(steps):
            delta_x = x_sim[t] - x_ref_traj[t]
            delta_u = -K_gains[t] @ delta_x
            u_curr = u_ref_traj[t] + delta_u
            max_torque_applied = max(max_torque_applied, abs(u_curr[0]))
            x_sim[t+1] = step(x_sim[t], u_curr.flatten())
            
        final_err = np.linalg.norm(x_sim[-2] - x_ref_traj[-1])
        print(f"  R_track={R_val:e} -> Final Error={final_err:.4e}, Max Torque={max_torque_applied:.2f} Nm")
