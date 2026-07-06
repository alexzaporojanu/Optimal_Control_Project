import numpy as np
import sys
sys.path.append('/home/alex/OPCON_Proj/Project')
import data
from dynamics import dynamics, step
from solver_ltv_lqr import backward_riccati
import control as ctrl

# Load Task 2 reference
data_dict = np.load('/home/alex/OPCON_Proj/Project/data/optimal_trajectory_task2.npy', allow_pickle=True).item()
x_ref_traj = data_dict['x'].T
u_ref_traj = data_dict['u'].T
steps = x_ref_traj.shape[0]

# Linearize
A_list, B_list = [], []
for t in range(steps):
    _, A, B = dynamics(x_ref_traj[t], u_ref_traj[t].flatten())
    A_list.append(A)
    B_list.append(B)

Q_lqr = data.Q_track
R_lqr = data.R_track

# Compute tracking DARE at the final equilibrium
A_eq, B_eq = A_list[-1], B_list[-1]
QQT_track = ctrl.dare(A_eq, B_eq, Q_lqr, R_lqr)[0]

# Compute gains
K_gains, P_list = backward_riccati(A_list, B_list, Q_lqr, R_lqr, QQT_track, steps)

# Perturbation: shoulder -0.2 rad
pert = np.array([-0.2, 0.0, 0.0, 0.0])
x_sim = np.zeros((steps + 1, 4))
x_sim[0] = x_ref_traj[0] + pert

print("Time-steps, errors and torques with tracking DARE:")
for t in range(steps):
    delta_x = x_sim[t] - x_ref_traj[t]
    delta_u = -K_gains[t] @ delta_x
    u_curr = u_ref_traj[t] + delta_u
    x_sim[t+1] = step(x_sim[t], u_curr.flatten())
    
    if t % 200 == 0 or t == steps-1:
        print(f"t={t:4d}: delta_x={delta_x.round(4)}, delta_u={delta_u.round(4)}, u_applied={u_curr.round(4)}")
        
final_err = np.linalg.norm(x_sim[-2] - x_ref_traj[-1])
print(f"\nFinal Error with Tracking DARE: {final_err:.4e}")
