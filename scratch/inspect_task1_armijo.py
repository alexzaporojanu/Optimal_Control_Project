import numpy as np
import sys
sys.path.append('/home/alex/OPCON_Proj/Project')
import data as cfg
import dynamics as dyn
import reference_trajectory as ref_gen
import solver_newton
import cost as cst
import control as ctrl
from equilibrium_finding import find_equilibrium

# Setup
tf = 10.0
TT = int(tf / cfg.dt)
Q_task = cfg.Q_task1
R_task = cfg.R_task1

x_start, u_start = find_equilibrium(cfg.theta2_eq1, cfg.inverted_eq1)
x_goal, u_goal   = find_equilibrium(cfg.theta2_eq2, cfg.inverted_eq2)
xx_ref, uu_ref = ref_gen.generate_step(tf, cfg.dt, x_start, x_goal, u_start, u_goal)

_, A_eq, B_eq = dyn.dynamics(x_goal, u_goal.flatten())
QQT = ctrl.dare(A_eq, B_eq, Q_task, R_task)[0]

# Run custom newton solver loop to print step sizes
xx = np.zeros((cfg.ns, TT, cfg.max_iters_task1 + 1))
uu = np.zeros((cfg.ni, TT, cfg.max_iters_task1 + 1))
xx[:, 0, 0] = x_start
uu[:, :, 0] = uu_ref

# Rollout iteration 0
for t in range(TT - 1):
    xx[:, t+1, 0] = dyn.step(xx[:, t, 0], uu[:, t, 0])

import armijo as arm
ns = cfg.ns
ni = cfg.ni
A = np.zeros((ns, ns, TT))
B = np.zeros((ns, ni, TT))
q = np.zeros((ns, TT))
r = np.zeros((ni, TT))
cc = np.zeros((ns,TT))
xx0 = np.zeros((ns,))
Qtilda = np.zeros((ns, ns, TT))
Rtilda = np.zeros((ni, ni, TT))
Stilda = np.zeros((ni, ns, TT))
lmbd = np.zeros((ns, TT, 100))    
dJ = np.zeros((ni,TT, 100))       
J = np.zeros(100)                 
descent = np.zeros(100)           
descent_arm = np.zeros(100)       
deltax = np.zeros((ns, TT, 100)) 
deltau = np.zeros((ni, TT, 100)) 

import solver_ltv_lqr as lqr

print("\n--- NEWTON SOLVER STEP SIZES ---")
for kk in range(10):
    J[kk] = 0
    for tt in range(TT-1):
        temp_cost = cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q_task, R_task)[0]
        J[kk] += temp_cost
        fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:]
        A[:,:,tt] = fx
        B[:,:,tt] = fu
        Qtilda[:,:,tt] = Q_task
        Rtilda[:,:,tt] = R_task

    term_cost, qT, QTilda = cst.termcost(xx[:,-1,kk], xx_ref[:,-1], QQT)
    J[kk] += term_cost

    lmbd_temp = qT
    lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
    
    for tt in reversed(range(TT-1)):                        
        qt, rt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Q_task, R_task)[1:3]          
        lmbd_temp = A[:,:,tt].T @ lmbd[:,tt+1,kk][:,None] + qt.reshape(-1, 1)       
        dJ_temp   = B[:,:,tt].T @ lmbd[:,tt+1,kk][:,None] + rt.reshape(-1, 1)       
        q[:,tt] = qt.squeeze()
        r[:,tt] = rt.squeeze()
        lmbd[:,tt,kk] = lmbd_temp.squeeze()
        dJ[:,tt,kk] = dJ_temp.squeeze()

    deltax[:,:,kk], deltau[:,:,kk], KK, sigma, _ = lqr.ltv_LQR_affine(A, B, Qtilda, Rtilda, Stilda, QTilda, TT, xx0, q, r, qT.squeeze())

    for tt in reversed(range(TT-1)): 
        descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk] 
    descent[kk] = abs(descent_arm[kk])

    if descent[kk] <= 1e-4:
        print(f"Converged at kk={kk}")
        break

    # Run select_stepsize but print how many backtracking steps were taken
    # Armijo logic: 
    #   trial_u = u + step * deltau
    #   if cost <= J + c * step * descent_arm
    c_armijo = cfg.armijo_c
    beta_armijo = cfg.armijo_beta
    stepsize = 1.0
    backtracks = 0
    
    while stepsize > 1e-6:
        # Rollout trial
        xx_trial = np.zeros((ns, TT))
        uu_trial = np.zeros((ni, TT))
        xx_trial[:, 0] = x_start
        
        for tt in range(TT-1):
            uu_trial[:, tt] = uu[:, tt, kk] + KK[:, :, tt] @ (xx_trial[:, tt] - xx[:, tt, kk]) + stepsize * sigma[:, tt]
            xx_trial[:, tt+1] = dyn.step(xx_trial[:, tt], uu_trial[:, tt])
            
        # Compute cost
        J_trial = 0
        for tt in range(TT-1):
            J_trial += cst.stagecost(xx_trial[:, tt], uu_trial[:, tt], xx_ref[:, tt], uu_ref[:, tt], Q_task, R_task)[0]
        J_trial += cst.termcost(xx_trial[:, -1], xx_ref[:, -1], QQT)[0]
        
        if np.isnan(J_trial) or J_trial > J[kk] + c_armijo * stepsize * descent_arm[kk]:
            stepsize *= beta_armijo
            backtracks += 1
        else:
            break
            
    print(f"Iteration k={kk}: Chosen stepsize={stepsize:.4f}, backtracks={backtracks}, Cost={J[kk]:.3f}, Descent={descent[kk]:.3e}")
    
    # Update state/input
    for tt in range(TT-1):
        uu[:, tt, kk+1] = uu[:, tt, kk] + KK[:, :, tt] @ (xx[:, tt, kk+1] - xx[:, tt, kk]) + stepsize * sigma[:, tt]
        xx[:, tt+1, kk+1] = dyn.step(xx[:, tt, kk+1], uu[:, tt, kk+1])
