#
# Optimal control of an Acrobot
# Newton Closed Loop Method
#

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import dynamics as dyn
import cost as cst
import solver_ltv_lqr as lqr
import armijo as arm
import data as cfg
import control as ctrl

def newton_method(xx, uu, xx_ref, uu_ref, x0, max_iters, Qt, Rt, QT,
                  armijo_plot=True, armijo_plot_number=2, save_path_armijo_base=None):
    """
    Regularized Newton's method for optimal control of an Acrobot.

    Parameters
    ----------
    xx : array, shape (ns, TT, max_iters+1)        Decision variable states.
    uu : array, shape (ni, TT, max_iters+1)        Decision variable inputs.
    xx_ref : array, shape (ns, TT)                 Reference curve states.
    uu_ref : array, shape (ni, TT)                 Reference curve inputs.
    x0 : array, shape (ns,)                        Initial condition.
    max_iters : int                                Maximum number of iterations.
    Qt : array, shape (ns, ns)                     Stage cost weight for states.
    Rt : array, shape (ni, ni)                     Stage cost weight for inputs.
    QT : array, shape (ns, ns)                     Terminal cost weight.
    armijo_plot : bool, optional                   Flag to plot Armijo rule
    armijo_plot_number : int, optional             Number of iterations to plot Armijo rule
    save_path_armijo_base : str, optional          Base path for saving Armijo plots.

    Returns
    -------
    xx : array, shape (ns, TT, max_iters+1)        Decision variable states at each iteration.
    uu : array, shape (ni, TT, max_iters+1)        Decision variable inputs at each iteration.
    descent : array, shape (max_iters+1,)          Descent at each iteration.
    J : array, shape (max_iters+1,)                Cost at each iteration.
    kk : int                                       Number of iterations.
    """
    
    ns = cfg.ns
    ni = cfg.ni
    TT = uu.shape[1]

    # ARMIJO PARAMETERS
    c = cfg.armijo_c
    beta = cfg.armijo_beta
    armijo_maxiters = cfg.armijo_maxiters   
    stepsize_0 = cfg.armijo_stepsize0         
    term_cond = cfg.term_cond


    # Linearization
    A = np.zeros((ns, ns, TT))
    B = np.zeros((ns, ni, TT))

    # Derivatives/Gradients of the cost (affine terms of the cost)
    q = np.zeros((ns, TT))
    r = np.zeros((ni, TT))

    # Initial conditions
    xx0 = np.zeros((ns,))

    # Cost matrices (regularized - so with only cost Hessian)
    Qtilda = np.zeros((ns, ns, TT))
    Rtilda = np.zeros((ni, ni, TT))
    Stilda = np.zeros((ni, ns, TT))

    # lambda for the co-state equation
    lmbd = np.zeros((ns, TT, max_iters+1))    

    # Cost and descent direction 
    dJ = np.zeros((ni, TT, max_iters+1))       
    J = np.zeros(max_iters+1)                 
    descent = np.zeros(max_iters+1)           
    descent_arm = np.zeros(max_iters+1)       

    # Decision variables
    deltax = np.zeros((ns, TT, max_iters+1)) 
    deltau = np.zeros((ni, TT, max_iters+1)) 

    ################################################################################################################

    for kk in range(max_iters):
        J[kk] = 0

        for tt in range(TT-1):
            temp_cost = cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
            J[kk] += temp_cost
            fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:]
            A[:,:,tt] = fx
            B[:,:,tt] = fu
            
            Qtilda[:,:,tt] = Qt
            Rtilda[:,:,tt] = Rt

        term_cost, qT, QTilda = cst.termcost(xx[:,-1,kk], xx_ref[:,-1], QT)
        J[kk] += term_cost

        # Descent direction calculation
        lmbd_temp = qT
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()
        
        for tt in reversed(range(TT-1)):                        
            qt, rt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[1:3]          

            lmbd_temp = A[:,:,tt].T @ lmbd[:,tt+1,kk][:,None] + qt.reshape(-1, 1)       # costate equation
            dJ_temp   = B[:,:,tt].T @ lmbd[:,tt+1,kk][:,None] + rt.reshape(-1, 1)       # gradient of J wrt u 
            
            q[:,tt] = qt.squeeze()
            r[:,tt] = rt.squeeze()
            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()

        # Solve the Affine LQR problem
        deltax[:,:,kk], deltau[:,:,kk], KK, sigma, _ = lqr.ltv_LQR_affine(A, B, Qtilda, Rtilda, Stilda, QTilda, TT, xx0, q, r, qT.squeeze())

        for tt in reversed(range(TT-1)): 
            descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk] 

        descent[kk] = abs(descent_arm[kk])

        # Determine armijo save path if plotting is enabled
        save_path = f"{save_path_armijo_base}_{kk}.png" if save_path_armijo_base and armijo_plot and (kk < 1 or kk%10 == 0 or kk==armijo_plot_number or kk==7) else None

        stepsize = arm.select_stepsize(stepsize_0, armijo_maxiters, c, beta, deltau[:, :, kk], xx_ref, uu_ref, x0, uu[:, :, kk], xx[:, :, kk], KK, sigma, J[kk], descent_arm[kk], kk, Qt, Rt, QT, armijo_plot, armijo_plot_number, save_path=save_path)

        # Update the current solution
        xx_temp = np.zeros((ns,TT))
        uu_temp = np.zeros((ni,TT))

        xx_temp[:,0] = x0

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt] @ (xx_temp[:,tt] - xx[:,tt,kk]) + stepsize * sigma[:,tt]
            xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])

        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Termination condition
        print(f'  Iter = {kk:3d} \t Descent = {descent[kk]:.3e} \t Cost = {J[kk]:.3e}')

        if descent[kk] <= term_cond:
            max_iters = kk
            print(f"Convergence achieved at iteration {kk}. Terminating optimization.")
            break

    return xx[:, :, :kk+2], uu[:, :, :kk+2], descent, descent_arm, J, kk