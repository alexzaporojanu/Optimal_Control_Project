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

    This solver optimizes a trajectory by iteratively computing a descent direction
    via an Affine Linear Time-Varying LQR approximation of the Newton step and
    performing a line search (Armijo rule) to find a stepsize that guarantees a
    sufficient decrease in the cost function.

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
    armijo_plot : bool, optional                   Flag to plot Armijo rule.
    armijo_plot_number : int, optional             Number of iterations to plot Armijo rule.
    save_path_armijo_base : str, optional          Base path for saving Armijo plots.

    Returns
    -------
    xx : array, shape (ns, TT, max_iters+1)        Decision variable states at each iteration.
    uu : array, shape (ni, TT, max_iters+1)        Decision variable inputs at each iteration.
    descent : array, shape (max_iters+1,)          Descent at each iteration.
    descent_arm : array, shape (max_iters+1,)      Directional derivative (inner product) for Armijo.
    J : array, shape (max_iters+1,)                Cost at each iteration.
    kk : int                                       Number of iterations.
    """
    
    # Extract dimensions and horizon length
    ns = cfg.ns
    ni = cfg.ni
    TT = uu.shape[1]

    # Load Armijo line search hyperparameters from config
    c = cfg.armijo_c
    beta = cfg.armijo_beta
    armijo_maxiters = cfg.armijo_maxiters   
    stepsize_0 = cfg.armijo_stepsize0         
    term_cond = cfg.term_cond

    # Pre-allocate arrays for linearizing dynamics along the trajectory:
    # x_{t+1} \approx f(x_t, u_t) + A_t \delta x_t + B_t \delta u_t
    A = np.zeros((ns, ns, TT))
    B = np.zeros((ns, ni, TT))

    # Pre-allocate cost gradient vectors (affine terms for the LQR approximation):
    # q_t = \nabla_{x_t} \ell(x_t, u_t)
    # r_t = \nabla_{u_t} \ell(x_t, u_t)
    q = np.zeros((ns, TT))
    r = np.zeros((ni, TT))

    # Initial state deviation is 0 since the initial state x0 is fixed
    xx0 = np.zeros((ns,))

    # Pre-allocate cost Hessian matrices (used in the quadratic approximation of the cost):
    # Qtilda = \nabla_{xx}^2 \ell, Rtilda = \nabla_{uu}^2 \ell, Stilda = \nabla_{ux}^2 \ell
    # For regularized Newton, we approximate these using the standard weights Qt, Rt.
    Qtilda = np.zeros((ns, ns, TT))
    Rtilda = np.zeros((ni, ni, TT))
    Stilda = np.zeros((ni, ns, TT))

    # Pre-allocate co-state trajectories (Lagrange multipliers for the dynamics constraints)
    lmbd = np.zeros((ns, TT+1, max_iters+1))    

    # Pre-allocate tracking arrays for analysis and line search
    dJ = np.zeros((ni, TT, max_iters+1))       # Gradients of the cost w.r.t input: \nabla_{u_t} J
    J = np.zeros(max_iters+1)                 # Total trajectory cost at each iteration
    descent = np.zeros(max_iters+1)           # Convergence metric (magnitude of directional derivative)
    descent_arm = np.zeros(max_iters+1)       # Directional derivative used for Armijo condition checks

    # Pre-allocate optimal search directions (state/control updates)
    deltax = np.zeros((ns, TT + 1, max_iters+1)) 
    deltau = np.zeros((ni, TT, max_iters+1)) 

    # ============================================================================================
    # MAIN OPTIMIZATION LOOP
    # ============================================================================================
    for kk in range(max_iters):
        J[kk] = 0

        # ----------------------------------------------------------------------------------------
        # 1. FORWARD PASS: EVALUATE COST AND LINEARIZE DYNAMICS
        # ----------------------------------------------------------------------------------------
        for tt in range(TT):
            # Compute stage cost at current time-step
            temp_cost = cst.stagecost(xx[:,tt,kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
            J[kk] += temp_cost
            
            # Linearize the system dynamics about the current trajectory: A_t = \partial f/\partial x, B_t = \partial f/\partial u
            fx, fu = dyn.dynamics(xx[:,tt,kk], uu[:,tt,kk])[1:]
            A[:,:,tt] = fx
            B[:,:,tt] = fu
            
            # Set cost Hessians for the LQR quadratic subproblem
            Qtilda[:,:,tt] = Qt
            Rtilda[:,:,tt] = Rt
            
        # Compute terminal cost, terminal gradient qT, and terminal Hessian QTilda
        term_cost, qT, QTilda = cst.termcost(xx[:,-1,kk], xx_ref[:,-1], QT)
        J[kk] += term_cost

        # ----------------------------------------------------------------------------------------
        # 2. BACKWARD PASS: COMPUTE CO-STATES AND GRADIENT OF THE COST W.R.T. CONTROLS
        # ----------------------------------------------------------------------------------------
        # Initialize co-state at terminal time T with the terminal cost gradient: \lambda_T = q_T
        lmbd_temp = qT
        lmbd[:,TT,kk] = lmbd_temp.squeeze()
        
        for tt in reversed(range(TT)):                        
            # Get stage cost gradients: qt = \nabla_x \ell, rt = \nabla_u \ell
            qt, rt = cst.stagecost(xx[:,tt, kk], uu[:,tt,kk], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[1:3]          

            # Propagate co-states backward in time: \lambda_t = A_t^T \lambda_{t+1} + q_t
            lmbd_temp = A[:,:,tt].T @ lmbd[:,tt+1,kk][:,None] + qt.reshape(-1, 1)
            # Compute input gradient (Hamiltonian derivative): \nabla_{u_t} H = B_t^T \lambda_{t+1} + r_t
            dJ_temp   = B[:,:,tt].T @ lmbd[:,tt+1,kk][:,None] + rt.reshape(-1, 1)
            
            q[:,tt] = qt.squeeze()
            r[:,tt] = rt.squeeze()
            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()

        # ----------------------------------------------------------------------------------------
        # 3. SOLVE AFFINE LTV-LQR FOR THE SEARCH DIRECTION
        # ----------------------------------------------------------------------------------------
        # Solves the quadratic subproblem for the search direction (Newton step):
        # min_{\delta x, \delta u} sum_t [ 1/2 * \delta x_t^T Q_t \delta x_t + 1/2 * \delta u_t^T R_t \delta u_t + q_t^T \delta x_t + r_t^T \delta u_t ]
        # subject to: \delta x_{t+1} = A_t \delta x_t + B_t \delta u_t
        # This returns:
        # - deltax, deltau: state and control search directions
        # - KK: feedback gain matrix for tracking the new trajectory variation
        # - sigma: feedforward control adjustment
        deltax[:,:,kk], deltau[:,:,kk], KK, sigma, _ = lqr.ltv_LQR_affine(A, B, Qtilda, Rtilda, Stilda, QTilda, TT, xx0, q, r, qT.squeeze())

        # ----------------------------------------------------------------------------------------
        # 4. COMPUTE DIRECTIONAL DERIVATIVE (DESCENT DIRECTION VALUE)
        # ----------------------------------------------------------------------------------------
        # Compute the directional derivative: DJ(u)[deltau] = \sum_t (\nabla_{u_t} J)^T \delta u_t
        for tt in reversed(range(TT)): 
            descent_arm[kk] += dJ[:,tt,kk].T @ deltau[:,tt,kk] 

        # We monitor the absolute value of the descent metric for convergence checks
        descent[kk] = abs(descent_arm[kk])

        # Verify it is indeed a descent direction (inner product with gradient must be negative)
        if descent_arm[kk] > 0:
            print(f"WARNING: Descent direction is not a descent direction at iteration {kk} (descent_arm = {descent_arm[kk]:.3e})")

        # Determine armijo save path if plotting is enabled
        save_path = f"{save_path_armijo_base}_{kk}.png" if save_path_armijo_base and armijo_plot and (kk < 1 or kk%10 == 0 or kk==armijo_plot_number or kk==7) else None

        # ----------------------------------------------------------------------------------------
        # 5. LINE SEARCH (ARMIJO RULE)
        # ----------------------------------------------------------------------------------------
        # Backtracking line search to find a stepsize \gamma \in (0, 1] that satisfies:
        # J(x(\gamma), u(\gamma)) \le J(x, u) + c * \gamma * descent_arm
        stepsize = arm.select_stepsize(stepsize_0, armijo_maxiters, c, beta, deltau[:, :, kk], xx_ref, uu_ref, x0, uu[:, :, kk], xx[:, :, kk], KK, sigma, J[kk], descent_arm[kk], kk, Qt, Rt, QT, armijo_plot, armijo_plot_number, save_path=save_path)

        # ----------------------------------------------------------------------------------------
        # 6. FORWARD ROLLOUT (CLOSE-LOOP SYSTEM SIMULATION)
        # ----------------------------------------------------------------------------------------
        # Apply closed-loop policy: u_t^{new} = u_t^{old} + K_t(x_t^{new} - x_t^{old}) + stepsize * \sigma_t
        # simulate system dynamics under this control law to get new state trajectory.
        xx_temp = np.zeros((ns, TT + 1))
        uu_temp = np.zeros((ni, TT))

        xx_temp[:,0] = x0

        for tt in range(TT):
            uu_temp[:,tt] = uu[:,tt,kk] + KK[:,:,tt] @ (xx_temp[:,tt] - xx[:,tt,kk]) + stepsize * sigma[:,tt]
            xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])

        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Print iteration status
        print(f'  Iter = {kk:3d} \t Descent = {descent[kk]:.3e} \t Cost = {J[kk]:.3e}')

        # ----------------------------------------------------------------------------------------
        # 7. CONVERGENCE TERMINATION CHECK
        # ----------------------------------------------------------------------------------------
        if descent[kk] <= term_cond:
            max_iters = kk
            print(f"Convergence achieved at iteration {kk}. Terminating optimization.")
            break

    return xx[:, :, :kk+2], uu[:, :, :kk+2], descent, descent_arm, J, kk