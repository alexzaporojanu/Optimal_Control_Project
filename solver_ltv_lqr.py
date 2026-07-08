#
# Acrobot — TV-LQR Solver (Time-Varying Linear Quadratic Regulator)
# Optimal Control Project — Parameter Set 3
#

import numpy as np


def backward_riccati(A_list, B_list, QQ, RR, QQf, steps):
    """
    Backward Riccati Equation for TV-LQR.
    Computes the sequence of optimal gains K_t and Riccati matrices P_t
    by solving the discrete-time Riccati equation backwards in time.
    
    Uses FORM 2 (Joseph Form) for maximum numerical robustness.
    """
    K_gains = [None] * steps   # K_t for t = 0, ..., steps-1
    P_list  = [None] * steps   # P_t for reference and MPC terminal cost

    # Boundary condition: P_T = Q_T
    P = QQf.copy()

    # Backward recursion: from t = steps-1 to t = 0
    for t in reversed(range(steps)):
        A = A_list[t]   # (ns x ns)
        B = B_list[t]   # (ns x ni)

        # S_t = R + B_t^T P_{t+1} B_t   (ni x ni) — Riccati "denominator"
        S = RR + B.T @ P @ B

        # K_t = S_t^-1 B_t^T P_{t+1} A_t — optimal feedback gain
        K_t = np.linalg.solve(S, B.T @ P @ A)   # (ni x ns)

        K_gains[t] = K_t

        # Riccati update: FORM 2 (Joseph Form - Robustified Symmetric Update)
        # P_t = Q + A_cl^T P_{t+1} A_cl + K_t^T R K_t
        A_cl = A - B @ K_t
        P = QQ + A_cl.T @ P @ A_cl + K_t.T @ RR @ K_t
        
        P_list[t]  = P.copy()   # save P_{t+1} associated to this step

    return K_gains, P_list


def ltv_LQR_affine(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin=None, rrin=None, qqfin=None):
    """
    Solves the LQR problem for LTV system with (time-varying) affine cost, using the Riccati equation.
    Uses the Robustified Closed-Loop (Joseph) Form for maximum stability during Newton iterations.
    
    Parameters:
        - AAin (ns x ns (x TT))                State dynamics matrix.
        - BBin (ns x ni (x TT))                Input dynamics matrix.
        - QQin (ns x ns (x TT))                State cost matrix.
        - RRin (ni x ni (x TT))                Input cost matrix.
        - SSin (ni x ns (x TT))                Affine cross term matrix.
        - QQfin (ns x ns)                      Terminal state cost matrix.
        - TT                                   Time horizon.
        - x0 (ns,)                             Initial condition.
        - qqin (ns x TT)                       State cost affine terms.
        - rrin (ni x TT)                       Input cost affine terms.
        - qqfin (ns)                           Terminal state cost affine terms.
    Returns:
        - xxout (ns x TT)                      State trajectory.
        - uuout (ni x TT)                      Input trajectory.
        - KK (ni x ns x TT-1)                  Optimal feedback gain sequence.
        - sigma (ni x TT-1)                    Optimal affine term sequence.
        - PP (ns x ns x TT)                    Riccati matrix sequence.
    """
    try:
        ns, lA = AAin.shape[1:]
    except:
        AAin = AAin[:,:,None]
        ns, lA = AAin.shape[1:]

    try:  
        ni, lB = BBin.shape[1:]
    except:
        BBin = BBin[:,:,None]
        ni, lB = BBin.shape[1:]

    try:
        nQ, lQ = QQin.shape[1:]
    except:
        QQin = QQin[:,:,None]
        nQ, lQ = QQin.shape[1:]

    try:
        nR, lR = RRin.shape[1:]
    except:
        RRin = RRin[:,:,None]
        nR, lR = RRin.shape[1:]

    try:
        nSi, nSs, lS = SSin.shape
    except:
        SSin = SSin[:,:,None]
        nSi, nSs, lS = SSin.shape

    # Check dimensions consistency
    if nQ != ns:
        print("Matrix Q does not match number of states")
        exit()
    if nR != ni:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != ns:
        print("Matrix S does not match number of states")
        exit()
    if nSi != ni:
        print("Matrix S does not match number of inputs")
        exit()

    if lA < TT:
        AAin = AAin.repeat(TT, axis=2)
    if lB < TT:
        BBin = BBin.repeat(TT, axis=2)
    if lQ < TT:
        QQin = QQin.repeat(TT, axis=2)
    if lR < TT:
        RRin = RRin.repeat(TT, axis=2)
    if lS < TT:
        SSin = SSin.repeat(TT, axis=2)

    # Initialization
    KK = np.zeros((ni, ns, TT))  # K_t
    sigma = np.zeros((ni, TT))   # sigma_t
    PP = np.zeros((ns, ns, TT + 1))   # Holds P_0 to P_T
    pp = np.zeros((ns, TT + 1))       # Holds p_0 to p_T

    QQ = QQin
    RR = RRin
    SS = SSin
    QQf = QQfin
    
    qq = qqin
    rr = rrin
    qqf = qqfin

    AA = AAin
    BB = BBin

    xx = np.zeros((ns, TT + 1))
    uu = np.zeros((ni, TT))

    xx[:,0] = x0  # Initialize state trajectory

    # Terminal Cost (t=T)
    PP[:,:,TT] = QQf  
    pp[:,TT] = qqf 
    
    # Solve Riccati equation backwards in time
    for tt in reversed(range(TT)):
        QQ_t = QQ[:,:,tt]
        RR_t = RR[:,:,tt]
        SS_t = SS[:,:,tt]
        AA_t = AA[:,:,tt]
        BB_t = BB[:,:,tt]

        PP_p = PP[:,:,tt+1]         # P_{t+1}
        pp_p = pp[:,tt+1][:,None]   # p_{t+1}
        
        qq_t = qq[:, tt][:,None]
        rr_t = rr[:, tt][:,None]
        
        # 1. Common intermediate computations
        MMt = RR_t + BB_t.T @ PP_p @ BB_t
        MMt_inv = np.linalg.inv(MMt)  # M_t^-1
        mmt = rr_t + BB_t.T @ pp_p    # m_t = r_t + B_t^T p_{t+1}

        Sigma_t = SS_t + BB_t.T @ PP_p @ AA_t
        
        # 2. Compute Gains (Moved to backward pass for efficiency)
        KKt = - MMt_inv @ Sigma_t
        sigma_t = - MMt_inv @ mmt
        
        # 3. Robustified Riccati update P_t (Joseph Form with cross-terms)
        A_cl = AA_t + BB_t @ KKt
        PPt = QQ_t + KKt.T @ RR_t @ KKt + KKt.T @ SS_t + SS_t.T @ KKt + A_cl.T @ PP_p @ A_cl
        
        # 4. Robustified Affine vector update p_t
        ppt = qq_t + A_cl.T @ pp_p + KKt.T @ rr_t

        # Save values for this timestep
        PP[:,:,tt] = PPt
        pp[:,tt] = ppt.squeeze()
        KK[:,:,tt] = KKt  
        sigma[:,tt] = sigma_t.squeeze()

    # Trajectory Calculation (Forward Integration)
    for tt in range(TT):
        uu[:, tt] = KK[:,:,tt] @ xx[:, tt] + sigma[:, tt]
        xx_p = AA[:,:,tt] @ xx[:,tt] + BB[:,:,tt] @ uu[:,tt]
        xx[:,tt+1] = xx_p

    print(f"Max gain: {np.max(np.abs(KK))}")
    print(f"Max affine term: {np.max(np.abs(sigma))}")

    return xx, uu, KK, sigma, PP