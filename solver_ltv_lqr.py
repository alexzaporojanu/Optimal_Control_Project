#
# Acrobot — TV-LQR Solver (Time-Varying Linear Quadratic Regulator)
# Progetto Optimal Control — Parameter Set 3
#
# Riferimento teorico:
#   [Slide 10] Optimal Control Based Tracking — sezione "TV-LQR"
#   [Slide 04] LQ Optimal Control             — sezione "Discrete Riccati Equation"
#   [Session5/4_solver_ltv_LQR.py]            — implementazione del professore (base)
#   [Session5/3_main_dlqr_tracking.py]        — pattern di utilizzo
#
# PROBLEMA TV-LQR PER TRACKING
# ==============================
# Dato un sistema LTV (tempo variante) ottenuto linearizzando l'Acrobot
# lungo la traiettoria di riferimento (x*, u*):
#
#   δx_{t+1} = A_t δx_t + B_t δu_t
#
# con δx_t = x_t - x*_t,  δu_t = u_t - u*_t (dinamica dell'errore)
#
# Si vuole minimizzare il costo di tracking:
#
#   J = Σ_{t=0}^{T-1} [δx_tᵀ Q δx_t + δu_tᵀ R δu_t] + δx_Tᵀ Q_T δx_T
#
# SOLUZIONE: Backward Riccati Equation [Slide 04, Slide 10]
#
#   P_T = Q_T
#   Per t = T-1, ..., 0:
#       S_t   = R + B_tᵀ P_{t+1} B_t              ← denominatore
#       K_t   = S_t⁻¹ (B_tᵀ P_{t+1} A_t)         ← guadagno ottimo
#       P_t   = Q + A_tᵀ P_{t+1} A_t - (A_tᵀ P_{t+1} B_t) K_t   ← Riccati
#
# LEGGE DI CONTROLLO (tracking):
#   u_t = u*_t - K_t (x_t - x*_t) = u*_t + K_t δx_t
#
# NOTA: questa formulazione in deviazione è ESATTA per sistemi lineari
# e APPROSSIMATA per sistemi nonlineari (validità locale attorno a x*).
# [Rif.: Slide 10 — "LQR Tracking via Linearization"]
#

import numpy as np


def backward_riccati(A_list, B_list, QQ, RR, QQf, steps):
    """
    Backward Riccati Equation per TV-LQR.

    Calcola la sequenza di guadagni ottimi K_t e la matrice di Riccati P_t
    risolvendo l'equazione di Riccati discreta all'indietro nel tempo.

    Forma standard dell'equazione di Riccati (FORM 1 — da [Session5/4_solver_ltv_LQR.py]):
        S_t = R + B_tᵀ P_{t+1} B_t
        K_t = S_t⁻¹ B_tᵀ P_{t+1} A_t
        P_t = Q + A_tᵀ P_{t+1} A_t - (A_tᵀ P_{t+1} B_t) K_t

    Equivalentemente (FORM 2 — usata anche in V4):
        P_t = Q + (A_t - B_t K_t)ᵀ P_{t+1} (A_t - B_t K_t) + K_tᵀ R K_t
    Le due forme sono MATEMATICAMENTE EQUIVALENTI ma la FORM 1 è più
    numericamente stabile (evita la sottrazione tra matrici grandi).
    Per questo motivo usiamo FORM 1, in accordo con [Session5/4_solver_ltv_LQR.py].

    [Rif.: Slide 04 — "Discrete-time Riccati Equation"]
    [Rif.: Slide 10 — "TV-LQR Backward Pass"]
    [Rif.: Session5/4_solver_ltv_LQR.py — ltv_LQR(), righe 121-141]

    Args:
        A_list : list di ndarray (ns, ns) — Jacobiani ∂F/∂x sulla traiettoria ref.
        B_list : list di ndarray (ns, ni) — Jacobiani ∂F/∂u sulla traiettoria ref.
        QQ     : ndarray (ns, ns) — matrice di peso stato (stage)
        RR     : ndarray (ni, ni) — matrice di peso ingresso
        QQf    : ndarray (ns, ns) — matrice di peso terminale (idealmente = DARE)
        steps  : int — numero di passi temporali (= len(A_list))

    Returns:
        K_gains : list di ndarray (ni, ns) — guadagni K_t per t=0,...,steps-1
        P_list  : list di ndarray (ns, ns) — Riccati P_t (usata in Task 4 come terminale)
    """
    K_gains = [None] * steps   # K_t per t = 0, ..., steps-1
    P_list  = [None] * steps   # P_t per riferimento e terminal cost MPC

    # Condizione al contorno: P_T = Q_T
    P = QQf.copy()

    # Ricorsione backward: da t = steps-1 a t = 0
    for t in reversed(range(steps)):
        A = A_list[t]   # (ns×ns)
        B = B_list[t]   # (ns×ni)

        # S_t = R + B_tᵀ P_{t+1} B_t   (ni×ni) — "denominatore" di Riccati
        S = RR + B.T @ P @ B

        # K_t = S_t⁻¹ B_tᵀ P_{t+1} A_t — guadagno feedback ottimo
        # Usa np.linalg.solve invece di inv per maggiore stabilità numerica
        K_t = np.linalg.solve(S, B.T @ P @ A)   # (ni×ns)

        K_gains[t] = K_t
        P_list[t]  = P.copy()   # salva P_{t+1} associato a questo step

        # Aggiornamento Riccati: FORM 1 (numericamente stabile)
        # P_t = Q + A_tᵀ P_{t+1} A_t - (A_tᵀ P_{t+1} B_t) K_t
        #     = Q + A_tᵀ P_{t+1} (A_t - B_t K_t)
        P = QQ + A.T @ P @ (A - B @ K_t)

    return K_gains, P_list

#in this case c_t = 0
def ltv_LQR_affine(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):

  """
	Solves the LQR problem, for LTV system with (time-varying) affine cost, using the Riccati equation
	
  Parameters
    - AAin (ns x ni (x TT)) matrix         State dynamics matrix.
    - BBin (ns x ni (x TT)) matrix         Input dynamics matrix.
    - QQin (ns x ns (x TT))                State cost matrix.
    - RRin (ni x ni (x TT))                Input cost matrix.
    - SSin (ni x ns (x TT))                Affine term matrix.
    - QQfin (ns x ns)                      Terminal state cost matrix.
    - TT                                   Time horizon.
    - x0 (ns,)                             Initial condition.
    - qqin (ns x (x TT))                   Steate cost affine terms.
    - rrin (ni x (x TT))                   Input cost affine terms.
    - qqf (ns x (x TT))                    Terminal state cost affine terms.
  Return
    - xxout (ns x TT)                      State trajectory.
    - uuout (ni x TT)                      Input trajectory.
    - KK (ni x ns x TT)                    Optimal feedback gain matrix/sequence.
    - sigma (ni x TT)                      Optimal affine term.
    - PP (ns x ns x TT)                    Riccati matrix.
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
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

  # Check dimensions consistency -- safety
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

  # Check for affine terms

  augmented = False

  if qqin is not None or rrin is not None or qqfin is not None:
    augmented = True

  #Initialization
  KK = np.zeros((ni, ns, TT-1)) #K_t
  sigma = np.zeros((ni, TT-1)) #sigma_t
  PP = np.zeros((ns, ns, TT)) #P_t
  pp = np.zeros((ns, TT)) #p_t

  QQ = QQin
  RR = RRin
  SS = SSin
  QQf = QQfin
  
  qq = qqin
  rr = rrin

  qqf = qqfin

  AA = AAin
  BB = BBin

  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0 #Initialization of the initial state
  
  #Terminal Cost (t=T)
  PP[:,:,-1] = QQf #P_T = Q_T
  pp[:,-1] = qqf #p_T = q_T
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):

    #Matrices of time t
    QQ_t = QQ[:,:,tt]
    RR_t = RR[:,:,tt]
    SS_t = SS[:,:,tt]
    AA_t = AA[:,:,tt]
    BB_t = BB[:,:,tt]

    #Matrices and vectors of time t+1
    PP_p = PP[:,:,tt+1] # P_{t+1}
    pp_p = pp[:,tt+1][:,None]   # p_{t+1}
    
    #Affine terms of time t
    qq_t = qq[:, tt][:,None]
    rr_t = rr[:, tt][:,None]
    
    # 1. Common intermiadate computation
    #M_t = R_t + B_t^T * P_{t+1} * B_t
    MMt = RR_t + BB_t.T @ PP_p @ BB_t
    MMt_inv = np.linalg.inv(MMt) #M_t^(-1)
    mmt = rr_t + BB_t.T @ pp_p #m_t = r_t + B_t^T p_{t+1} because c_t = 0

    # Sigma_t = S_t + B_t^T P_{t+1} A_t
    Sigma_t = SS_t + BB_t.T @ PP_p @ AA_t
    
    # 2. Riccati P_t
    # P_t = Q_t + A_t^T P_{t+1} A_t - Sigma_t^T M_t^{-1} Sigma_t (Check the calculation, but it's correct)
    PPt = QQ_t + AA_t.T @ PP_p @ AA_t - Sigma_t.T @ MMt_inv @ Sigma_t
    
    # 3. Affine vector p_t
    # p_t = q_t + A_t^T p_{t+1} - Sigma_t^T M_t^{-1} m_t
    ppt = qq_t + AA_t.T @ pp_p - Sigma_t.T @ MMt_inv @ mmt

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze()


  # Evaluate KK and sigma (Forward Pass)
  
  for tt in range(TT-1):

    #Re-computing (Could be optimized by saving M_inv and Sigma_t)
    PP_p = PP[:,:,tt+1]
    pp_p = pp[:,tt+1][:,None]
    
    QQ_t = QQ[:,:,tt]
    RR_t = RR[:,:,tt]
    BB_t = BB[:,:,tt]
    SS_t = SS[:,:,tt]
    AA_t = AA[:,:,tt]

    qq_t = qq[:, tt][:,None]
    rr_t = rr[:, tt][:,None]
    
    # Check positive definiteness
    MMt = RR_t + BB_t.T @ PP_p @ BB_t
    MMt_inv = np.linalg.inv(MMt)
    
    Sigma_t = SS_t + BB_t.T @ PP_p @ AA_t
    
    mmt = rr_t + BB_t.T @ pp_p  
   
    # for other purposes we could add a regularization step here...

    # 1. Gain K_t (K_t^* in the slide)
    # K_t = - M_t^{-1} * Sigma_t
    KKt = - MMt_inv @ Sigma_t

    # 2. Bias sigma_t (sigma_t^* in the slide)
    # sigma_t = - M_t^{-1} * m_t
    sigma_t = - MMt_inv @ mmt

    KK[:,:,tt] = KKt  
    sigma[:,tt] = sigma_t.squeeze()

  # Trajectory Calculation (Forward Integration) - Implemented for convenience
  for tt in range(TT - 1):
    # Trajectory

    # u_t = K_t * x_t + sigma_t
    uu[:, tt] = KK[:,:,tt] @ xx[:, tt] + sigma[:, tt]
    # x_{t+1} = A * x_t + B * u_t
    xx_p = AA[:,:,tt] @ xx[:,tt] + BB[:,:,tt] @ uu[:,tt]

    xx[:,tt+1] = xx_p

    xxout = xx
    uuout = uu

  return xxout, uuout, KK, sigma, PP
