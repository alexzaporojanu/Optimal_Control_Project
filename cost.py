#
# Acrobot — Funzioni di Costo
# Progetto Optimal Control — Parameter Set 3
#
# Riferimento teorico:
#   [Slide 06] Optimal Control Shooting  — sezione "Cost Function"
#   [Slide 07] Gradient Method           — sezione "Stage and Terminal Cost"
#   [Session4/1_cost.py]                 — pattern del professore
#
# FORMULAZIONE DEL COSTO
# ======================
# Il problema di ottimo minimizza il costo totale:
#
#   J(x₀, u) = Σ_{t=0}^{T-1} l(x_t, u_t) + l_T(x_T)
#
# con costo di stadio quadratico:
#   l(x, u)  = ½ (x - x_ref)ᵀ Q  (x - x_ref) + ½ (u - u_ref)ᵀ R  (u - u_ref)
#
# e costo terminale:
#   l_T(x_T) = ½ (x_T - x_ref_T)ᵀ Q_T (x_T - x_ref_T)
#
# SCELTA DEI PESI
# ===============
# La scelta dei pesi Q, R, Q_T riflette le priorità del task:
#
#   Q  (stage): pesi BASSI → il robot può muoversi liberamente per immagazzinare
#               energia durante lo swing-up (non vogliamo penalizzare troppo
#               le traiettorie "intermedie" necessarie per raggiungere l'alto)
#
#   R  (input): peso BASSO → incoraggia l'uso della coppia. Non zero per evitare
#               la singolarità di Q_uu nel backward pass (essenziale per iDDP)
#
#   Q_T (term): pesi MOLTO ALTI → impone il vincolo di raggiungere il target
#               a fine orizzonte. È il "pull" che porta il robot in posizione eretta.
#               Nota: Q_T verrà SOVRASCRITTO dalla soluzione DARE nel main
#               per una stima teoricamente fondata del costo infinito-orizzonte.
#

import numpy as np
import dynamics as dyn

ns = dyn.ns   # 4 — [θ₁, θ₂, θ̇₁, θ̇₂]
ni = dyn.ni   # 1 — [τ]

# =============================================================================
# MATRICI DI PESO — Stage Cost
# =============================================================================
#
# Q = diag(w_pos, w_pos, w_vel, w_vel)
# Pesi posizione (angoli) e velocità separati per flessibilità.
# Valori calibrati empiricamente per lo swing-up dell'Acrobot Set 3.
#
# [Rif.: Session4/1_cost.py — pattern identico QQt, RRt]

w_pos_stage = 2.0     # peso angoli durante il movimento
w_vel_stage = 0.01    # peso velocità durante il movimento (molto basso)
QQt = np.diag([w_pos_stage, w_pos_stage, w_vel_stage, w_vel_stage])

# Costo ingresso — piccolo per incoraggiare la coppia ma non nullo
# (garantisce Q_uu > 0 nel backward pass Newton — criticamente importante)
RRt = 1e-3 * np.eye(ni)

# =============================================================================
# MATRICI DI PESO — Terminal Cost (valore di default)
# =============================================================================
#
# NOTA IMPORTANTE: questo è il valore di DEFAULT usato se la DARE non è
# disponibile. Nel main (task1, task2), QQT viene SOVRASCRITTO con la
# soluzione della DARE per una stima teoricamente ottimale.
# [Rif.: Session4/10_main_gradient_optcon_method.py — righe 83-88]
#
# Q_T DEVE essere molto più grande di Q per imporre il vincolo terminale
# (principio del "big weight at the end" — Heuristic Terminal Cost)

w_pos_term = 20000.0   # peso terminale angoli (10000x del stage)
w_vel_term = 100.0     # peso terminale velocità
QQT = np.diag([w_pos_term, w_pos_term, w_vel_term, w_vel_term])


# =============================================================================
# FUNZIONE: COSTO DI STADIO l(x, u)
# =============================================================================
def stagecost(xx, uu, xx_ref, uu_ref):
    """
    Calcola il costo di stadio quadratico e i suoi gradienti.

    l(x, u) = ½ (x-xref)ᵀ Q (x-xref) + ½ (u-uref)ᵀ R (u-uref)

    Gradienti (usati nel backward pass — [Slide 07, 08]):
        ∂l/∂x = Q (x - x_ref)   = lx
        ∂l/∂u = R (u - u_ref)   = lu

    Per costo quadratico le Hessiane sono costanti:
        ∂²l/∂x² = Q = QQt
        ∂²l/∂u² = R = RRt

    [Rif.: Session4/1_cost.py — funzione stagecost identica]

    Args:
        xx     : ndarray (ns,) — stato corrente
        uu     : ndarray (ni,) — ingresso corrente
        xx_ref : ndarray (ns,) — stato di riferimento
        uu_ref : ndarray (ni,) — ingresso di riferimento

    Returns:
        ll : float         — costo scalare l(x,u)
        lx : ndarray (ns,) — gradiente ∂l/∂x
        lu : ndarray (ni,) — gradiente ∂l/∂u
    """
    # Reshape a vettori colonna per operazioni matriciali
    xx     = xx.reshape(ns, 1)
    uu     = uu.reshape(ni, 1)
    xx_ref = xx_ref.reshape(ns, 1)
    uu_ref = uu_ref.reshape(ni, 1)

    dx = xx - xx_ref   # errore di stato
    du = uu - uu_ref   # errore di ingresso

    ll = 0.5 * dx.T @ QQt @ dx + 0.5 * du.T @ RRt @ du   # scalare

    lx = (QQt @ dx).flatten()   # gradiente rispetto allo stato (ns,)
    lu = (RRt @ du).flatten()   # gradiente rispetto all'ingresso (ni,)

    return ll.squeeze(), lx, lu


# =============================================================================
# FUNZIONE: COSTO TERMINALE l_T(x_T)
# =============================================================================
def termcost(xT, xT_ref, QQT_in=None):
    """
    Calcola il costo terminale quadratico e il suo gradiente.

    l_T(x_T) = ½ (x_T - x_ref_T)ᵀ Q_T (x_T - x_ref_T)

    Gradiente (inizializzazione backward pass):
        ∂l_T/∂x = Q_T (x_T - x_ref_T)   = lTx

    Il parametro QQT_in permette di passare esternamente la matrice terminale
    calcolata dalla DARE (aggiornato nel main). Se None, usa il default QQT.
    [Rif.: Session4/1_cost.py — funzione termcost con argomento QQT opzionale]

    Args:
        xT      : ndarray (ns,)   — stato finale
        xT_ref  : ndarray (ns,)   — stato finale di riferimento
        QQT_in  : ndarray (ns,ns) oppure None — matrice terminale (default: QQT)

    Returns:
        llT : float        — costo terminale scalare
        lTx : ndarray (ns,) — gradiente ∂l_T/∂x
    """
    if QQT_in is None:
        QQT_in = QQT

    xT     = xT.reshape(ns, 1)
    xT_ref = xT_ref.reshape(ns, 1)

    dx = xT - xT_ref

    llT = 0.5 * dx.T @ QQT_in @ dx
    lTx = (QQT_in @ dx).flatten()

    return llT.squeeze(), lTx
