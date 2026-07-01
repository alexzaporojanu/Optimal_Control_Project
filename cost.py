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

# MATRICI DI PESO — Stage Cost
# [Rif.: Session4/1_cost.py — pattern identico QQt, RRt]
# [Rif.: Session4/10_main_gradient_optcon_method.py — righe 83-88]

import numpy as np
import data as dt

ns = dt.ns
ni = dt.ni

def stagecost(xx, uu, xx_ref, uu_ref, Q, R):
    """
    Calcola il costo di stadio quadratico e i suoi gradienti.
    Riceve dinamicamente le matrici Q e R definite in config.py
    """
    xx     = xx.reshape(ns, 1)
    uu     = uu.reshape(ni, 1)
    xx_ref = xx_ref.reshape(ns, 1)
    uu_ref = uu_ref.reshape(ni, 1)

    dx = xx - xx_ref
    du = uu - uu_ref

    ll = 0.5 * dx.T @ Q @ dx + 0.5 * du.T @ R @ du

    lx = (Q @ dx).flatten()
    lu = (R @ du).flatten()

    return ll.squeeze(), lx, lu

def termcost(xT, xT_ref, QT):
    """
    Calcola il costo terminale quadratico e il suo gradiente.
    Riceve dinamicamente la matrice QT (spesso calcolata dalla DARE)
    """
    xT     = xT.reshape(ns, 1)
    xT_ref = xT_ref.reshape(ns, 1)

    dx = xT - xT_ref

    llT = 0.5 * dx.T @ QT @ dx
    lTx = (QT @ dx).flatten()

    return llT.squeeze(), lTx