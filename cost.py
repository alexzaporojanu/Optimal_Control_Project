#
# Acrobot — Cost Functions
# Optimal Control Project — Parameter Set 3
#

import numpy as np
import data as dt

ns = dt.ns
ni = dt.ni

def stagecost(xx, uu, xx_ref, uu_ref, Q, R):
    """
    Computes the quadratic stage cost and its gradients.
    Dynamically receives matrices Q and R defined in data.py
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
    Computes the quadratic terminal cost and its gradient.
    Dynamically receives matrix QT (often computed from DARE)
    """
    xT     = xT.reshape(ns, 1)
    xT_ref = xT_ref.reshape(ns, 1)

    dx = xT - xT_ref

    llT = 0.5 * dx.T @ QT @ dx
    lTx = (QT @ dx).flatten()

    return llT.squeeze(), lTx, QT