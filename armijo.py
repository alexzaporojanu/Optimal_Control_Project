#
# Acrobot — Armijo Line Search (Stepsize Selection)
# Progetto Optimal Control — Parameter Set 3
#
# Riferimento teorico:
#   [Slide 07] Gradient Method — sezione "Step Size Selection / Armijo Rule"
#   [Session4/3_armijo.py]    — implementazione del professore (base di questo file)
#
# REGOLA DI ARMIJO (Sufficient Decrease Condition)
# ================================================
# Data la direzione di discesa Δu e il costo corrente J(u^k), il passo α
# viene scelto come il più grande α = β^i (con β ∈ (0,1)) tale che:
#
#   J(u^k + α Δu^k) ≤ J(u^k) + c · α · ∇J(u^k)ᵀ Δu^k
#
# dove:
#   c ∈ (0,1)        : fattore di riduzione (tipicamente c = 0.5)
#   β ∈ (0,1)        : fattore di contrazione (tipicamente β = 0.7)
#   ∇J ᵀ Δu < 0     : la direzione Δu è una direzione di discesa
#
# Nel caso Newton/iDDP (Closed-Loop Update):
#   u_t^{k+1} = u_t^k + α · Δu_t + K_t (x_t^{k+1} - x_t^k)
#
# Il termine K_t (x - x_nom) è il feedback che stabilizza il rollout
# lungo traiettorie divergenti — critico per sistemi altamente nonlineari.
# [Rif.: Slide 08 — "Closed-Loop Forward Pass"]
#

import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import dynamics as dyn


def select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                    deltau, xx_ref, uu_ref, x0, uu, JJ, descent_arm,
                    QQT, K_fb=None, xx_nom=None, plot=False, save_path=None):
    """
    Seleziona il passo α con la regola di Armijo (backtracking line search).

    Supporta due modalità:
      - OPEN-LOOP  (Gradient method): u_new = u + α Δu
      - CLOSED-LOOP (Newton/iDDP):   u_new = u + α Δu + K(x - x_nom)

    La modalità viene selezionata automaticamente in base alla presenza
    del parametro K_fb (guadagno feedback dal backward pass iDDP).

    [Rif.: Session4/3_armijo.py — interfaccia identica con aggiunta K_fb e save_path]
    [Rif.: Slide 07 — "Armijo Backtracking", Slide 08 — "Closed-Loop Forward Pass"]

    Args:
        stepsize_0      : float         — passo iniziale (tipicamente 1.0 per Newton)
        armijo_maxiters : int           — max iterazioni di backtracking
        cc              : float         — costante Armijo c ∈ (0,1)
        beta            : float         — fattore contrazione β ∈ (0,1)
        deltau          : ndarray (ni,TT) — direzione di discesa Δu
        xx_ref          : ndarray (ns, TT+1) — traiettoria di riferimento
        uu_ref          : ndarray (ni, TT)   — ingresso di riferimento
        x0              : ndarray (ns,)  — condizione iniziale
        uu              : ndarray (ni,TT) — ingresso corrente u^k
        JJ              : float          — costo corrente J(u^k)
        descent_arm     : float          — prodotto ∇J ᵀ Δu (negativo se discesa)
        QQT             : ndarray (ns,ns) — matrice costo terminale
        K_fb            : ndarray (ni,ns,TT) oppure None — guadagni feedback (Newton)
        xx_nom          : ndarray (ns,TT) oppure None   — traiettoria nominale x^k
        plot            : bool  — se True, mostra il plot della ricerca (debug)
        save_path       : str   — percorso dove salvare l'immagine del plot (PNG)

    Returns:
        stepsize : float — passo α selezionato
    """
    TT = uu.shape[1]
    ns = x0.shape[0]
    ni = uu.shape[0]

    # Helper: estrae il riferimento al tempo k
    if isinstance(xx_ref, np.ndarray) and xx_ref.ndim > 1:
        get_xref = lambda k: xx_ref[:, k]
    else:
        get_xref = lambda k: xx_ref

    if isinstance(uu_ref, np.ndarray) and uu_ref.ndim > 1:
        get_uref = lambda k: uu_ref[:, k]
    else:
        get_uref = lambda k: np.atleast_1d(uu_ref)

    stepsizes    = []
    costs_armijo = []
    stepsize     = stepsize_0

    for ii in range(armijo_maxiters):

        # ------------------------------------------------------------------
        # FORWARD ROLLOUT con passo α corrente
        # ------------------------------------------------------------------
        xx_temp = np.zeros((ns, TT + 1))
        uu_temp = np.zeros((ni, TT))
        xx_temp[:, 0] = x0

        for k in range(TT):
            du_ff = stepsize * deltau[:, k]

            if K_fb is not None and xx_nom is not None:
                dx_err = xx_temp[:, k] - xx_nom[:, k]
                du_fb  = K_fb[:, :, k] @ dx_err
                uu_temp[:, k] = uu[:, k] + du_ff + du_fb
            else:
                uu_temp[:, k] = uu[:, k] + du_ff

            xx_temp[:, k+1] = dyn.step(xx_temp[:, k], uu_temp[:, k])

        # ------------------------------------------------------------------
        # CALCOLO COSTO J(u + α Δu)
        # ------------------------------------------------------------------
        JJ_temp = 0.0
        for k in range(TT):
            JJ_temp += cst.stagecost(xx_temp[:, k], uu_temp[:, k],
                                     get_xref(k), get_uref(k))[0]

        JJ_temp += cst.termcost(xx_temp[:, -1], get_xref(TT - 1), QQT)[0]

        stepsizes.append(stepsize)
        costs_armijo.append(JJ_temp)

        # ------------------------------------------------------------------
        # CONDIZIONE DI ARMIJO
        # ------------------------------------------------------------------
        if JJ_temp > JJ + cc * stepsize * descent_arm:
            stepsize = beta * stepsize
        else:
            if plot or save_path is not None:
                _plot_armijo(stepsize_0, stepsizes, costs_armijo,
                             JJ, descent_arm, cc, save_path)
            return stepsize

        if ii == armijo_maxiters - 1:
            print("WARNING [Armijo]: nessun passo trovato con la regola di Armijo! "
                  f"Uso l'ultimo: α = {stepsize:.3e}")
            if plot or save_path is not None:
                _plot_armijo(stepsize_0, stepsizes, costs_armijo,
                             JJ, descent_arm, cc, save_path)

    return stepsize


def _plot_armijo(stepsize_0, stepsizes, costs_armijo, JJ, descent_arm, cc, save_path=None):
    """
    Visualizza e salva la ricerca di Armijo.
    """
    s_axis = np.linspace(0, stepsize_0, 20)

    fig = plt.figure(100)
    plt.clf()
    plt.plot(stepsizes, costs_armijo, 'k*-', markersize=8, label='$J(u^k + \\alpha \\Delta u)$')
    plt.plot(s_axis, [JJ + descent_arm * s for s in s_axis],
             'r--', lw=1.5, label='Approx. lineare')
    plt.plot(s_axis, [JJ + cc * descent_arm * s for s in s_axis],
             'g--', lw=1.5, label=f'Soglia Armijo (c={cc})')
    plt.scatter(stepsizes, costs_armijo, marker='*', zorder=5)
    plt.xlabel('Step size $\\alpha$', fontsize=12)
    plt.ylabel('Costo $J$', fontsize=12)
    plt.title('Armijo Line Search', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"  [Armijo] Grafico salvato in: {save_path}")
        
    plt.draw()
    plt.pause(0.1)
