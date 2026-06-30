#
# Acrobot — Armijo Line Search (Stepsize Selection)
# Optimal Control Project — Parameter Set 3
#
# Theoretical Reference:
#   [Slide 07] Gradient Method — "Step Size Selection / Armijo Rule"
#
# ARMIJO RULE (Sufficient Decrease Condition)
# ================================================
# Given the descent direction Δu and the current cost J(u^k), the step α
# is chosen as the largest α = β^i (with β ∈ (0,1)) such that:
#
#   J(u^k + α Δu^k) ≤ J(u^k) + c · α · ∇J(u^k)ᵀ Δu^k
#
# where:
#   c ∈ (0,1)        : reduction factor (typically c = 0.5)
#   β ∈ (0,1)        : contraction factor (typically β = 0.7)
#   ∇J ᵀ Δu < 0      : the direction Δu is a descent direction
#
# In the Newton/iDDP case (Closed-Loop Update):
#   u_t^{k+1} = u_t^k + α · Δu_t + K_t (x_t^{k+1} - x_t^k)
#
# The K_t (x - x_nom) term is the feedback that stabilizes the rollout
# along divergent trajectories — critical for highly nonlinear systems.
# [Ref: Slide 08 — "Closed-Loop Forward Pass"]
#

import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import dynamics as dyn


def select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                    deltau, xx_ref, uu_ref, x0, uu, JJ, descent_arm,
                    QQT, K_fb=None, xx_nom=None, plot=False, save_path=None):
    """
    Selects the stepsize α using the Armijo rule (backtracking line search).
    """
    TT = uu.shape[1]
    ns = x0.shape[0]
    ni = uu.shape[0]

    # Helper: extract the reference at time k
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

    # Safety check: if descent_arm is positive, it is NOT a descent direction!
    if descent_arm > 0:
        print(f"      [Armijo] WARNING: Positive directional derivative ({descent_arm:.2e})! Forcing small step.")
        return 1e-3

    for ii in range(armijo_maxiters):

        # ------------------------------------------------------------------
        # FORWARD ROLLOUT with current step alpha
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
        # CALCULATE COST J(u + alpha * du)
        # ------------------------------------------------------------------
        JJ_temp = 0.0
        for k in range(TT):
            JJ_temp += cst.stagecost(xx_temp[:, k], uu_temp[:, k],
                                     get_xref(k), get_uref(k))[0]

        JJ_temp += cst.termcost(xx_temp[:, -1], get_xref(TT - 1), QQT)[0]

        stepsizes.append(stepsize)
        costs_armijo.append(JJ_temp)

        # ------------------------------------------------------------------
        # ARMIJO CONDITION
        # ------------------------------------------------------------------
        if JJ_temp > JJ + cc * stepsize * descent_arm:
            stepsize = beta * stepsize
        else:
            if plot or save_path is not None:
                _plot_armijo(stepsize_0, stepsizes, costs_armijo, JJ, descent_arm, cc,
                             deltau, xx_ref, uu_ref, x0, uu, QQT, K_fb, xx_nom, save_path)
            return stepsize

        if ii == armijo_maxiters - 1:
            print(f"      [Armijo] WARNING: Iteration limit reached! Using: α = {stepsize:.3e}")
            if plot or save_path is not None:
                _plot_armijo(stepsize_0, stepsizes, costs_armijo, JJ, descent_arm, cc,
                             deltau, xx_ref, uu_ref, x0, uu, QQT, K_fb, xx_nom, save_path)

    return stepsize


def _plot_armijo(stepsize_0, stepsizes, costs_armijo, JJ, descent_arm, cc,
                 deltau, xx_ref, uu_ref, x0, uu, QQT, K_fb=None, xx_nom=None, save_path=None):
    """
    Visualizes and saves the Armijo search. 
    It densely evaluates the forward rollout over 50 alpha steps to plot 
    the TRUE continuous cost function curve, then overlays the backtracking steps.
    """
    TT = uu.shape[1]
    ns = x0.shape[0]
    ni = uu.shape[0]

    # Generate a dense array of alpha steps for a smooth continuous curve
    steps_plot = np.linspace(0, stepsize_0, 50)
    costs_plot = []

    # Helper: extract references
    if isinstance(xx_ref, np.ndarray) and xx_ref.ndim > 1:
        get_xref = lambda k: xx_ref[:, k]
    else:
        get_xref = lambda k: xx_ref

    if isinstance(uu_ref, np.ndarray) and uu_ref.ndim > 1:
        get_uref = lambda k: uu_ref[:, k]
    else:
        get_uref = lambda k: np.atleast_1d(uu_ref)

    # 1. DENSE EVALUATION: Calculate the exact cost for every point on the smooth curve
    for alpha in steps_plot:
        xx_temp = np.zeros((ns, TT + 1))
        uu_temp = np.zeros((ni, TT))
        xx_temp[:, 0] = x0

        for k in range(TT):
            du_ff = alpha * deltau[:, k]

            if K_fb is not None and xx_nom is not None:
                dx_err = xx_temp[:, k] - xx_nom[:, k]
                du_fb  = K_fb[:, :, k] @ dx_err
                uu_temp[:, k] = uu[:, k] + du_ff + du_fb
            else:
                uu_temp[:, k] = uu[:, k] + du_ff

            xx_temp[:, k+1] = dyn.step(xx_temp[:, k], uu_temp[:, k])

        JJ_temp = 0.0
        for k in range(TT):
            JJ_temp += cst.stagecost(xx_temp[:, k], uu_temp[:, k],
                                     get_xref(k), get_uref(k))[0]

        JJ_temp += cst.termcost(xx_temp[:, -1], get_xref(TT - 1), QQT)[0]
        costs_plot.append(JJ_temp)

    # 2. PLOTTING
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # True cost function (Continuous Curve)
    ax.plot(steps_plot, costs_plot, 'b-', lw=2, label=r'$J(u^k + \alpha \Delta u)$ (Actual Cost)')
    
    # Linear approximation and threshold
    ax.plot(steps_plot, [JJ + descent_arm * s for s in steps_plot],
             'r--', lw=1.5, label='Linear Approx.')
    ax.plot(steps_plot, [JJ + cc * descent_arm * s for s in steps_plot],
             'g--', lw=1.5, label=f'Armijo Threshold (c={cc})')
             
    # Tested points (Backtracking guesses)
    ax.scatter(stepsizes, costs_armijo, color='orange', marker='*', s=150, edgecolor='black', zorder=5, label='Tested Steps')
    
    # Highlight accepted step
    if len(stepsizes) > 0:
        ax.scatter(stepsizes[-1], costs_armijo[-1], color='red', marker='o', s=100, edgecolor='black', zorder=6, label=f'Accepted: $\\alpha$={stepsizes[-1]:.3f}')
    
    ax.set_xlabel(r'Step size $\alpha$', fontsize=12)
    ax.set_ylabel('Cost $J$', fontsize=12)
    ax.set_title('Armijo Line Search (Dense Evaluation)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"      [Armijo] Plot saved in: {save_path}")
        
    # Close immediately to prevent GUI freezing
    plt.close(fig)