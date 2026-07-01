import numpy as np
import matplotlib.pyplot as plt
import cost as cst
import dynamics as dyn

def select_stepsize(stepsize_0, armijo_maxiters, cc, beta,
                    deltau, xx_ref, uu_ref, x0, uu, JJ, descent_arm,
                    Q, R, QQT, K_fb=None, xx_nom=None, plot=False, save_path=None):
    """
    Armijo Line Search to find a suitable step size for the forward pass in iLQR / iDDP.
    This function iteratively tests step sizes to ensure sufficient decrease in the cost function.
    Args:
        stepsize_0 (float): Initial step size to test.
        armijo_maxiters (int): Maximum number of iterations for the Armijo search.
        cc (float): Sufficient decrease condition constant.
        beta (float): Step size reduction factor.
        deltau (np.ndarray): Feedforward control updates from the backward pass.
        xx_ref (np.ndarray): Reference state trajectory.
        uu_ref (np.ndarray): Reference control trajectory.
        x0 (np.ndarray): Initial state.
        uu (np.ndarray): Current control trajectory.
        JJ (float): Current cost value.
        descent_arm (float): Directional derivative of the cost along deltau.
        Q (np.ndarray): State cost matrix.
        R (np.ndarray): Control cost matrix.
        QQT (np.ndarray): Terminal state cost matrix.
        K_fb (np.ndarray, optional): Feedback gain matrices from the backward pass.
        xx_nom (np.ndarray, optional): Nominal state trajectory for feedback.
        plot (bool, optional): Whether to plot the Armijo search results.
        save_path (str, optional): Path to save the plot if plotting is enabled.
    Returns:
        float: The selected step size that satisfies the Armijo condition.
    """
    TT = uu.shape[1]
    ns = x0.shape[0]
    ni = uu.shape[0]

    stepsizes    = []
    costs_armijo = []
    stepsize     = stepsize_0

    if descent_arm > 0:
        print(f"      [Armijo] WARNING: Positive directional derivative ({descent_arm:.2e})! Forcing small step.")
        return 1e-3

    for ii in range(armijo_maxiters):
        xx_temp = np.zeros((ns, TT))
        uu_temp = np.zeros((ni, TT))
        xx_temp[:, 0] = x0

        for k in range(TT - 1):
            du_ff = stepsize * deltau[:, k]

            if K_fb is not None and xx_nom is not None:
                dx_err = xx_temp[:, k] - xx_nom[:, k]
                du_fb  = K_fb[:, :, k] @ dx_err
                uu_temp[:, k] = uu[:, k] + du_ff + du_fb
            else:
                uu_temp[:, k] = uu[:, k] + du_ff

            xx_temp[:, k+1] = dyn.step(xx_temp[:, k], uu_temp[:, k])

        # CALCOLO DEL COSTO USANDO LE MATRICI Q E R PASSATE
        JJ_temp = 0.0
        for k in range(TT - 1):
            JJ_temp += cst.stagecost(xx_temp[:, k], uu_temp[:, k],
                                     xx_ref[:, k], uu_ref[:, k], Q, R)[0]

        JJ_temp += cst.termcost(xx_temp[:, TT - 1], xx_ref[:, TT - 1], QQT)[0]

        stepsizes.append(stepsize)
        costs_armijo.append(JJ_temp)

        if JJ_temp > JJ + cc * stepsize * descent_arm:
            stepsize = beta * stepsize
        else:
            if plot or save_path is not None:
                _plot_armijo(stepsize_0, stepsizes, costs_armijo, JJ, descent_arm, cc,
                             deltau, xx_ref, uu_ref, x0, uu, Q, R, QQT, K_fb, xx_nom, save_path)
            return stepsize

        if ii == armijo_maxiters - 1:
            print(f"      [Armijo] WARNING: Iteration limit reached! Using: α = {stepsize:.3e}")
            if plot or save_path is not None:
                print(f"Saving Armijo plot to: {save_path}")
                _plot_armijo(stepsize_0, stepsizes, costs_armijo, JJ, descent_arm, cc,
                             deltau, xx_ref, uu_ref, x0, uu, Q, R, QQT, K_fb, xx_nom, save_path)

    return stepsize

def _plot_armijo(stepsize_0, stepsizes, costs_armijo, JJ, descent_arm, cc,
                 deltau, xx_ref, uu_ref, x0, uu, Q, R, QQT, K_fb=None, xx_nom=None, save_path=None):
    
    TT = uu.shape[1]
    ns = x0.shape[0]
    ni = uu.shape[0]

    steps_plot = np.linspace(0, stepsize_0, 50)
    costs_plot = []

    for alpha in steps_plot:
        xx_temp = np.zeros((ns, TT))
        uu_temp = np.zeros((ni, TT))
        xx_temp[:, 0] = x0

        for k in range(TT - 1):
            du_ff = alpha * deltau[:, k]

            if K_fb is not None and xx_nom is not None:
                dx_err = xx_temp[:, k] - xx_nom[:, k]
                du_fb  = K_fb[:, :, k] @ dx_err
                uu_temp[:, k] = uu[:, k] + du_ff + du_fb
            else:
                uu_temp[:, k] = uu[:, k] + du_ff

            xx_temp[:, k+1] = dyn.step(xx_temp[:, k], uu_temp[:, k])

        JJ_temp = 0.0
        for k in range(TT - 1):
            JJ_temp += cst.stagecost(xx_temp[:, k], uu_temp[:, k],
                                     xx_ref[:, k], uu_ref[:, k], Q, R)[0]

        JJ_temp += cst.termcost(xx_temp[:, TT - 1], xx_ref[:, TT - 1], QQT)[0]
        costs_plot.append(JJ_temp)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(steps_plot, costs_plot, 'b-', lw=2, label=r'$J(u^k + \alpha \Delta u)$')
    ax.plot(steps_plot, [JJ + descent_arm * s for s in steps_plot], 'r--', lw=1.5, label='Linear Approx.')
    ax.plot(steps_plot, [JJ + cc * descent_arm * s for s in steps_plot], 'g--', lw=1.5, label=f'Threshold (c={cc})')
    ax.scatter(stepsizes, costs_armijo, color='orange', marker='*', s=150, edgecolor='black', zorder=5, label='Tested Steps')
    if len(stepsizes) > 0:
        ax.scatter(stepsizes[-1], costs_armijo[-1], color='red', marker='o', s=100, edgecolor='black', zorder=6, label=f'Accepted: $\\alpha$={stepsizes[-1]:.3f}')
    
    ax.set_xlabel(r'Step size $\alpha$', fontsize=12)
    ax.set_ylabel('Cost $J$', fontsize=12)
    ax.set_title('Armijo Line Search', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.4)
    fig.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300)
    plt.close(fig)