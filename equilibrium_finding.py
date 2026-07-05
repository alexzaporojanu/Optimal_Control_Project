#
# Acrobot — Equilibrium Finding via SQP
# Optimal Control Project — Parameter Set 3
#

import numpy as np
import scipy.linalg
from dynamics import dynamics


def _equality_constraint(xx, uu):
    """
    Evaluates the equilibrium constraint h(x,u) = F(x,u) - x and its Jacobian.
    h(x, u) = F(x, u) - x   in R^4
    dh/dz = [dF/dx - I,  dF/du] = [A - I,  B]   in R^(4x5)
    where A = dF/dx, B = dF/du are discrete dynamics Jacobians.
    """
    x_next, A, B = dynamics(xx, uu)

    h    = x_next - xx               # (4,) — constraint violation
    dh_x = A - np.eye(len(xx))       # (4x4)
    dh_u = B                          # (4x1)
    dh   = np.hstack([dh_x, dh_u])  # (4x5) — combined Jacobian

    return h, dh


def _cost_quadratic(xx, uu, Q, R, xref, uref):
    """
    Quadratic regularization cost to keep the solution close to the initial guess.
    c(x, u)  = 1/2 (x-xref)^T Q (x-xref) + 1/2 (u-uref)^T R (u-uref)
    dc/dz    = [Q(x-xref); R(u-uref)]   (combined vector)
    d^2c/dz^2  = block_diag(Q, R)         (Hessian = B_k in SQP)
    """
    dx = xx - xref
    du = uu - uref

    c    = 0.5 * dx.T @ Q @ dx + 0.5 * du.T @ R @ du
    dc   = np.hstack([Q @ dx, R @ du])           # (5,)
    ddc  = scipy.linalg.block_diag(Q, R)         # (5x5)

    return float(c), dc, ddc


def find_equilibrium(x_guess, u_guess, label="", max_iters=50, tol=1e-8):
    """
    Finds an equilibrium (x*, u*) of the Acrobot with SQP.
    Iteratively solves the KKT system:
        [B  dh^T] [delta_z] = [-dc]
        [dh   0 ] [lambda ] = [-h ]
    where B = cost Hessian, dh = Jacobian of constraint.
    """
    print(f"\n--- Equilibrium Search: {label} ---")

    nx, nu = 4, 1
    nz = nx + nu   # 5 decision variables [x; u]

    z = np.hstack([x_guess.flatten(), u_guess.flatten()])   # z0

    # Regularization cost weights to keep it close to guess
    Q_reg = np.diag([10.0, 10.0, 1.0, 1.0])
    R_reg = np.diag([0.1])

    for k in range(max_iters):
        xx_k = z[:nx]
        uu_k = z[nx:]

        # 1. Compute cost, gradient, and Hessian
        _, dc, B_k = _cost_quadratic(xx_k, uu_k, Q_reg, R_reg,
                                      x_guess.flatten(), u_guess.flatten())

        # 2. Compute constraint and its Jacobian
        h_k, dh_k = _equality_constraint(xx_k, uu_k)

        # Convergence check
        constr_norm = np.linalg.norm(h_k)
        if k % 10 == 0:
            print(f"  Iter {k:3d}: ||h|| = {constr_norm:.3e}")
        if constr_norm < tol:
            print(f"  Converged! ||h|| = {constr_norm:.2e} (< {tol:.0e}) "
                  f"at iter {k}.")
            break

        # 3. Build and solve KKT system (SQP step)
        KKT = np.block([
            [B_k,      dh_k.T               ],   # (5x5), (5x4)
            [dh_k,     np.zeros((nx, nx))   ]    # (4x5), (4x4)
        ])                                        # total: (9x9)

        rhs = np.hstack([-dc, -h_k])             # (9,)

        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            print(f"  WARNING [iter {k}]: Singular KKT system. Using pseudo-inverse fallback.")
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

        dz = sol[:nz]   # extract only delta_z (Lagrange multipliers are not needed here)

        # 4. Pure Newton step update
        z = z + dz

    x_eq = z[:nx]
    u_eq = z[nx:]

    # Final verification
    x_check, _, _ = dynamics(x_eq, u_eq)
    err = np.linalg.norm(x_check - x_eq)
    print(f"  Equilibrium: x = {x_eq.round(6)}")
    print(f"              u = {u_eq.round(6)}")
    print(f"  Verification F(x*,u*)-x* = {err:.2e}")

    return x_eq, u_eq


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    # ---- Equilibrium 1: Downward (rest position) ----
    x_down_guess = np.array([0.0, 0.0, 0.0, 0.0])
    u_down_guess = np.array([0.0])
    x_eq1, u_eq1 = find_equilibrium(x_down_guess, u_down_guess, "DOWNWARD (theta1=0)")

    # ---- Equilibrium 2: Upward (unstable position) ----
    x_up_guess = np.array([np.pi, 0.0, 0.0, 0.0])
    u_up_guess = np.array([0.0])
    x_eq2, u_eq2 = find_equilibrium(x_up_guess, u_up_guess, "UPWARD (theta1=pi)")

    # Save data for other tasks
    np.save('data/equilibrium_data.npy', {
        'x_eq1': x_eq1,
        'x_eq2': x_eq2,
        'u_eq1': u_eq1,
        'u_eq2': u_eq2
    })
    print("\nEquilibrium data saved to 'data/equilibrium_data.npy'")
    print(f"x_eq1 (down):   {x_eq1.round(4)}")
    print(f"x_eq2 (up):    {x_eq2.round(4)}")
