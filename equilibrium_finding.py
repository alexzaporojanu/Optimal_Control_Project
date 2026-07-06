#
# Acrobot — Equilibrium Finding via scipy.optimize.root
# Optimal Control Project — Parameter Set 3
#

import numpy as np
from scipy.optimize import root
from dynamics import step
import data


def find_equilibrium(theta2_target, inverted=False, label=""):
    """
    Finds an equilibrium (x*, u*) of the Acrobot for a given elbow angle theta2.

    Uses scipy.optimize.root on the discrete-time RK4 dynamics to find
    the shoulder angle theta1 and holding torque u such that the system
    remains stationary: x_{t+1} = x_t  (zero velocity change).

    Parameters:
        theta2_target (float): Desired elbow angle [rad].
        inverted (bool): If True, selects the inverted (upright) branch.
                         If False, selects the hanging (downward) branch.
        label (str): Label for console output.

    Returns:
        x_eq (np.array): Exact equilibrium state [theta1, theta2, 0, 0].
        u_eq (np.array): Exact holding torque [tau].
    """
    print(f"\n--- Equilibrium Search: {label} ---")
    print(f"  Target: theta2 = {np.degrees(theta2_target):.2f} deg, "
          f"inverted = {inverted}")

    # --- Initial guess ---
    # Simple branch selection: 0 for hanging, pi for inverted
    theta1_guess = np.pi if inverted else 0.0
    u_guess = 0.0

    # --- Residual function ---
    # At equilibrium with zero velocities, after one RK4 step the
    # velocities must remain zero. We solve for (theta1, u) such that
    # x_next[2] = 0  and  x_next[3] = 0.
    def residual(free_vars):
        th1, u = free_vars
        xx = np.array([th1, theta2_target, 0.0, 0.0])
        uu = np.array([u])
        x_next = step(xx, uu)
        return [x_next[2], x_next[3]]

    # --- Solve ---
    sol = root(residual, [theta1_guess, u_guess], method='hybr')

    if not sol.success:
        print(f"  WARNING: Root-finder did not converge: {sol.message}")

    theta1_eq = sol.x[0]
    theta1_eq = np.arctan2(np.sin(theta1_eq), np.cos(theta1_eq)) # if theta1 exceeds [-pi, pi], wrap it back to the principal range
    tau_eq = sol.x[1]

    # --- Assemble result ---
    x_eq = np.array([theta1_eq, theta2_target, 0.0, 0.0])
    u_eq = np.array([tau_eq])

    # --- Verification ---
    x_check = step(x_eq, u_eq)
    residual_norm = np.linalg.norm(x_check - x_eq)

    print(f"  Equilibrium: x = {x_eq.round(6)}")
    print(f"               u = {u_eq.round(6)}")
    print(f"  Verification ||F(x*,u*) - x*|| = {residual_norm:.2e}")

    return x_eq, u_eq


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":

    # ---- Equilibrium 1 ----
    x_eq1, u_eq1 = find_equilibrium(
        data.theta2_eq1, data.inverted_eq1, "Equilibrium 1")

    # ---- Equilibrium 2 ----
    x_eq2, u_eq2 = find_equilibrium(
        data.theta2_eq2, data.inverted_eq2, "Equilibrium 2")

    print(f"\nx_eq1: {x_eq1.round(4)}")
    print(f"u_eq1: {u_eq1.round(4)}")
    print(f"x_eq2: {x_eq2.round(4)}")
    print(f"u_eq2: {u_eq2.round(4)}")
