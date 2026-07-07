#
# Acrobot — Reference Trajectory Generation
# Optimal Control Project — Parameter Set 3
#

import numpy as np


def generate_step(tf, dt, x_start, x_goal, u_start, u_goal):
    """
    Generates the STEP reference for Task 1.
    The reference changes instantaneously from (x_start, u_start) to (x_goal, u_goal) at half time (T/2).
    """
    steps = int(tf / dt)
    ns    = len(x_start)
    ni    = len(u_start)

    xx_ref = np.zeros((ns, steps + 1))
    uu_ref = np.zeros((ni, steps))

    half = steps // 2

    # First half: initial equilibrium
    xx_ref[:, :half] = x_start.reshape(-1, 1)
    uu_ref[:, :half] = u_start.reshape(-1, 1)
    # Second half (including T): target equilibrium
    xx_ref[:, half:] = x_goal.reshape(-1, 1)
    uu_ref[:, half:] = u_goal.reshape(-1, 1)

    return xx_ref, uu_ref


def generate_smooth(tf, dt, x_start, x_goal, u_start, u_goal,
                    t_start_frac=0.2, t_end_frac=0.8):
    """
    Generates the SMOOTH reference (quintic polynomial) for Task 2.
    Uses a Hermite quintic polynomial for the transition:
        s(tau) = 10tau^3 - 15tau^4 + 6tau^5   with normalized tau in [0, 1]
    Properties:
        starts with zero velocity and acceleration
        ends with zero velocity and acceleration
    -> C^2 interpolation (continuous first and second derivatives) — ideal for
    reference trajectories for the optimizer.
    """
    steps = int(tf / dt)
    ns    = len(x_start)
    ni    = len(u_start)

    xx_ref = np.zeros((ns, steps + 1))
    uu_ref = np.zeros((ni, steps))

    t_start = t_start_frac * tf
    t_end   = t_end_frac   * tf
    duration = t_end - t_start

    for k in range(steps + 1):
        t = k * dt
        if t <= t_start:
            xx_ref[:, k] = x_start
            if k < steps:
                uu_ref[:, k] = u_start
        elif t >= t_end:
            xx_ref[:, k] = x_goal
            if k < steps:
                uu_ref[:, k] = u_goal
        else:
            # Hermite quintic polynomial
            tau = (t - t_start) / duration   # tau in [0, 1]
            s   = 10*tau**3 - 15*tau**4 + 6*tau**5
            xx_ref[:, k] = x_start + (x_goal - x_start) * s
            if k < steps:
                uu_ref[:, k] = u_start + (u_goal - u_start) * s

    return xx_ref, uu_ref


def generate_extended(dt, x_start, x_goal, u_start, u_goal,
                      t_pre=5.0, t_move=10.0, t_post=5.0):
    """
    Generates the EXTENDED reference (3 phases) for Task 2.
    Structure:
        [0, t_pre]              : Pre-wait — constant at (x_start, u_start)
        [t_pre, t_pre+t_move]   : Smooth Move — quintic transition
        [t_pre+t_move, tf]      : Post-hold — constant at (x_goal, u_goal)
    This 3-phase approach:
      1. Gives the system time to settle at the initial equilibrium
      2. Executes the smooth transition (quintic polynomial)
      3. Gives the system time to absorb the arrival at the new equilibrium
    """
    N_pre  = int(t_pre  / dt)
    N_move = int(t_move / dt)
    N_post = int(t_post / dt)
    tf     = t_pre + t_move + t_post

    ns = len(x_start)
    ni = len(u_start)

    # Pre phase: constant at x_start / u_start
    xx_pre  = np.tile(x_start.reshape(-1, 1), (1, N_pre))
    uu_pre  = np.tile(u_start.reshape(-1, 1), (1, N_pre))

    # Move phase: smooth quintic
    xx_move, uu_move = generate_smooth(t_move, dt, x_start, x_goal, u_start, u_goal)
    # xx_move has N_move+1 columns — we use the first N_move

    # Post phase: constant at x_goal / u_goal
    xx_post = np.tile(x_goal.reshape(-1, 1), (1, N_post))
    uu_post = np.tile(u_goal.reshape(-1, 1), (1, N_post))

    # Concatenation:
    # xx_move[:, :-1] avoids duplicating the junction point between Move and Post
    xx_ref = np.hstack([xx_pre, xx_move[:, :-1], xx_post])
    uu_ref = np.hstack([uu_pre, uu_move, uu_post])

    TT = uu_ref.shape[1]
    xx_ref = np.hstack([xx_ref, xx_ref[:, -1:]])

    return xx_ref, uu_ref, TT, tf, N_pre, N_move
