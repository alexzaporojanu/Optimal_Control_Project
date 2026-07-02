#
# Optimal control of an Acrobot
# Animation Script
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import data

# Ensure plots don't block Ctrl+C in the terminal
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

def animate_trajectory(tt, xx, xx_ref=None, title="Acrobot Optimal Trajectory"):
    """
    Animates the trajectory of the Acrobot.
    
    Args:
        tt: Time array (T,)
        xx: State array (4, T)
        xx_ref: Optional reference state array (4, T) to plot as a ghost trace
        title: Title of the animation figure
    """
    th1 = xx[0, :]
    th2 = xx[1, :]
    
    if xx_ref is not None:
        th1_ref = xx_ref[0, :]
        th2_ref = xx_ref[1, :]

    l1 = data.l1
    l2 = data.l2

    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + l2 * np.sin(th1 + th2)
    y2 = y1 - l2 * np.cos(th1 + th2)

    if xx_ref is not None:
        x1_ref = l1 * np.sin(th1_ref)
        y1_ref = -l1 * np.cos(th1_ref)
        x2_ref = x1_ref + l2 * np.sin(th1_ref + th2_ref)
        y2_ref = y1_ref - l2 * np.cos(th1_ref + th2_ref)

    fig, (ax_robot, ax_states) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    bound = l1 + l2 + 0.5
    ax_robot.set_xlim(-bound, bound)
    ax_robot.set_ylim(-bound, bound)
    ax_robot.set_aspect('equal')
    ax_robot.grid(True, alpha=0.3, linestyle='--')
    ax_robot.set_title("2D Physics Simulation", fontsize=14)
    ax_robot.set_xlabel("X [m]")
    ax_robot.set_ylabel("Y [m]")

    if xx_ref is not None:
        line_robot_ref, = ax_robot.plot([], [], 'o-', lw=6, markersize=10, color='gray', alpha=0.3)
    
    line_robot, = ax_robot.plot([], [], 'o-', lw=6, markersize=10, color='#2c3e50')
    line_trace, = ax_robot.plot([], [], '-', lw=1.5, alpha=0.5, color='#e74c3c')
    
    ax_robot.plot([0], [0], 'ks', markersize=12)

    ax_states.plot(tt, th1, label=r'$\theta_1$ (Shoulder)', color='blue', lw=2)
    ax_states.plot(tt, th2, label=r'$\theta_2$ (Elbow)', color='cyan', lw=2)
    
    if xx_ref is not None:
        ax_states.plot(tt, th1_ref, 'k--', alpha=0.5, label=r'$\theta_1^*$ (Ref)')
        ax_states.plot(tt, th2_ref, 'k:', alpha=0.5, label=r'$\theta_2^*$ (Ref)')

    ax_states.set_xlim(tt[0], tt[-1])
    ax_states.set_ylim(min(np.min(th1), np.min(th2)) - 0.5, max(np.max(th1), np.max(th2)) + 0.5)
    ax_states.set_xlabel("Time [s]", fontsize=12)
    ax_states.set_ylabel("Angle [rad]", fontsize=12)
    ax_states.set_title("State Evolution", fontsize=14)
    ax_states.grid(True, alpha=0.3)
    ax_states.legend(loc="upper left")

    vline = ax_states.axvline(tt[0], color='red', lw=2, linestyle='--')
    pt1, = ax_states.plot([], [], 'bo', markersize=8, markeredgecolor='black')
    pt2, = ax_states.plot([], [], 'co', markersize=8, markeredgecolor='black')

    history_x, history_y = [], []

    def init():
        line_robot.set_data([], [])
        line_trace.set_data([], [])
        vline.set_xdata([tt[0]])
        pt1.set_data([], [])
        pt2.set_data([], [])
        ret = [line_robot, line_trace, vline, pt1, pt2]
        if xx_ref is not None:
            line_robot_ref.set_data([], [])
            ret.append(line_robot_ref)
        return tuple(ret)

    def animate(i):
        line_robot.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        
        if xx_ref is not None:
            line_robot_ref.set_data([0, x1_ref[i], x2_ref[i]], [0, y1_ref[i], y2_ref[i]])

        history_x.append(x2[i])
        history_y.append(y2[i])
        if len(history_x) > 150:
            history_x.pop(0)
            history_y.pop(0)
        line_trace.set_data(history_x, history_y)

        vline.set_xdata([tt[i]])
        pt1.set_data([tt[i]], [th1[i]])
        pt2.set_data([tt[i]], [th2[i]])

        ret = [line_robot, line_trace, vline, pt1, pt2]
        if xx_ref is not None:
            ret.append(line_robot_ref)
        return tuple(ret)

    interval_ms = (tt[1] - tt[0]) * 1000
    print("Rendering animation... (Close the window to exit)")

    ani = animation.FuncAnimation(
        fig, animate, frames=len(tt), 
        init_func=init, blit=True, interval=interval_ms
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    task_name = "task1"
    if len(sys.argv) > 1:
        task_name = sys.argv[1]

    filepath = f"data/optimal_trajectory_{task_name}.npy"

    if not os.path.exists(filepath):
        print(f"Error: Could not find '{filepath}'.")
        print("Make sure you have run the main task script successfully first!")
        sys.exit(1)

    print(f"Loading trajectory from: {filepath}")
    traj_data = np.load(filepath, allow_pickle=True).item()

    xx = traj_data['x']  # State trajectory
    tt = traj_data['t']  # Time vector
    
    # Handle the fact that some tasks might save as (TT, 4) instead of (4, TT)
    if xx.shape[0] != 4:
        xx = xx.T

    animate_trajectory(tt, xx, title=f'Acrobot Optimal Trajectory Animation ({task_name.upper()})')