#
# Optimal control of an Acrobot
# Armijo Step Size Selection Rule script
#

import numpy as np
import matplotlib.pyplot as plt
import mpld3
import cost as cst
import dynamics as dyn
import data as cfg


def select_stepsize(stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref,  x0, uu, xx, KK, sigma, JJ, descent_arm, kk, Qt, Rt, QT, armijo_plot = False, armijo_plot_number = 3, save_path=None):

      TT = uu.shape[1]


      stepsizes = []  # list of stepsizes
      costs_armijo = []

      stepsize = stepsize_0

      ns = xx_ref.shape[0]
      ni = uu_ref.shape[0]

      with np.errstate(over='ignore', invalid='ignore'):
            for ii in range(armijo_maxiters):

                  # temp solution update
                  xx_temp = np.zeros((ns, TT + 1))
                  uu_temp = np.zeros((ni, TT))

                  xx_temp[:,0] = x0

                  for tt in range(TT):
                        uu_temp[:,tt] = uu[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx[:,tt]) + stepsize * sigma[:,tt]
                        xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])

                  # temp cost calculation
                  JJ_temp = 0

                  for tt in range(TT):
                        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
                        JJ_temp += temp_cost

                  temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QT)[0]
                  JJ_temp += temp_cost

                  # save the stepsize
                  stepsizes.append(stepsize)      

                  # save the cost associated to the stepsize
                  costs_armijo.append(JJ_temp)    

                  if np.isnan(JJ_temp) or not np.isfinite(JJ_temp) or JJ_temp > JJ + cc*stepsize*descent_arm:
                        # update the stepsize
                        stepsize = beta*stepsize

                  else:
                        #print(f'Armijo stepsize = {stepsize} at iteration k = {kk}')
                        break
                  
                  if ii == armijo_maxiters -1:
                        print("WARNING: no stepsize was found with armijo rule!")

      # print(f"Armijo at iteration {kk}: stepsize = {stepsize}")  
            
      ############################
      # Descent Armijo Plot
      ############################
      plt.rcParams["figure.figsize"] = (10,6)

      if armijo_plot and (kk < 1 or kk%10 == 0 or kk==armijo_plot_number or kk==7):

            # stepsizes for visualization (from 0 to stepsize_0 = 1, adding the armijo intermediate steps for better visualization)
            steps = list(np.linspace(0,stepsize_0,int(cfg.armijo_plot_resolution)))
            for iteration in range(ii):
                  arm_step_size = beta**(iteration+1)
                  if not (arm_step_size in steps): 
                        steps.append(beta**(iteration+1))
            steps.sort()
            steps = np.array(steps)

            costs = np.zeros(len(steps))

            with np.errstate(over='ignore', invalid='ignore'):
                  for ii_plot in range(len(steps)):
                        step = steps[ii_plot]

                        # temp solution update
                        xx_temp = np.zeros((ns, TT + 1))
                        uu_temp = np.zeros((ni, TT))

                        xx_temp[:,0] = x0
                        for tt in range(TT):
                              uu_temp[:,tt] = uu[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx[:,tt]) + step * sigma[:,tt]
                              xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])
                        
                        # temp cost computation
                        JJ_temp = 0
                        for tt in range(TT):
                              temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
                              JJ_temp += temp_cost

                        temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QT)[0]
                        JJ_temp += temp_cost

                        costs[ii_plot] = JJ_temp


            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - \\gamma^k*d^k)$', linewidth = 2)
            ax.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - \\gamma^k*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$', linewidth = 2)
            ax.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - c*\\gamma^k\\nabla J(\\mathbf{u}^k)^{\\top} d^k$', linewidth = 2)

            # plot the tested stepsize
            ax.scatter(stepsizes, costs_armijo, marker='*', s=100, zorder = 5) 

            ax.grid()
            ax.set_xlabel('$\\gamma^k$')
            ax.set_ylabel("$g(\\gamma^k)$")
            ax.legend()
            ax.set_title(f"Armijo rule at iteration k = {kk} | Cost J={JJ:.3e} | Descent={abs(descent_arm):.3e}")
            fig.tight_layout()

            if save_path is not None:
                  plt.savefig(save_path)
                  html_path = save_path.replace('.png', '.html')
                  mpld3.save_html(fig, html_path)
            plt.close(fig)

      return stepsize