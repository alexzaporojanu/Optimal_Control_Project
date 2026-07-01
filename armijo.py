#
# Optimal control of an Acrobot
# Armijo Step Size Selection Rule script
#

import numpy as np
import matplotlib.pyplot as plt
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

      for ii in range(armijo_maxiters):

            # temp solution update
            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))

            xx_temp[:,0] = x0

            for tt in range(TT-1):
                  uu_temp[:,tt] = uu[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx[:,tt]) + stepsize * sigma[:,tt]

                  # If you don't want to use the closed loop (2 step procedure way)
                  # for Newton method comment the line above and uncomment the line below

                  # uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
                  xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])

            # temp cost calculation
            JJ_temp = 0

            for tt in range(TT-1):
                  temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
                  JJ_temp += temp_cost

            temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QT)[0]
            JJ_temp += temp_cost

            # save the stepsize
            stepsizes.append(stepsize)      

            # save the cost associated to the stepsize
            costs_armijo.append(JJ_temp)    

            if JJ_temp > JJ + cc*stepsize*descent_arm:
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

            for ii_plot in range(len(steps)):
                  step = steps[ii_plot]

                  # temp solution update
                  xx_temp = np.zeros((ns,TT))
                  uu_temp = np.zeros((ni,TT))

                  xx_temp[:,0] = x0
                  for tt in range(TT-1):
                        uu_temp[:,tt] = uu[:,tt] + KK[:, :, tt] @ (xx_temp[:,tt] - xx[:,tt]) + step * sigma[:,tt]
                        
                        # If you don't want to use the closed loop (2 step procedure way)
                        # for Newton method comment the line above and uncomment the line below
                        
                        # uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
                        xx_temp[:,tt+1] = dyn.step(xx_temp[:,tt], uu_temp[:,tt])
                  
                  # temp cost computation
                  JJ_temp = 0
                  for tt in range(TT-1):
                        temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt], Qt, Rt)[0]
                        JJ_temp += temp_cost

                  temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1],QT)[0]
                  JJ_temp += temp_cost

                  costs[ii_plot] = JJ_temp


            plt.figure(1)
            plt.clf()

      
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - \\gamma^k*d^k)$', linewidth = 2)
            plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - \\gamma^k*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$', linewidth = 2)
            plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - c*\\gamma^k\\nabla J(\\mathbf{u}^k)^{\\top} d^k$', linewidth = 2)

            # plot the tested stepsize
            plt.scatter(stepsizes, costs_armijo, marker='*', s=100, zorder = 5) 

            plt.grid()
            plt.xlabel('$\\gamma^k$')
            plt.ylabel("$g(\\gamma^k)$")
            plt.legend()
            plt.title(f"Armijo rule at iteration k = {kk} | Cost J={JJ:.3e} | Descent={abs(descent_arm):.3e}")
            plt.draw()

            if save_path is not None:
                  plt.savefig(save_path)
            plt.show(block=False)
            plt.pause(0.001)

      return stepsize