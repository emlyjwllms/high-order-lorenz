# Lorenz attractor nonlinear solution - training data generation

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

# xdot function
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot])

def dfdx(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    J = np.zeros((3,3))
    J[0][0] = -sigma # df1/dx
    J[0][1] = sigma # df1/dy
    J[0][2] = 0 # df1/dz
    J[1][0] = r-z # df2/dx
    J[1][1] = -1 # df2/dy
    J[1][2] = -x # df2/dz
    J[2][0] = y # df3/dx
    J[2][1] = x # df3/dy
    J[2][2] = -beta # df3/dz
    return J

def naive_model(alpha, lambda_max = 0.95):
    mu = alpha * np.sqrt(2*lambda_max)
    return mu*xtilde[n,:]


if __name__ == "__main__":
    # nonlinear ODE - lorenz system
    # xdot = f(x)
    # x(0) = x0

    # params
    sigma = 10
    r = 28
    beta = 8/3

    #dts = np.array([0.01,0.005,0.001])
    #x0s = np.array([0.5,0.6,0.7,0.8,0.9]) # factor in front of the perturbation

    x_n = []
    x_np1 = []
    dt_train = []
    alpha_train = []

    i = 0
    print(i)
    while i <= 1000: # generate 100 samples
        #for dt in dts:

        # simulation parameters for global problem (need to check stability criterion)
        TA = 10
        TB = 5
        dt = np.random.uniform(0.01,0.001)
        t = np.arange(-TB,TA+TB+1,dt)
        N = len(t)

            #for x0 in x0s:

        # initial conditions
        x_EM = np.zeros((N,3))
        x_EM[0,:] = [-8.67139571762,4.98065219709,25]

        xtilde = np.zeros((N,3))
        x0 = np.random.uniform(0.1,1)
        xtilde[0,:] = x0 * np.array([1,1,1])

        # randomly select alpha with mean 1.25 and std 0.1 (edge case of stability basically - will give us values >= 1 most of the time)
        alpha = np.random.normal(1.25,0.1)
        
        # if 80% (4/5) of runs are stable, keep the triple (x0,x1,dt) and alpha
        jj = 0
        for c in range(5):
            # time integration
            for n in range(N-1):
                
                tn = t[n]
                dW = np.sqrt(dt) * np.random.randn(N)

                # Euler-Maruyama method
                x_EM[n+1,:] = x_EM[n,:] + f(x_EM[n,:])*dt

                xtilde[n+1,:] = xtilde[n,:] + np.matmul(dfdx(x_EM[n,:]),xtilde[n,:])*dt + naive_model(alpha)*dW[n]

            # check stability criterion log(abs(xtilde)) <= 1 for last half of the trajectory
            if np.max(np.log(np.abs(xtilde[int(N/2):,:]))) <= 1:
                jj += 1
                xtilde_np1 = xtilde[1,:] # save (one of) the stable cases

        if jj >= 4:
            # add to training set
            x_n.append(xtilde[0,:]) # save x0
            x_np1.append(xtilde_np1) # save x1
            dt_train.append(dt) # save dt
            alpha_train.append(alpha) # save alpha

            # plt.plot(t,np.log(np.abs(xtilde[:,0])),label=r"$\tilde{x}$")
            # plt.plot(t,np.log(np.abs(xtilde[:,1])),label=r"$\tilde{y}$")
            # plt.plot(t,np.log(np.abs(xtilde[:,2])),label=r"$\tilde{z}$")
            # plt.xlabel("t")
            # plt.yscale("log")
            # plt.title(r"$\alpha$ = " + str(alpha))
            # plt.ylabel(r"$\log(|\tilde{\mathbf{x}}|)$")
            # plt.legend()
            # plt.grid()
            # plt.show()

            print(str(xtilde[0,:]) + ", dt = " + str(dt) + ", alpha = " + str(alpha) + ", i = " + str(i))
            i += 1


    np.savez('diffusion-training-data', x_n=x_n, x_np1=x_np1, dt_train=dt_train, alpha_train=alpha_train)