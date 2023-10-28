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

if __name__ == "__main__":
    # nonlinear ODE - lorenz system
    # xdot = f(x)
    # x(0) = x0

    # params
    sigma = 10
    r = 28
    beta = 8/3

    dts = np.array([0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001])
    x0s = np.array([[-8,5,25],[-10,6,20],[-5,2,15],[-7,1,22]])

    x_EMs_n = np.empty((28,3))
    x_EMs_np1 = np.empty((28,3))
    dt_train = np.empty(28)

    i = 0
    for dt in dts:

        # simulation parameters for local problem
        TB = 0
        t = np.arange(TB,TB+2*dt,dt)
        N = len(t)

        for x0 in x0s:

            # initial conditions
            x_EM = np.zeros((N,3))
            x_EM[0,:] = x0

            # time integration
            for n in range(N-1):
                
                tn = t[n]
                dW = np.sqrt(dt) * np.random.randn(N)

                # Euler-Maruyama method
                x_EM_n = x_EM[n,:]
                x_EM_np1 = x_EM_n + f(x_EM_n)*dt

                x_EMs_n[i,:] = x_EM_n
                x_EMs_np1[i,:] = x_EM_np1
                dt_train[i] = dt

                i += 1


    np.savez('diffusion-training-data', x_EMs_n=x_EMs_n, x_EM_np1=x_EM_np1, dt_train=dt_train )