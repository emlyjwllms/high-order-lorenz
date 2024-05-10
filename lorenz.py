# Lorenz attractor with forward euler and backward euler for time-stepping

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from lorenz_functions import *

# residual vector
def resid(x):
    return (x - xn)/dt - f(x,jac_par)

def drdx(x):
    return np.eye(3)/dt - dfdx(x,jac_par)

if __name__ == "__main__":
    # nonlinear ODE - lorenz system
    # xdot = f(x)
    # x(0) = x0

    # dictionary with jacobian parameters
    jac_par = {'sigma': 10, 'r': 28, 'beta': 8/3}

    # simulation parameters
    TA = 10
    TB = 5
    dt = 1/10
    t = np.arange(-TB,TA+TB+1,dt)
    N = len(t)

    # initial conditions
    x_FE = np.zeros((N,3))
    x_FE[0,:] = [-8.67139571762,4.98065219709,25]
    x_BE = np.zeros((N,3))
    x_BE[0,:] = [-8.67139571762,4.98065219709,25]
    x_RK = np.zeros((N,3))
    x_RK[0,:] = [-8.67139571762,4.98065219709,25]

    # time integration
    for n in range(N-1):
        tn = t[n]
        # forward Euler
        x_FE[n+1,:] = x_FE[n,:] + f(x_FE[n,:],jac_par)*dt
        # backward Euler
        xn = x_BE[n,:]
        x_BE[n+1,:] = scipy.optimize.root(resid, xn, jac=drdx).x
        # RK4
        k1 = f(x_RK[n,:],jac_par)
        k2 = f(x_RK[n,:] + dt*k1/2,jac_par)
        k3 = f(x_RK[n,:] + dt*k2/2,jac_par)
        k4 = f(x_RK[n,:] + dt*k3,jac_par)
        x_RK[n+1,:] = x_RK[n,:] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

    print('saving data')
    np.savez('data/lorenz_dt'+str(int(1/dt)), t=t, x_FE=x_FE, x_BE=x_BE, x_RK=x_RK )

