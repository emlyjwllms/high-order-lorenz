# stochastic Lorenz system

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

# drift vector
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot])

# diffusion matrix
def diff(xv):
    return np.sqrt(dt)*np.ones(3)*np.linalg.norm(xv)

def eulermaruyama(x0,dW,N,dt):
    x_em = np.zeros((N,3))
    x = x0
    for i in range(N):
        x = x + f(x)*dt + diff(x)*dW[i]
        x_em[i,:] = x
    return x_em

def milstein(x0,dW,N,dt):
    x_mil = np.zeros((N,3))
    x = x0
    for i in range(N):
        x = x + f(x)*dt + diff(x)*dW[i] + 0.5*(x*np.sqrt(dt)/np.linalg.norm(x))*diff(x)*(dW[i]**2 - dt)
        x_mil[i,:] = x
    return x_mil

def rk(x0,dW,N,dt):
    x_rk = np.zeros((N,3))
    x = x0
    for i in range(N):
        x_tilde = x + f(x)*dt + diff(x)*np.sqrt(dt)
        x = x + f(x)*dt + diff(x)*dW[i] + 1/(2*np.sqrt(dt))*(diff(x_tilde) - diff(x))*(dW[i]**2 - dt)
        x_rk[i,:] = x
    return x_rk

if __name__ == "__main__":
    # nonlinear ODE - lorenz system
    # xdot = f(x)
    # x(0) = x0

    # params
    sigma = 10
    r = 28
    beta = 8/3

    # simulation parameters
    TA = 10
    TB = 5
    dt = 0.001
    t = np.arange(-TB,TA+TB+1,dt)
    N = len(t)

    mc = 100

    # initial conditions
    x_EM = np.zeros((N,3))
    x_MIL = np.zeros((N,3))
    x_RK = np.zeros((N,3))
    x_EM[0,:] = [-8.67139571762,4.98065219709,25]
    x_MIL[0,:] = [-8.67139571762,4.98065219709,25]
    x_RK[0,:] = [-8.67139571762,4.98065219709,25]
    x_EMs = np.empty((mc,N,3))
    x_MILs = np.empty((mc,N,3))
    x_RKs = np.empty((mc,N,3))

    for i in range(mc):
        np.random.seed(i)
        dW = np.sqrt(dt) * np.random.randn(N)
        W = np.cumsum(dW)

        x_EMs[i] = eulermaruyama(x_EM[0,:],dW,N,dt)
        x_MILs[i] = milstein(x_MIL[0,:],dW,N,dt)
        x_RKs[i] = rk(x_RK[0,:],dW,N,dt)

    E_x_EM = np.sum(x_EMs,0) / mc
    E_x_MIL = np.sum(x_MILs,0) / mc
    E_x_RK = np.sum(x_RKs,0) / mc

    np.savez('stochastic_lorenz_dt1000', t=t, E_x_EM=E_x_EM, E_x_MIL=E_x_MIL, E_x_RK=E_x_RK)
