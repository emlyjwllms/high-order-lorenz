# stochastic Lorenz attractor with forward euler and backward euler for time-stepping

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

# diffusion matrix
def diff(xyz,lamb=0.1):
    x,y,z = xyz
    return lamb*(x+y+z)*np.ones(3)

# residual vector for BE
def resid(x):
    return (x - xn)/dt - f(x)

def drdx(x):
    return np.eye(3)/dt - dfdx(x)

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

if __name__ == "__main__":
    # nonlinear ODE - lorenz system
    # x' = f(x)
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

    # initial conditions
    x_FE = np.zeros((N,3))
    x_FE[0,:] = [-8.67139571762,4.98065219709,25]
    x_BE = np.zeros((N,3))
    x_BE[0,:] = [-8.67139571762,4.98065219709,25]

    # Brownian increment
    dW = np.sqrt(dt) * np.random.normal(0,1,size=N)

    # time integration
    for n in range(N-1):
        tn = t[n]
        # forward Euler
        x_FE[n+1,:] = x_FE[n,:] + f(x_FE[n,:])*dt
        x_FE[n+1,:] = x_FE[n,:] + f(x_FE[n,:])*dt + diff(x_FE[n,:])*dW[n]
        # backward Euler
        #xn = x_BE[n,:]
        #x_BE[n+1,:] = scipy.optimize.root(resid, xn, jac=drdx).x



    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    ax1.plot(*x_FE.T,label="FE")
    #ax1.plot(*x_BE.T,label="BE")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()
    plt.show()

    fig1 = plt.figure(figsize=(7,7))
    ax1 = fig1.add_subplot(3,1,1)
    ax1.plot(t,x_FE[:,0],label="FE")
    #ax1.plot(t,x_BE[:,0],label="BE")
    ax1.set_title("X")
    ax1.legend()
    ax1.grid()

    ax2 = fig1.add_subplot(3,1,2)
    ax2.plot(t,x_FE[:,1],label="FE")
    #ax2.plot(t,x_BE[:,1],label="BE")
    ax2.set_title("Y")
    ax2.legend()
    ax2.grid()

    ax3 = fig1.add_subplot(3,1,3)
    ax3.plot(t,x_FE[:,2],label="FE")
    #ax3.plot(t,x_BE[:,2],label="BE")
    ax3.set_title("Z")
    ax3.legend()
    ax3.grid()
    plt.show()




