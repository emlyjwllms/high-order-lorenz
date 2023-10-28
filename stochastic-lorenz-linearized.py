# Lorenz attractor with Euler-Maruyama method

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

def naive_model(alpha, stabilize, lambda_max = 0.95):
    if stabilize:
        mu = alpha * np.sqrt(2*lambda_max)
    else:
        mu = 0
    return mu*xtilde[n,:]*dW[n]

if __name__ == "__main__":
    # nonlinear ODE - lorenz system
    # xdot = f(x)
    # x(0) = x0

    # params
    sigma = 10
    r = 28
    beta = 8/3

    ############### STABILIZE
    stabilize = True

    # simulation parameters
    TA = 10
    TB = 5
    dt = 0.001
    t = np.arange(-TB,TA+TB+1,dt)
    N = len(t)

    # initial conditions
    x_EM = np.zeros((N,3))
    xtilde = np.zeros((N,3))
    x_EM[0,:] = [-8.67139571762,4.98065219709,25]
    xtilde[0,:] = [1,1,1]

    # time integration
    for n in range(N-1):
        tn = t[n]
        dW = np.sqrt(dt) * np.random.randn(N)

        # Euler-Maruyama method
        x_EM[n+1,:] = x_EM[n,:] + f(x_EM[n,:])*dt

        alpha = 1.5
        xtilde[n+1,:] = xtilde[n,:] + np.matmul(dfdx(x_EM[n,:]),xtilde[n,:])*dt + naive_model(alpha,stabilize)


    plt.figure(1)
    plt.plot(t,x_EM[:,0],label=r"$x$")
    plt.plot(t,x_EM[:,1],label=r"$y$")
    plt.plot(t,x_EM[:,2],label=r"$z$")
    plt.xlabel("t")
    plt.ylabel(r"$\mathbf{x}$")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(t,xtilde[:,0],label=r"$\tilde{x}$")
    plt.plot(t,xtilde[:,1],label=r"$\tilde{y}$")
    plt.plot(t,xtilde[:,2],label=r"$\tilde{z}$")
    plt.xlabel("t")
    plt.title(r"$\alpha$ = " + str(alpha))
    plt.ylabel(r"$\tilde{\mathbf{x}}$")
    if not stabilize:
        plt.ylim(-200,200)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.plot(t,np.log(np.abs(xtilde[:,0])),label=r"$\tilde{x}$")
    plt.plot(t,np.log(np.abs(xtilde[:,1])),label=r"$\tilde{y}$")
    plt.plot(t,np.log(np.abs(xtilde[:,2])),label=r"$\tilde{z}$")
    plt.xlabel("t")
    plt.yscale("log")
    plt.title(r"$\alpha$ = " + str(alpha))
    plt.ylabel(r"$\log(|\tilde{\mathbf{x}}|)$")
    plt.legend()
    plt.grid()
    plt.show()

    #np.savez('lorenz_dt100', t=t, x_EM=x_EM )