import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
import pandas as pd
import xarray as xr

# xdot function
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot]).T

def jacobian(xv):
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

def hessian(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    H1 = np.zeros((3,3)) # sigma*(y-x)
    H1[0][0] = 0 # d^2 f1 / dx^2
    H1[0][1] = 0 # d^2 f1 / dx dy
    H1[0][2] = 0 # d^2 f1 / dx dz
    H1[1][0] = 0 # d^2 f1 / dy dx
    H1[1][1] = 0 # d^2 f1 / dy^2
    H1[1][2] = 0 # d^2 f1 / dy dz
    H1[2][0] = 0 # d^2 f1 / dz dx
    H1[2][1] = 0 # d^2 f1 / dz dy
    H1[2][2] = 0 # d^2 f1 / dz^2

    H2 = np.zeros((3,3)) # x*(r-z) - y
    H2[0][0] = 0 # d^2 f2 / dx^2
    H2[0][1] = 0 # d^2 f2 / dx dy
    H2[0][2] = -1 # d^2 f2 / dx dz
    H2[1][0] = 0 # d^2 f2 / dy dx
    H2[1][1] = 0 # d^2 f2 / dy^2
    H2[1][2] = 0 # d^2 f2 / dy dz
    H2[2][0] = -1 # d^2 f2 / dz dx
    H2[2][1] = 0 # d^2 f2 / dz dy
    H2[2][2] = 0 # d^2 f2 / dz^2

    H3 = np.zeros((3,3)) # x*y - beta*z
    H3[0][0] = 0 # d^2 f3 / dx^2
    H3[0][1] = 1 # d^2 f3 / dx dy
    H3[0][2] = 0 # d^2 f3 / dx dz
    H3[1][0] = 1 # d^2 f3 / dy dx
    H3[1][1] = 0 # d^2 f3 / dy^2
    H3[1][2] = 0 # d^2 f3 / dy dz
    H3[2][0] = 0 # d^2 f3 / dz dx
    H3[2][1] = 0 # d^2 f3 / dz dy
    H3[2][2] = 0 # d^2 f3 / dz^2

    return np.array([H1,H2,H3])

def filter(t,x_IN,Delta):
    
    dt = t[1]-t[0]
    Ne = len(x_IN)
    el = int(Delta/dt)+1

    Xs = np.zeros(Ne)
    Ys = np.zeros(Ne)
    Zs = np.zeros(Ne)

    # apply filter
    for n in np.arange(el,Ne-el):
        ind = np.arange(int(n-el/2),int(n+el/2))
        I_x = x_IN[ind,0]
        I_y = x_IN[ind,1]
        I_z = x_IN[ind,2]
        E_I_x = np.mean(I_x)
        E_I_y = np.mean(I_y)
        E_I_z = np.mean(I_z)
        Xs[n] = E_I_x
        Ys[n] = E_I_y
        Zs[n] = E_I_z

    x_OUT = np.array([Xs,Ys,Zs]).T

    return x_OUT

def lorenz_sgs(Delta):

    lorenz = np.load('dg_lorenz_dt1000_p3.npz')
    x_DG = lorenz['xh']
    t = lorenz['t']
    dt = t[1]-t[0]

    xbar = filter(t,x_DG.T,Delta)
    fbar = filter(t,f(x_DG),Delta)
    fxbar = f(xbar.T)
    s = fbar - fxbar
    fx = f(x_DG)
    xp = x_DG.T - xbar

    return s, fbar, fx, fxbar, x_DG, xbar, xp, Delta, dt, t


if __name__ == "__main__":

    # params
    sigma = 10
    r = 28
    beta = 8/3

    Del = 0.025

    s, fbar, fx, fxbar, x_DG, xbar, xp, Delta, dt, t = lorenz_sgs(Del)

    #s_model = 0.5*xp.T*(hes + Delta**2/12 * jac.T * hes * jac)*xp

    # offset to avoid boundary effects
    el = int(Delta/dt)+1

    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_subplot(3,1,1)
    ax1.plot(t[int(el):-int(el)],s[int(el):-int(el),0])
    ax1.set_ylabel(r"$\mathbf{s}_1$")
    #ax1.set_title("subgrid dynamics")
    ax1.set_yticks(np.array([-1,-0.5,0,0.5,1]))
    ax1.grid()

    ax2 = fig1.add_subplot(3,1,2)
    ax2.plot(t[int(el):-int(el)],s[int(el):-int(el),1])
    ax2.set_ylabel(r"$\mathbf{s}_2$")
    ax2.grid()

    ax3 = fig1.add_subplot(3,1,3)
    ax3.plot(t[int(el):-int(el)],s[int(el):-int(el),2])
    ax3.set_ylabel(r"$\mathbf{s}_3$")
    ax3.set_xlabel(r"$t$")
    ax3.grid()

    for i in range(100):
        jac = jacobian(xbar[i,:])
        hes = hessian(xbar[i,:])
        s_model = 0.5*xp[i,:].T*(hes + Delta**2/12 * jac.T * hes * jac)*xp[i,:]
        print(s_model.shape)

    # plt.show()