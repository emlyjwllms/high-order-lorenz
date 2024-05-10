# Lorenz attractor with spatial-filtered dynamics

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

def plegendre(x,porder):
    
    try:
        y = np.zeros((len(x),porder+1))
        dy = np.zeros((len(x),porder+1))
        ddy = np.zeros((len(x),porder+1))
    except TypeError: # if passing in single x-point
        y = np.zeros((1,porder+1))
        dy = np.zeros((1,porder+1))
        ddy = np.zeros((1,porder+1))

    y[:,0] = 1
    dy[:,0] = 0
    ddy[:,0] = 0

    if porder >= 1:
        y[:,1] = x
        dy[:,1] = 1
        ddy[:,1] = 0
    
    for i in np.arange(1,porder):
        y[:,i+1] = ((2*i+1)*x*y[:,i]-i*y[:,i-1])/(i+1)
        dy[:,i+1] = ((2*i+1)*x*dy[:,i]+(2*i+1)*y[:,i]-i*dy[:,i-1])/(i+1)
        ddy[:,i+1] = ((2*i+1)*x*ddy[:,i]+2*(2*i+1)*dy[:,i]-i*ddy[:,i-1])/(i+1)

    # return y,dy,ddy
    return y,dy

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

def filter(fl):
    
    barx = np.zeros(N)
    bary = np.zeros(N)
    barz = np.zeros(N)

    for j in range(el, N-el):
        c = cs[:,:,j-int(el/2):j+int(el/2)] # coefficients, for x_h = c*phi, for elements within filter width
        xhqxs = 0
        xhqys = 0
        xhqzs = 0
        for ele in range(el):
            ci = c[:,:,ele] # coeffs for one element at a time
            # calculate x,y,z approximations, x_h = c*phi
            xhqx = phi @ ci[0,:]
            xhqy = phi @ ci[1,:]
            xhqz = phi @ ci[2,:]
            if fl == 'x':
                xhqxs += np.sum(w @ xhqx * dt)/2
                xhqys += np.sum(w @ xhqy * dt)/2
                xhqzs += np.sum(w @ xhqz * dt)/2
            if fl == 'f(x)':
                fx = f(np.array([xhqx,xhqy,xhqz]))
                xhqxs += np.sum(w @ fx[:,0] * dt)/2
                xhqys += np.sum(w @ fx[:,1] * dt)/2
                xhqzs += np.sum(w @ fx[:,2] * dt)/2
        # calculate sum, \sum_q{w_q * x_h * dt} (basically the integral part of Eq. 5.1)
        barx[j] = xhqxs
        bary[j] = xhqys
        barz[j] = xhqzs

    return np.array([barx,bary,barz])/Delta

if __name__ == "__main__":

    # params
    sigma = 10
    r = 28
    beta = 8/3

    porder = 2

    lorenz = np.load('data/dg_lorenz_dt100_p' + str(porder) + '.npz')
    # stoch_lorenz = np.load('dg_stochastic_lorenz_dt100_p' + str(porder) + '.npz')
    # xbar_sgs = stoch_lorenz['xh']

    Delta = 0.2 #stoch_lorenz['Delta']
    

    xh = lorenz['xh']
    cs = lorenz['cs']
    t = lorenz['t']
    dt = t[1]-t[0]

    N = len(t)
    xi, w = scipy.special.roots_legendre(porder+1)
    phi, dphi = plegendre(xi,porder)

    # number of elements within the filter width
    el = int(Delta/dt)

    xbar = filter('x')
    xp = xh - xbar

    fx = f(xh).T

    fbar = filter('f(x)')

    fxbar = f(xbar).T

    s = fbar - fxbar

    s_model = np.zeros((3,len(xbar.T)))
    s_random = np.zeros((3,len(xbar.T)))

    for i in range(len(xbar.T)):

        jac = jacobian(xbar[:,i])
        hes = hessian(xbar[:,i])

        xpi = xp[:,i]

        xpi_random = np.array([np.random.randn(),np.random.randn(),np.random.randn()])

        s_model[:,i] = 0.5*np.matmul(np.matmul(xpi.T,hes + (Delta**2)/12 * jac.T * hes * jac),xpi)
        s_random[:,i] = 0.5*np.matmul(np.matmul(xpi_random.T,hes + (Delta**2)/12 * jac.T * hes * jac),xpi_random)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.scatter(xbar[0,el:-el],s[0,el:-el],s=5)
    plt.xlabel(r"$\overline{\mathbf{x}}_{1}$")
    plt.ylabel(r"${\mathbf{s}}_{1}$")
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.scatter(xbar[1,el:-el],s[1,el:-el],s=5)
    plt.xlabel(r"$\overline{\mathbf{x}}_{2}$")
    plt.ylabel(r"${\mathbf{s}}_{2}$")
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.scatter(xbar[2,el:-el],s[2,el:-el],s=5)
    plt.xlabel(r"$\overline{\mathbf{x}}_{3}$")
    plt.ylabel(r"${\mathbf{s}}_{3}$")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(4,4))
    ax1 = fig.add_subplot(1,1,1,projection='3d')
    ax1.plot(*xbar,label=r"true")
    # ax1.plot(*xbar_sgs,label=r"model")
    ax1.set_xlabel(r"$\overline{\mathbf{x}}$")
    ax1.set_ylabel(r"$\overline{\mathbf{y}}$")
    ax1.set_zlabel(r"$\overline{\mathbf{z}}$")
    ax1.set_box_aspect(aspect=None, zoom=0.8)
    ax1.legend()
    fig.tight_layout()

    plt.savefig('plots/dg-lorenz-3d-filtered.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
    plt.show()

    fig2 = plt.figure(figsize=(10,8))
    ax11 = fig2.add_subplot(3,1,1,projection='3d')
    p = ax11.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=s[0,el:-el],s=10,cmap=mpl.cm.Blues)
    ax11.set_xlabel(r"$\overline{\mathbf{x}}$")
    ax11.set_ylabel(r"$\overline{\mathbf{y}}$")
    ax11.set_zlabel(r"$\overline{\mathbf{z}}$")
    ax11.set_title(r"$\Delta$ = " + str(Delta))
    fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax11,label=r"$\mathbf{s}_x$")

    ax21 = fig2.add_subplot(3,1,2,projection='3d')
    p = ax21.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=s[1,el:-el],s=50,cmap=mpl.cm.Greens)
    ax21.set_xlabel(r"$\overline{\mathbf{x}}$")
    ax21.set_ylabel(r"$\overline{\mathbf{y}}$")
    ax21.set_zlabel(r"$\overline{\mathbf{z}}$")
    fig2.colorbar(p,shrink=0.5,pad=0.1,ax=ax21,label=r"$\mathbf{s}_y$")

    ax31 = fig2.add_subplot(3,1,3,projection='3d')
    p = ax31.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=s[2,el:-el],s=5,cmap=mpl.cm.Oranges)
    ax31.set_xlabel(r"$\overline{\mathbf{x}}$")
    ax31.set_ylabel(r"$\overline{\mathbf{y}}$")
    ax31.set_zlabel(r"$\overline{\mathbf{z}}$")
    fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax31,label=r"$\mathbf{s}_z$")

    plt.show()













