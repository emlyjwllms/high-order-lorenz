# plotting and dynamics functions for the Lorenz system

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm
import scipy

def cumulative_mean(t,x,bnd):

    if bnd != 0:
        i_t = np.linspace(1,len(t[bnd:-bnd]),len(t[bnd:-bnd]))
        x_mean_x = np.cumsum(x[0,bnd:-bnd])/i_t
        x_mean_y = np.cumsum(x[1,bnd:-bnd])/i_t
        x_mean_z = np.cumsum(x[2,bnd:-bnd])/i_t
        X = np.array([x_mean_x,x_mean_y,x_mean_z])
    if bnd == 0:
        i_t = np.linspace(1,len(t),len(t))
        x_mean_x = np.cumsum(x[0,:])/i_t
        x_mean_y = np.cumsum(x[1,:])/i_t
        x_mean_z = np.cumsum(x[2,:])/i_t
        X = np.array([x_mean_x,x_mean_y,x_mean_z])

    return X


def interp_3(t,T,X):

    x_x = np.interp(t,T,X[:,0])
    x_y = np.interp(t,T,X[:,1])
    x_z = np.interp(t,T,X[:,2])
    x = np.array([x_x, x_y, x_z])

    return x

def plot_3(t,data_dictionary,tlabel,xlabel,ylabel,zlabel,figname,tight=True,show=True,save=True):
    plt.figure(figsize=(12,4))

    for label in data_dictionary:
        data = data_dictionary[label]
        plt.subplot(1,3,1)
        plt.plot(t,data[:,0],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(xlabel)
        # plt.ylim([-22,22])
        plt.legend()
        plt.grid(True)

        plt.subplot(1,3,2)
        plt.plot(t,data[:,1],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(ylabel)
        # plt.ylim([-30,25])
        plt.legend()
        plt.grid(True)

        plt.subplot(1,3,3)
        plt.plot(t,data[:,2],label=label)
        plt.xlabel(tlabel)
        plt.ylabel(zlabel)
        # plt.ylim([-2,55])
        plt.legend()
        plt.grid(True)

    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(figname,dpi=300,format='png',transparent=True)
    if show:
        plt.show()

    return

def plot_3d(data_dictionary,xlabel,ylabel,zlabel,figname,tight=False,show=True,save=True):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(1,1,1,projection='3d')

    for label in data_dictionary:
        data = data_dictionary[label]
        ax.plot(*data.T,label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.legend()
    if tight:
        plt.tight_layout()
    if save:
        plt.savefig(figname,dpi=300,format='png',transparent=True)
    if show:
        plt.show()

    return

# xdot function
def f(xv, jac_par):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    sigma = jac_par['sigma']
    r = jac_par['r']
    beta = jac_par['beta']
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot])

def dfdx(xv, jac_par):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    sigma = jac_par['sigma']
    r = jac_par['r']
    beta = jac_par['beta']
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

def filter(fl,N,Delta,dt,quad_par,jac_par):

    # number of elements within the filter width
    el = int(Delta/dt)

    cs = quad_par['cs']
    w = quad_par['w']
    phi = quad_par['phi']
    
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
                fx = f(np.array([xhqx,xhqy,xhqz]),jac_par).T
                xhqxs += np.sum(w @ fx[:,0] * dt)/2
                xhqys += np.sum(w @ fx[:,1] * dt)/2
                xhqzs += np.sum(w @ fx[:,2] * dt)/2
        # calculate sum, \sum_q{w_q * x_h * dt} (basically the integral part of Eq. 5.1)
        barx[j] = xhqxs
        bary[j] = xhqys
        barz[j] = xhqzs

    return np.array([barx,bary,barz])/Delta