# DG for Lorenz attractor

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.special import jacobi

from lorenz_functions import *

# xdot function
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot]).T

# residual vector
def resid(c,xh0):
    r = np.zeros((3,porder+1))
    c = np.reshape(c,(3,porder+1))
    for k in range((porder+1)):
        phi_k = phi[:,k] # phi_k (xi_q)
        dphi_k = dphi[:,k] # phi'_k (xi_q)
        phi_k_left = phi_left[:,k] # phi_k (xi_q = -1)
        phi_k_right = phi_right[:,k] # phi_k (xi_q = +1)

        phi_c = np.vstack([[phi @ c[0,:]],[phi @ c[1,:]],[phi @ c[2,:]]])
        
        r[:,k] = np.diag(np.vstack([[dphi_k.T @ np.diag(w) @ phi @ c[0,:]],[dphi_k.T @ np.diag(w) @ phi @ c[1,:]],[dphi_k.T @ np.diag(w) @ phi @ c[2,:]]]) + np.tile(phi_k.T @ np.diag(0.5*dt*w),(3,1)) @ f(phi_c) - np.vstack([[phi_k_right.T @ phi_right @ c[0,:]],[phi_k_right.T @ phi_right @ c[1,:]],[phi_k_right.T @ phi_right @ c[2,:]]]) + np.vstack([[phi_k_left.T @ np.array([xh0[0]])],[phi_k_left.T @ np.array([xh0[1]])],[phi_k_left.T @ np.array([xh0[2]])]]))
    
    return np.reshape(r,(3*(porder+1),))



if __name__ == "__main__":

    porder = 2
    print('porder: ' + str(porder))

    # params
    sigma = 10
    r = 28
    beta = 8/3

    # simulation parameters
    TA = 10
    TB = 5
    dt = 1/10000
    t = np.arange(-TB,TA+TB+1,dt)
    N = len(t)

    # quadrature points
    xi, w = scipy.special.roots_legendre(porder+1)
    Nq = len(xi)
    print("Number of quadrature points: ", str(len(xi)))
    print(str(xi))
    print("Number of quadrature weights: ", str(len(w)))
    print(str(w))

    # initial conditions
    xh = np.zeros((3,N)) # states x elements
    cs = np.zeros((3,porder+1,N))
    # epsilon = 10**-3
    epsilon = 0
    x0 = np.array([-8.67139571762,4.98065219709,25+epsilon]).T
    xh[:,0] = x0
    xhqx = []
    xhqy = []
    xhqz = []
    tq = []

    # precompute polynomials
    phi, dphi = plegendre(xi,porder)
    phi_left, dphi_left = plegendre(-1,porder) # dphi_left not used
    phi_right, dphi_right = plegendre(1,porder) # dphi_right not used

    xh0 = xh[:,0]
    cguess = np.array([np.append(xh0[0],np.zeros(porder)),
                       np.append(xh0[1],np.zeros(porder)),
                       np.append(xh0[2],np.zeros(porder))])

    # integrate across elements
    print('loop through elements')
    for j in range(1,N): # loop across I_j's
        t0 = t[j-1]
        tf = t[j]
        cguess = np.reshape(cguess,(3*(porder+1),)) # reshape from (states x p+1) to (states*(p+1) x 1)
        c = scipy.optimize.root(resid, cguess, args=(xh0,)).x # solve residual function above
        c = np.reshape(c,(3,porder+1)) # reshape back to (states x p+1)
        xhqx = np.append(xhqx,phi @ c[0,:])
        xhqy = np.append(xhqy,phi @ c[1,:])
        xhqz = np.append(xhqz,phi @ c[2,:])
        tq = np.append(tq,dt*xi/2 + (t0+tf)/2)
        cs[:,:,j] = c
        # compute xh
        xh[:,j] = (np.vstack([phi_right @ c[0,:],phi_right @ c[1,:],phi_right @ c[2,:]])).T
        cguess = c
        xh0 = xh[:,j]

    print('saving data')
    np.savez('data/dg_lorenz_dt'+str(int(1/dt))+'_p'+str(porder), xh=xh, cs=cs, t=t)




