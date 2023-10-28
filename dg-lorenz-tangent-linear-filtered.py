# DG tangent linear model (TLM) for Lorenz attractor filtered

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

# xdot_tilde function
def fbar_tilde(xvbar_tilde,xvbar):
    return np.matmul(jac+jac_sgs,xvbar_tilde).T

# xbardot function
def fbar(xvbar):
    return f(xvbar) + sgs(xvbar)

# xdot function
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot]).T

def sgs(xvbar):
    s = 0.5*np.matmul(np.matmul(xp0.T,hes + (Delta**2)/12 * jac.T * hes * jac),xp0)
    return s

# partial s / partial x
def jacobian_sgs(xvbar):
    J = (Delta**2) * np.matmul(np.matmul(xp0.T,jac.T * dJ * hes),xp0) / 12
    return J

def partial_jacobian(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    J1 = np.zeros((3,3)) # partial J / partial x
    J1[0][0] = 0
    J1[0][1] = 0
    J1[0][2] = 0
    J1[1][0] = 0
    J1[1][1] = 0
    J1[1][2] = 1
    J1[2][0] = 0
    J1[2][1] = -1
    J1[2][2] = 0

    J2 = np.zeros((3,3)) # partial J / partial y
    J2[0][0] = 0
    J2[0][1] = 0
    J2[0][2] = 1
    J2[1][0] = 0
    J2[1][1] = 0
    J2[1][2] = 0
    J2[2][0] = 0
    J2[2][1] = 0
    J2[2][2] = 0

    J3 = np.zeros((3,3)) # partial J / partial z
    J3[0][0] = 0
    J3[0][1] = -1
    J3[0][2] = 0
    J3[1][0] = 0
    J3[1][1] = 0
    J3[1][2] = 0
    J3[2][0] = 0
    J3[2][1] = 0
    J3[2][2] = 0

    return np.array([J1,J2,J3])

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

# residual vector
def resid_filtered(c,xh0bar):
    r = np.zeros((3,porder+1))
    c = np.reshape(c,(3,porder+1))
    for k in range((porder+1)):
        phi_k = phi[:,k] # phi_k (xi_q)
        dphi_k = dphi[:,k] # phi'_k (xi_q)
        phi_k_left = phi_left[:,k] # phi_k (xi_q = -1)
        phi_k_right = phi_right[:,k] # phi_k (xi_q = +1)

        phi_c = np.vstack([[phi @ c[0,:]],[phi @ c[1,:]],[phi @ c[2,:]]])
        
        r[:,k] = np.diag(np.vstack([[dphi_k.T @ np.diag(w) @ phi @ c[0,:]],[dphi_k.T @ np.diag(w) @ phi @ c[1,:]],[dphi_k.T @ np.diag(w) @ phi @ c[2,:]]]) + np.tile(phi_k.T @ np.diag(0.5*dt*w),(3,1)) @ fbar(phi_c) - np.vstack([[phi_k_right.T @ phi_right @ c[0,:]],[phi_k_right.T @ phi_right @ c[1,:]],[phi_k_right.T @ phi_right @ c[2,:]]]) + np.vstack([[phi_k_left.T @ np.array([xh0bar[0]])],[phi_k_left.T @ np.array([xh0bar[1]])],[phi_k_left.T @ np.array([xh0bar[2]])]]))
    
    return np.reshape(r,(3*(porder+1),))

# tangent linear residual vector
def tangent_resid_filtered(c,xh0bar_tilde):
    r = np.zeros((3,porder+1))
    c = np.reshape(c,(3,porder+1))
    for k in range((porder+1)):
        phi_k = phi[:,k] # phi_k (xi_q)
        dphi_k = dphi[:,k] # phi'_k (xi_q)
        phi_k_left = phi_left[:,k] # phi_k (xi_q = -1)
        phi_k_right = phi_right[:,k] # phi_k (xi_q = +1)

        phi_c = np.vstack([[phi @ c[0,:]],[phi @ c[1,:]],[phi @ c[2,:]]])
        
        r[:,k] = np.diag(np.vstack([[dphi_k.T @ np.diag(w) @ phi @ c[0,:]],[dphi_k.T @ np.diag(w) @ phi @ c[1,:]],[dphi_k.T @ np.diag(w) @ phi @ c[2,:]]]) + np.tile(phi_k.T @ np.diag(0.5*dt*w),(3,1)) @ fbar_tilde(phi_c,xh0bar) - np.vstack([[phi_k_right.T @ phi_right @ c[0,:]],[phi_k_right.T @ phi_right @ c[1,:]],[phi_k_right.T @ phi_right @ c[2,:]]]) + np.vstack([[phi_k_left.T @ np.array([xh0bar_tilde[0]])],[phi_k_left.T @ np.array([xh0bar_tilde[1]])],[phi_k_left.T @ np.array([xh0bar_tilde[2]])]]))
    
    return np.reshape(r,(3*(porder+1),))

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
    dt = 1/100
    t = np.arange(-TB,TA+TB+1,dt)
    N = len(t)

    Delta = 0.04

    N = len(t)
    xi, w = scipy.special.roots_legendre(porder+1)
    phi, dphi = plegendre(xi,porder)

    # quadrature points
    xi, w = scipy.special.roots_legendre(porder+1)
    Nq = len(xi)
    print("Number of quadrature points: ", str(len(xi)))
    print(str(xi))
    print("Number of quadrature weights: ", str(len(w)))
    print(str(w))

    # initial conditions
    xhbar = np.zeros((3,N)) # states x elements
    csbar = np.zeros((3,porder+1,N))
    x0bar = np.array([-8.67139571762,4.98065219709,25]).T
    xhbar[:,0] = x0bar
    xhqxbar = []
    xhqybar = []
    xhqzbar = []
    tq = []

    # initial conditions tangent linear model
    xhbar_tilde = np.zeros((3,N)) # states x elements
    csbar_tilde = np.zeros((3,porder+1,N))
    epsilon = 10e-3
    x0bar_tilde = np.array([epsilon,epsilon,epsilon]).T
    xhbar_tilde[:,0] = x0bar_tilde
    xhqxbar_tilde = []
    xhqybar_tilde = []
    xhqzbar_tilde = []

    # precompute polynomials
    phi, dphi = plegendre(xi,porder)
    phi_left, dphi_left = plegendre(-1,porder) # dphi_left not used
    phi_right, dphi_right = plegendre(1,porder) # dphi_right not used

    xh0bar = xhbar[:,0]
    cguessbar = np.array([np.append(xh0bar[0],np.zeros(porder)),
                       np.append(xh0bar[1],np.zeros(porder)),
                       np.append(xh0bar[2],np.zeros(porder))])
    xh0bar_tilde = xhbar_tilde[:,0]
    cguessbar_tilde = np.array([np.append(xh0bar_tilde[0],np.zeros(porder)),
                       np.append(xh0bar_tilde[1],np.zeros(porder)),
                       np.append(xh0bar_tilde[2],np.zeros(porder))])

    # integrate across elements
    print('loop through elements')
    for j in range(1,N): # loop across I_j's
        t0 = t[j-1]
        tf = t[j]

        xp0 = np.random.randn(3)

        jac = jacobian(xh0bar)
        hes = hessian(xh0bar)
        dJ = partial_jacobian(xh0bar)
        jac_sgs = jacobian_sgs(xh0bar)

        cguessbar = np.reshape(cguessbar,(3*(porder+1),)) # reshape from (states x p+1) to (states*(p+1) x 1)
        cguessbar_tilde = np.reshape(cguessbar_tilde,(3*(porder+1),))

        cbar = scipy.optimize.root(resid_filtered, cguessbar_tilde, args=(xh0bar,)).x # solve residual function above
        cbar_tilde = scipy.optimize.root(tangent_resid_filtered, cguessbar_tilde, args=(xh0bar_tilde,)).x

        cbar = np.reshape(cbar,(3,porder+1)) # reshape back to (states x p+1)
        cbar_tilde = np.reshape(cbar_tilde,(3,porder+1))

        xhqxbar = np.append(xhqxbar,phi @ cbar[0,:])
        xhqybar = np.append(xhqybar,phi @ cbar[1,:])
        xhqzbar = np.append(xhqzbar,phi @ cbar[2,:])

        xhqxbar_tilde = np.append(xhqxbar_tilde,phi @ cbar_tilde[0,:])
        xhqybar_tilde = np.append(xhqybar_tilde,phi @ cbar_tilde[1,:])
        xhqzbar_tilde = np.append(xhqzbar_tilde,phi @ cbar_tilde[2,:])

        tq = np.append(tq,dt*xi/2 + (t0+tf)/2)

        csbar[:,:,j] = cbar
        csbar_tilde[:,:,j] = cbar_tilde

        # compute xhbar
        xhbar[:,j] = (np.vstack([phi_right @ cbar[0,:],phi_right @ cbar[1,:],phi_right @ cbar[2,:]])).T
        xhbar_tilde[:,j] = (np.vstack([phi_right @ cbar_tilde[0,:],phi_right @ cbar_tilde[1,:],phi_right @ cbar_tilde[2,:]])).T
        
        cguessbar = cbar
        cguessbar_tilde = cbar_tilde

        xh0bar = xhbar[:,j]
        xh0bar_tilde = xhbar_tilde[:,j]

    np.savez('dg_lorenz_dt100_p'+str(porder)+'_stoch', xhbar = xhbar, csbar=csbar, xhbar_tilde=xhbar_tilde, csbar_tilde=csbar_tilde, t=t)

