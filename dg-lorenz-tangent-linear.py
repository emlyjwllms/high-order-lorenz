# DG tangent linear model (TLM) for Lorenz attractor

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
def f_tilde(xv_tilde,xv):
    # x_tilde = xv_tilde[0]
    # y_tilde = xv_tilde[1]
    # z_tilde = xv_tilde[2]
    jac = jacobian(xv)
    return np.matmul(jac,xv_tilde).T

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

# tangent linear residual vector
def tangent_resid(c,xh0_tilde):
    r = np.zeros((3,porder+1))
    c = np.reshape(c,(3,porder+1))
    for k in range((porder+1)):
        phi_k = phi[:,k] # phi_k (xi_q)
        dphi_k = dphi[:,k] # phi'_k (xi_q)
        phi_k_left = phi_left[:,k] # phi_k (xi_q = -1)
        phi_k_right = phi_right[:,k] # phi_k (xi_q = +1)

        phi_c = np.vstack([[phi @ c[0,:]],[phi @ c[1,:]],[phi @ c[2,:]]])
        
        r[:,k] = np.diag(np.vstack([[dphi_k.T @ np.diag(w) @ phi @ c[0,:]],[dphi_k.T @ np.diag(w) @ phi @ c[1,:]],[dphi_k.T @ np.diag(w) @ phi @ c[2,:]]]) + np.tile(phi_k.T @ np.diag(0.5*dt*w),(3,1)) @ f_tilde(phi_c,xh0) - np.vstack([[phi_k_right.T @ phi_right @ c[0,:]],[phi_k_right.T @ phi_right @ c[1,:]],[phi_k_right.T @ phi_right @ c[2,:]]]) + np.vstack([[phi_k_left.T @ np.array([xh0_tilde[0]])],[phi_k_left.T @ np.array([xh0_tilde[1]])],[phi_k_left.T @ np.array([xh0_tilde[2]])]]))
    
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
    x0 = np.array([-8.67139571762,4.98065219709,25]).T
    xh[:,0] = x0
    xhqx = []
    xhqy = []
    xhqz = []
    tq = []

    # initial conditions tangent linear model
    xh_tilde = np.zeros((3,N)) # states x elements
    cs_tilde = np.zeros((3,porder+1,N))
    epsilon = 10e-3
    x0_tilde = np.array([epsilon,epsilon,epsilon]).T
    xh_tilde[:,0] = x0_tilde
    xhqx_tilde = []
    xhqy_tilde = []
    xhqz_tilde = []

    # precompute polynomials
    phi, dphi = plegendre(xi,porder)
    phi_left, dphi_left = plegendre(-1,porder) # dphi_left not used
    phi_right, dphi_right = plegendre(1,porder) # dphi_right not used

    xh0 = xh[:,0]
    xh0_tilde = xh_tilde[:,0]
    cguess = np.array([np.append(xh0[0],np.zeros(porder)),
                       np.append(xh0[1],np.zeros(porder)),
                       np.append(xh0[2],np.zeros(porder))])
    cguess_tilde = np.array([np.append(xh0_tilde[0],np.zeros(porder)),
                       np.append(xh0_tilde[1],np.zeros(porder)),
                       np.append(xh0_tilde[2],np.zeros(porder))])

    # integrate across elements
    print('loop through elements')
    for j in range(1,N): # loop across I_j's
        t0 = t[j-1]
        tf = t[j]

        cguess = np.reshape(cguess,(3*(porder+1),)) # reshape from (states x p+1) to (states*(p+1) x 1)
        cguess_tilde = np.reshape(cguess_tilde,(3*(porder+1),))

        c = scipy.optimize.root(resid, cguess, args=(xh0,)).x # solve residual function above
        c_tilde = scipy.optimize.root(tangent_resid, cguess_tilde, args=(xh0_tilde,)).x

        c = np.reshape(c,(3,porder+1)) # reshape back to (states x p+1)
        c_tilde = np.reshape(c_tilde,(3,porder+1))

        xhqx = np.append(xhqx,phi @ c[0,:])
        xhqy = np.append(xhqy,phi @ c[1,:])
        xhqz = np.append(xhqz,phi @ c[2,:])

        xhqx_tilde = np.append(xhqx_tilde,phi @ c_tilde[0,:])
        xhqy_tilde = np.append(xhqy_tilde,phi @ c_tilde[1,:])
        xhqz_tilde = np.append(xhqz_tilde,phi @ c_tilde[2,:])

        tq = np.append(tq,dt*xi/2 + (t0+tf)/2)

        cs[:,:,j] = c
        cs_tilde[:,:,j] = c_tilde

        # compute xh
        xh[:,j] = (np.vstack([phi_right @ c[0,:],phi_right @ c[1,:],phi_right @ c[2,:]])).T
        xh_tilde[:,j] = (np.vstack([phi_right @ c_tilde[0,:],phi_right @ c_tilde[1,:],phi_right @ c_tilde[2,:]])).T
        
        cguess = c
        cguess_tilde = c_tilde

        xh0 = xh[:,j]
        xh0_tilde = xh_tilde[:,j]


    np.savez('dg_lorenz_dt100_p'+str(porder), xh=xh, cs=cs, xh_tilde=xh_tilde, cs_tilde=cs_tilde, t=t)




