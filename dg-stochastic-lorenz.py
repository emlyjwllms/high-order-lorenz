# DG for (stochastic) Lorenz attractor

# filtered dynamics

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

# xdot function
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot]).T + s_model

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
    # epsilon = 10**-3
    epsilon = 0
    x0 = np.array([-8.67139571762,4.98065219709,25+epsilon]).T
    xh[:,0] = x0
    xhqx = []
    xhqy = []
    xhqz = []
    tq = []

    Delta = 0.04

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

        jac = jacobian(xh0)
        hes = hessian(xh0)

        #xpi = xp[:,i]

        xpi = np.array([np.random.randn(),np.random.randn(),np.random.randn()])

        s_model = 0.5*np.matmul(np.matmul(xpi.T,hes + (Delta**2)/12 * jac.T * hes * jac),xpi)

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


    np.savez('dg_stochastic_lorenz_dt100_p'+str(porder), xh=xh, cs=cs, t=t, Delta=Delta)




