# DG for stochastic Lorenz

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
    return (np.array([xdot,ydot,zdot]) - 0.5*diff(xv)*(xv*np.sqrt(dt)/np.linalg.norm(xv)) + diff(xv)*dW[j]/dt).T

# diffusion matrix
def diff(xv):
    return np.sqrt(dt)*np.ones(3)*np.linalg.norm(xv)

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
    #print(np.shape(r))
    #print(r)
    #print(dr_dc)
    return np.reshape(r,(3*(porder+1),))

def gaussquad1d(pgauss):

    """     
    gaussquad1d calculates the gauss integration points in 1d for [-1,1]
    [x,w]=gaussquad1d(pgauss)

      x:         coordinates of the integration points 
      w:         weights  
      pgauss:         order of the polynomila integrated exactly 
    """

    n = math.ceil((pgauss+1)/2)
    P = jacobi(n, 0, 0)
    x = np.sort(np.roots(P))

    A = np.zeros((n,n))
    for i in range(1,n+1):
        P = jacobi(i-1,0,0)
        A[i-1,:] = np.polyval(P,x)

    r = np.zeros((n,), dtype=float)
    r[0] = 2.0
    w = np.linalg.solve(A,r)

    # map from [-1,1] to [0,1]
    #x = (x + 1.0)/2.0
    #w = w/2.0

    return x, w

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

    porder = 3
    print('porder: ' + str(porder))

    # params
    sigma = 10
    r = 28
    beta = 8/3

    # simulation parameters
    TA = 10
    TB = 5
    dt = 0.01
    t = np.arange(-TB,TA+TB+1,dt)
    N = len(t)

    # quadrature points
    xi, w = gaussquad1d(porder+1)
    Nq = len(xi)
    print("Number of quadrature points: ", str(len(xi)))
    print(str(xi))
    print("Number of quadrature weights: ", str(len(w)))
    print(str(w))

    # precompute polynomials
    phi, dphi = plegendre(xi,porder)
    phi_left, dphi_left = plegendre(-1,porder) # dphi_left not used
    phi_right, dphi_right = plegendre(1,porder) # dphi_right not used

    
    mc = 100

    xhs = np.empty((mc,3,N))

    for i in range(mc):
        np.random.seed(i)
        dW = np.sqrt(dt) * np.random.randn(N)
        W = np.cumsum(dW)

        # initial conditions
        xh = np.zeros((3,N)) # states x elements
        x0 = np.array([-8.67139571762,4.98065219709,25]).T
        xh[:,0] = x0

        xh0 = xh[:,0]
        cguess = np.array([np.append(xh0[0],np.zeros(porder)),
                        np.append(xh0[1],np.zeros(porder)),
                        np.append(xh0[2],np.zeros(porder))])
        #print(cguess)
        # integrate across elements
        #print('loop through elements')
        for j in range(1,N): # loop across I_j's
            t0 = t[j-1]
            tf = t[j]
            #print(cguess)
            # optimize.root can only take 1D arrays
            cguess = np.reshape(cguess,(3*(porder+1),)) # reshape from (states x p+1) to (states*(p+1) x 1)
            c = scipy.optimize.root(resid, cguess, args=(xh0,)).x # solve residual function above
            c = np.reshape(c,(3,porder+1)) # reshape back to (states x p+1)
            #print(c)
            # compute xh
            xh[:,j] = (np.vstack([phi_right @ c[0,:],phi_right @ c[1,:],phi_right @ c[2,:]])).T
            cguess = c
            xh0 = xh[:,j]
        
        xhs[i,:,:] = xh

    E_xh = np.mean(xhs,0)

    np.savez('dg_stochastic_lorenz_dt100_p3', E_xh=E_xh)




