# CAN DELETE

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

# f = xdot function for Lorenz
def f(xv):
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot]).T

# for Legendre basis
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

# Jacobian of Lorenz system
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

# Hessian of Lorenz system
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


if __name__ == "__main__":

    # Restrictions:
    # Delta must be a factor of dt (and divisible by 2)

    # params
    sigma = 10
    r = 28
    beta = 8/3

    porder = 2 # 0, 1, 2, 3

    # filter width
    Delta = 0.04
    
    # load in simulation data for dt = 0.01 and porder
    lorenz = np.load('dg_lorenz_dt100_p' + str(porder) + '.npz')
    xh = lorenz['xh'] # element points (right boundary of each element in the simulation), I use these points to calculate x'
    cs = lorenz['cs'] # coefficients (quadrature)
    t = lorenz['t'] # time array from original simulation
    dt = t[1]-t[0] # dt of original simulation (0.01)

    N = len(t) # number of elements in original simulation
    xi, w = scipy.special.roots_legendre(porder+1) # quadrature points and weights
    phi, dphi = plegendre(xi,porder) # legendre basis functions

    # number of elements within the filter width
    el = int(Delta/dt)

    # storage arrays
    xbarx = np.zeros(N)
    xbary = np.zeros(N)
    xbarz = np.zeros(N)

    # FIND X-BAR

    # loop through elements of original simulation
    for j in range(el,N-el): # loop across I_j's
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
            xhqxs += np.sum(w @ xhqx * dt)/2
            xhqys += np.sum(w @ xhqy * dt)/2
            xhqzs += np.sum(w @ xhqz * dt)/2
        # calculate sum, \sum_q{w_q * x_h * dt} (basically the integral part of Eq. 5.1)
        xbarx[j] = xhqxs
        xbary[j] = xhqys
        xbarz[j] = xhqzs

    xbar = np.array([xbarx,xbary,xbarz])/Delta


    # calculate x'
    # this is primarily what the model is based on
    # I think there is a more rigorous way to do this, and that I'm oversimplifying it

    xp = xh - xbar # xh = right boundary of each element in original simulation (phi_right * c)
    

    # plot states
    fig1 = plt.figure(figsize=(20,10))
    ax1 = fig1.add_subplot(3,1,1)
    ax2 = fig1.add_subplot(3,1,2)
    ax3 = fig1.add_subplot(3,1,3)

    ax1.plot(t,xh[0,:],label="element points")
    ax1.plot(t,xbar[0,:],label="filtered")
    ax1.plot(t,xp[0,:],label="perturbation")
    ax1.legend()
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")
    
    ax2.plot(t,xh[1,:])
    ax2.plot(t,xbar[1,:])
    ax2.plot(t,xp[1,:])
    ax2.set_xlabel("t")
    ax2.set_ylabel("y")

    ax3.plot(t,xh[2,:])
    ax3.plot(t,xbar[2,:])
    ax3.plot(t,xp[2,:])
    ax3.set_xlabel("t")
    ax3.set_ylabel("z")

    ax1.set_title('states filter check')
    plt.show()


    # storage arrays
    fbarx = np.zeros(N)
    fbary = np.zeros(N)
    fbarz = np.zeros(N)

    # FIND F-BAR (basically f(x) instead of x now)

    # loop through elements of original simulation
    for j in range(el,N-el): # loop across I_j's
        c = cs[:,:,j-int(el/2):j+int(el/2)] # coefficients, for x_h = c*phi, for elements within filter width
        fxhqxs = 0
        fxhqys = 0
        fxhqzs = 0
        for ele in range(el):
            ci = c[:,:,ele] # coeffs for one element at a time
            # calculate x,y,z approximations, x_h = c*phi
            xhqx = phi @ ci[0,:]
            xhqy = phi @ ci[1,:]
            xhqz = phi @ ci[2,:]
            fx = f(np.array([xhqx,xhqy,xhqz]))
            fxhqxs += np.sum(w @ fx[:,0] * dt)/2
            fxhqys += np.sum(w @ fx[:,1] * dt)/2
            fxhqzs += np.sum(w @ fx[:,2] * dt)/2
        # calculate sum, \sum_q{w_q * f(x_h) * dt} (basically the integral part of Eq. 5.1)
        fbarx[j] = fxhqxs
        fbary[j] = fxhqys
        fbarz[j] = fxhqzs

    fbar = np.array([fbarx,fbary,fbarz])/Delta

    fxbar = f(xbar).T # f(xbar)

    # plot dynamics
    fig2 = plt.figure(figsize=(20,10))
    ax1 = fig2.add_subplot(3,1,1)
    ax2 = fig2.add_subplot(3,1,2)
    ax3 = fig2.add_subplot(3,1,3)

    ax1.plot(t,fbar[0,:],label="bar{f(x)}")
    ax1.plot(t,fxbar[0,:],label="f(xbar)")
    ax1.legend()
    ax1.set_xlabel("t")
    ax1.set_ylabel(r"$f_1$")
    
    ax2.plot(t,fbar[1,:])
    ax2.plot(t,fxbar[1,:])
    ax2.set_xlabel("t")
    ax2.set_ylabel(r"$f_2$")

    ax3.plot(t,fbar[2,:])
    ax3.plot(t,fxbar[2,:])
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$f_3$")

    ax1.set_title('dynamics filter check')
    plt.show()


    # calculate subgrid dynamics

    s = fbar - fxbar

    s_model = np.zeros((3,len(xbar.T)))
    s_random = np.zeros((3,len(xbar.T)))

    # loop through elements
    for i in range(len(xbar.T)):

        jac = jacobian(xbar[:,i])
        hes = hessian(xbar[:,i])

        xpi = xp[:,i]

        xpi_random = np.array([np.random.randn(),np.random.randn(),np.random.randn()])

        s_model[:,i] = 0.5*np.matmul(np.matmul(xpi.T,hes + (Delta**2)/12 * jac.T * hes * jac),xpi)
        s_random[:,i] = 0.5*np.matmul(np.matmul(xpi_random.T,hes + (Delta**2)/12 * jac.T * hes * jac),xpi_random)


    fig1 = plt.figure(figsize=(12,8))
    ax1 = fig1.add_subplot(3,1,1)
    ax1.plot(t,s[0,:])
    ax1.plot(t[el:-el],s_model[0,el:-el],label=r"$\mathbf{s}_{model}$")
    ax1.plot(t[el:-el],s_random[0,el:-el],label=r"$\mathbf{s}_{random}$")
    ax1.set_ylabel(r"$\mathbf{s}_1$")
    ax1.set_title(r"$\Delta t$ = " + str(round(dt,3)) + ", $\Delta$ = " + str(Delta) + ", $p$ = " + str(porder))
    ax1.set_yticks(np.array([-1,-0.5,0,0.5,1]))
    ax1.legend()
    ax1.grid()

    ax2 = fig1.add_subplot(3,1,2)
    ax2.plot(t,s[1,:])
    ax2.plot(t[el:-el],s_model[1,el:-el])
    ax2.plot(t[el:-el],s_random[1,el:-el])
    ax2.set_ylabel(r"$\mathbf{s}_2$")
    ax2.grid()

    ax3 = fig1.add_subplot(3,1,3)
    ax3.plot(t,s[2,:])
    ax3.plot(t[el:-el],s_model[2,el:-el])
    ax3.plot(t[el:-el],s_random[2,el:-el])
    ax3.set_ylabel(r"$\mathbf{s}_3$")
    ax3.set_xlabel(r"$t$")
    ax3.grid()

    # plt.savefig('plots/lorenz-sgs-model-Delta-'+str(Delta)+'.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

    plt.show()

    fig4 = plt.figure(figsize=(12,8))
    ax1 = fig4.add_subplot(3,1,1)
    ax1.plot(t[el:-el],s_random[0,el:-el] + fxbar[0,el:-el],label=r"bar{f(x)}_{random}")
    ax1.plot(t[el:-el],s_model[0,el:-el] + fxbar[0,el:-el],label=r"bar{f(x)}_{model}")
    ax1.plot(t,fbar[0,:],label="bar{f(x)}")
    ax1.set_ylabel(r"bar{f(x)}_1")
    ax1.set_title(r"s_{random} + fxbar = fbar_{random}")
    ax1.legend()
    ax1.grid()

    ax2 = fig4.add_subplot(3,1,2)
    ax2.plot(t[el:-el],s_random[1,el:-el] + fxbar[1,el:-el],label=r"bar{f(x)}_{random}")
    ax2.plot(t[el:-el],s_model[1,el:-el] + fxbar[1,el:-el],label=r"bar{f(x)}_{model}")
    ax2.plot(t,fbar[1,:],label="bar{f(x)}")
    ax2.set_ylabel(r"bar{f(x)}_2")
    ax2.grid()

    ax3 = fig4.add_subplot(3,1,3)
    ax3.plot(t[el:-el],s_random[2,el:-el] + fxbar[2,el:-el],label=r"bar{f(x)}_{random}")
    ax3.plot(t[el:-el],s_model[2,el:-el] + fxbar[2,el:-el],label=r"bar{f(x)}_{model}")
    ax3.plot(t,fbar[2,:],label="bar{f(x)}")
    ax3.set_ylabel(r"bar{f(x)}_3")
    ax3.set_xlabel(r"$t$")
    ax3.grid()

    # plt.savefig('plots/lorenz-sgs-model-Delta-'+str(Delta)+'.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

    plt.show()