# plotting code for TLM

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


dg_lorenz = np.load('dg_lorenz_dt100_p2.npz')
xh = dg_lorenz['xh']
cs = dg_lorenz['cs']
xh_tilde = dg_lorenz['xh_tilde']
cs_tilde = dg_lorenz['cs_tilde']
t = dg_lorenz['t']

plot_quad = False

N = len(t)
dt = t[1] - t[0]
porder = 2

xi, w = scipy.special.roots_legendre(porder+1)
phi, dphi = plegendre(xi,porder)

fig1 = plt.figure(figsize=(12,4))
ax1 = fig1.add_subplot(1,2,2)
ax2 = fig1.add_subplot(1,2,1)

if plot_quad:
    # integrate across elements
    for j in range(1,N): # loop across I_j's
        t0 = t[j-1]
        tf = t[j]
        c_tilde = cs_tilde[:,:,j] # solve residual function above
        c = cs[:,:,j]
        ax1.scatter(dt*xi/2 + (t0+tf)/2, phi @ c_tilde[0,:])
        ax1.scatter(dt*xi/2 + (t0+tf)/2, phi @ c_tilde[1,:])
        ax1.scatter(dt*xi/2 + (t0+tf)/2, phi @ c_tilde[2,:])
        ax2.scatter(dt*xi/2 + (t0+tf)/2, phi @ c[0,:])
        ax2.scatter(dt*xi/2 + (t0+tf)/2, phi @ c[1,:])
        ax2.scatter(dt*xi/2 + (t0+tf)/2, phi @ c[2,:])

ax1.plot(t,xh_tilde[0,:],color='C0')
ax1.set_ylabel(r"$\tilde{\mathbf{x}}$")

ax1.plot(t,xh_tilde[1,:],color='C1')

ax1.plot(t,xh_tilde[2,:],color='C2')
ax1.set_ylim(-20,20)

ax2.plot(t,xh[0,:],color='C0',label=r"$x$")
ax2.set_ylabel(r"$\mathbf{x}$")

ax2.plot(t,xh[1,:],color='C1',label=r"$y$")

ax2.plot(t,xh[2,:],color='C2',label=r"$z$")

ax1.set_xlabel(r"$t$")
ax2.set_xlabel(r"$t$")
ax2.legend()

#plt.savefig('plots/dg-lorenz-tangent-linear.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
plt.show()

# Lyapunov covariant vectors

lambda_1 = 0.95
lambda_2 = 1.1
lambda_3 = 1.05

dg_lorenz_perturbed = np.load('dg_lorenz_dt100_p2_perturbed.npz')
xh_perturbed = dg_lorenz_perturbed['xh']
cs_perturbed = dg_lorenz_perturbed['cs']

fig0 = plt.figure(figsize=(14,4))
ax1 = fig0.add_subplot(1,3,1)
ax1.semilogy(t,np.sqrt(np.abs(xh_perturbed[0,:]-xh[0,:])**2),'C0')
ax1.semilogy(t,10**-1.5*np.exp(lambda_1*t),'k--',label=r"$\exp(\lambda_1 t)$ with $\lambda_1$ = " + str(lambda_1))
ax1.legend(loc='lower right')
ax1.set_xlabel(r"$t$")
ax1.set_title(r"$x$")
ax1.set_ylabel(r"$||\mathbf{x}_h^p - \mathbf{x}_h||$")
ax1.set_ylim(10**-6,10**2)

ax2 = fig0.add_subplot(1,3,2)
ax2.semilogy(t,np.sqrt(np.abs(xh_perturbed[1,:]-xh[1,:])**2),'C1')
ax2.semilogy(t,10**-1.5*np.exp(lambda_2*t),'k--',label=r"$\exp(\lambda_2 t)$ with $\lambda_2$ = " + str(lambda_2))
ax2.legend(loc='lower right')
ax2.set_xlabel(r"$t$")
ax2.set_title(r"$y$")
ax2.set_ylim(10**-6,10**2)

ax3 = fig0.add_subplot(1,3,3)
ax3.semilogy(t,np.sqrt(np.abs(xh_perturbed[2,:]-xh[2,:])**2),'C2')
ax3.semilogy(t,10**-1*np.exp(lambda_3*t),'k--',label=r"$\exp(\lambda_3 t)$ with $\lambda_3$ = " + str(lambda_3))
ax3.legend(loc='lower right')
ax3.set_xlabel(r"$t$")
ax3.set_title(r"$z$")
ax3.set_ylim(10**-5,10**2)

plt.savefig('plots/dg-lorenz-tangent-linear-lyapunov-spectrum.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
plt.show()

phi_1 = xh_tilde*np.exp(-lambda_1*t)
phi_2 = xh_tilde*np.exp(-lambda_2*t)
phi_3 = xh_tilde*np.exp(-lambda_3*t)

fig1 = plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot(3,1,1)
ax2 = fig1.add_subplot(3,1,2)
ax3 = fig1.add_subplot(3,1,3)

ax1.plot(t,phi_1[0,:],label=r"$x$")
ax1.plot(t,phi_1[1,:],label=r"$y$")
ax1.plot(t,phi_1[2,:],label=r"$z$")
ax1.set_ylabel(r"$\phi_1$")
ax1.set_ylim(-50,50)

ax2.plot(t,phi_2[0,:],label=r"$x$")
ax2.plot(t,phi_2[1,:],label=r"$y$")
ax2.plot(t,phi_2[2,:],label=r"$z$")
ax2.set_ylabel(r"$\phi_2$")
ax2.set_ylim(-60,60)

ax3.plot(t,phi_3[0,:],label=r"$x$")
ax3.plot(t,phi_3[1,:],label=r"$y$")
ax3.plot(t,phi_3[2,:],label=r"$z$")
ax3.set_xlabel(r"$t$")
ax3.set_ylabel(r"$\phi_3$")
ax3.set_ylim(-60,60)

plt.savefig('plots/dg-lorenz-tangent-linear-lyapunov-vectors.png',dpi=300,format='png',transparent=True,bbox_inches='tight')
plt.show()




