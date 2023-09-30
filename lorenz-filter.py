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

    lorenz = np.load('dg_lorenz_dt100_p' + str(porder) + '.npz')

    plots = True

    widths = np.array([0.02,0.04,0.1,0.2,0.4,1])
    #widths = np.array([0.04])

    fig3 = plt.figure(figsize=(12,10))
    ax311 = fig3.add_subplot(3,2,1)
    ax321 = fig3.add_subplot(3,2,3)
    ax331 = fig3.add_subplot(3,2,5)
    ax312 = fig3.add_subplot(3,2,2)
    ax322 = fig3.add_subplot(3,2,4)
    ax332 = fig3.add_subplot(3,2,6)

    fig39 = plt.figure(figsize=(12,10))
    ax3119 = fig39.add_subplot(3,2,1)
    ax3219 = fig39.add_subplot(3,2,3)
    ax3319 = fig39.add_subplot(3,2,5)
    ax3129 = fig39.add_subplot(3,2,2)
    ax3229 = fig39.add_subplot(3,2,4)
    ax3329 = fig39.add_subplot(3,2,6)

    fig33 = plt.figure(figsize=(20,10))
    ax3311 = fig33.add_subplot(3,3,1)
    ax3321 = fig33.add_subplot(3,3,4)
    ax3331 = fig33.add_subplot(3,3,7)
    ax3312 = fig33.add_subplot(3,3,2)
    ax3322 = fig33.add_subplot(3,3,5)
    ax3332 = fig33.add_subplot(3,3,8)
    ax33129 = fig33.add_subplot(3,3,3)
    ax33229 = fig33.add_subplot(3,3,6)
    ax33329 = fig33.add_subplot(3,3,9)

    fig55 = plt.figure(figsize=(20,10))
    ax41 = fig55.add_subplot(3,3,1)
    ax42 = fig55.add_subplot(3,3,4)
    ax43 = fig55.add_subplot(3,3,7)
    ax44 = fig55.add_subplot(3,3,2)
    ax45 = fig55.add_subplot(3,3,5)
    ax46 = fig55.add_subplot(3,3,8)
    ax47 = fig55.add_subplot(3,3,3)
    ax48 = fig55.add_subplot(3,3,6)
    ax49 = fig55.add_subplot(3,3,9)
    

    for Delta in widths:

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

        Delta = round(Delta,3)

        if plots == True:
    
            # plot filtered states: x, xbar, x'

            fig1 = plt.figure(figsize=(20,10))
            ax1 = fig1.add_subplot(3,3,1)
            ax1.plot(t[el:-el],xh[0,el:-el],label=r"$\mathbf{x}$")
            ax1.plot(t[el:-el],xbar[0,el:-el],label=r"$\overline{\mathbf{x}}$")
            ax1.plot(t[el:-el],xp[0,el:-el],label=r"${\mathbf{x'}}$")
            ax1.set_ylabel(r"$x$")
            ax1.legend(loc='upper left')
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,4)
            ax2.plot(t[el:-el],xh[1,el:-el],label=r"$\mathbf{x}$")
            ax2.plot(t[el:-el],xbar[1,el:-el],label=r"$\overline{\mathbf{x}}$")
            ax2.plot(t[el:-el],xp[1,el:-el],label=r"${\mathbf{x'}}$")
            ax2.set_ylabel(r"$y$")
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,7)
            ax3.plot(t[el:-el],xh[2,el:-el],label=r"$\mathbf{x}$")
            ax3.plot(t[el:-el],xbar[2,el:-el],label=r"$\overline{\mathbf{x}}$")
            ax3.plot(t[el:-el],xp[2,el:-el],label=r"${\mathbf{x'}}$")
            ax3.set_ylabel(r"$z$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            # plot filtered dynamics: f(x), f(xbar), fbar(x)

            ax1 = fig1.add_subplot(3,3,2)
            ax1.plot(t[el:-el],fx[0,el:-el],label=r"$\mathbf{f}(\mathbf{x})$")
            ax1.plot(t[el:-el],fxbar[0,el:-el],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax1.plot(t[el:-el],fbar[0,el:-el],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax1.set_ylabel(r"$\mathbf{f}_1$")
            ax1.legend(loc='upper left')
            ax1.set_title(r"$\Delta = $" + str(Delta))
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,5)
            ax2.plot(t[el:-el],fx[1,el:-el],label=r"$\mathbf{f}(\mathbf{x})$")
            ax2.plot(t[el:-el],fxbar[1,el:-el],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax2.plot(t[el:-el],fbar[1,el:-el],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax2.set_ylabel(r"$\mathbf{f}_2$")
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,8)
            ax3.plot(t[el:-el],fx[2,el:-el],label=r"$\mathbf{f}(\mathbf{x})$")
            ax3.plot(t[el:-el],fxbar[2,el:-el],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax3.plot(t[el:-el],fbar[2,el:-el],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax3.set_ylabel(r"$\mathbf{f}_3$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            # plot subgrid dynamics

            ax1 = fig1.add_subplot(3,3,3)
            ax1.plot(t[el:-el],s[0,el:-el])
            ax1.set_ylabel(r"$\mathbf{s}_1$")
            #ax1.set_title("subgrid dynamics")
            ax1.set_yticks(np.array([-1,-0.5,0,0.5,1]))
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,6)
            ax2.plot(t[el:-el],s[1,el:-el])
            ax2.set_ylabel(r"$\mathbf{s}_2$")
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,9)
            ax3.plot(t[el:-el],s[2,el:-el])
            ax3.set_ylabel(r"$\mathbf{s}_3$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            plt.savefig('plots/lorenz-filter-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

            # plot subgrid model vs exact

            fig4 = plt.figure(figsize=(8,8))
            ax1 = fig4.add_subplot(3,1,1)
            ax1.plot(t[el:-el],s[0,el:-el],label=r"$\mathbf{s}$")
            ax1.plot(t[el:-el],s_model[0,el:-el],label=r"$\mathbf{s}_{model}$")
            ax1.plot(t[el:-el],s_random[0,el:-el],label=r"$\mathbf{s}_{random}$",zorder=1)
            ax1.set_ylabel(r"$\mathbf{s}_1$")
            ax1.set_title(r"$\Delta t$ = " + str(round(dt,3)) + ", $\Delta$ = " + str(Delta) + ", $p$ = " + str(porder))
            ax1.set_yticks(np.array([-1,-0.5,0,0.5,1]))
            ax1.legend()
            ax1.grid()

            ax2 = fig4.add_subplot(3,1,2)
            ax2.plot(t[el:-el],s[1,el:-el])
            ax2.plot(t[el:-el],s_model[1,el:-el])
            ax2.plot(t[el:-el],s_random[1,el:-el],zorder=1)
            ax2.set_ylabel(r"$\mathbf{s}_2$")
            ax2.grid()

            ax3 = fig4.add_subplot(3,1,3)
            ax3.plot(t[el:-el],s[2,el:-el])
            ax3.plot(t[el:-el],s_model[2,el:-el])
            ax3.plot(t[el:-el],s_random[2,el:-el],zorder=1)
            ax3.set_ylabel(r"$\mathbf{s}_3$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            plt.savefig('plots/lorenz-s-model-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

            # plot reconstructed dynamics from random model

            fig44 = plt.figure(figsize=(8,8))
            ax1 = fig44.add_subplot(3,1,1)
            ax1.plot(t[el:-el],s_random[0,el:-el] + fxbar[0,el:-el],label=r"$\overline{\mathbf{f}}_{random}$")
            ax1.plot(t[el:-el],s_model[0,el:-el] + fxbar[0,el:-el],label=r"$\overline{\mathbf{f}}_{model}$")
            ax1.plot(t[el:-el],fbar[0,el:-el],label=r"$\overline{\mathbf{f}}$")
            ax1.set_ylabel(r"$\overline{\mathbf{f}}_1$")
            ax1.set_title(r"$\mathbf{s} + \mathbf{f}(\overline{\mathbf{x}}) = \overline{\mathbf{f}}$")
            ax1.legend()
            ax1.grid()

            ax2 = fig44.add_subplot(3,1,2)
            ax2.plot(t[el:-el],s_random[1,el:-el] + fxbar[1,el:-el])
            ax2.plot(t[el:-el],s_model[1,el:-el] + fxbar[1,el:-el])
            ax2.plot(t[el:-el],fbar[1,el:-el])
            ax2.set_ylabel(r"$\overline{\mathbf{f}}_2$")
            ax2.grid()

            ax3 = fig44.add_subplot(3,1,3)
            ax3.plot(t[el:-el],s_random[2,el:-el] + fxbar[2,el:-el])
            ax3.plot(t[el:-el],s_model[2,el:-el] + fxbar[2,el:-el])
            ax3.plot(t[el:-el],fbar[2,el:-el])
            ax3.set_ylabel(r"$\overline{\mathbf{f}}_3$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            plt.savefig('plots/lorenz-fbar-model-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

            # phase plots for subgrid and perturbation dynamics

            # subgrid s plots
            fig2 = plt.figure(figsize=(10,8))
            ax11 = fig2.add_subplot(3,2,1,projection='3d')
            p = ax11.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=s[0,el:-el],s=10,cmap=mpl.cm.Blues)
            ax11.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax11.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax11.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            ax11.set_title(r"$\Delta$ = " + str(round(Delta,3)))
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax11,label=r"$\mathbf{s}_1$")

            ax21 = fig2.add_subplot(3,2,3,projection='3d')
            p = ax21.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=s[1,el:-el],s=10,cmap=mpl.cm.Greens)
            ax21.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax21.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax21.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax21,label=r"$\mathbf{s}_2$")

            ax31 = fig2.add_subplot(3,2,5,projection='3d')
            p = ax31.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=s[2,el:-el],s=5,cmap=mpl.cm.Oranges)
            ax31.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax31.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax31.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax31,label=r"$\mathbf{s}_3$")

            # perturbation x' plots
            ax12 = fig2.add_subplot(3,2,2,projection='3d')
            p = ax12.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=xp[0,el:-el],s=10,cmap=mpl.cm.Blues)
            ax12.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax12.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax12.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax12,label=r"$\mathbf{x'}_1$")

            ax22 = fig2.add_subplot(3,2,4,projection='3d')
            p = ax22.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=xp[1,el:-el],s=10,cmap=mpl.cm.Greens)
            ax22.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax22.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax22.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax22,label=r"$\mathbf{x'}_2$")

            ax32 = fig2.add_subplot(3,2,6,projection='3d')
            p = ax32.scatter(xbar[0,el:-el],xbar[1,el:-el],xbar[2,el:-el],c=xp[2,el:-el],s=10,cmap=mpl.cm.Oranges)
            ax32.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax32.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax32.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax32,label=r"$\mathbf{x'}_3$")

            plt.savefig('plots/lorenz-sgs-xp-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

            #plt.show()

        ax311.plot(t[el:-el],s[0,el:-el],color="tab:blue", alpha=Delta)
        ax311.set_ylabel(r"$\mathbf{s}_1$")
        ax311.set_yticks(np.array([-1,-0.5,0,0.5,1]))
        ax311.grid()

        ax321.plot(t[el:-el],s[1,el:-el],color="tab:green", alpha=Delta)
        ax321.set_ylabel(r"$\mathbf{s}_2$")
        ax321.grid()

        ax331.plot(t[el:-el],s[2,el:-el],color="tab:orange", alpha=Delta)
        ax331.set_xlabel(r"$t$")
        ax331.set_ylabel(r"$\mathbf{s}_3$")
        ax331.grid()

        ax312.plot(t[el:-el],xp[0,el:-el],color="tab:blue", alpha=Delta)
        ax312.set_ylabel(r"$\mathbf{x'}_1$")
        ax312.grid()

        ax322.plot(t[el:-el],xp[1,el:-el],color="tab:green", alpha=Delta)
        ax322.set_ylabel(r"$\mathbf{x'}_2$")
        ax322.grid()

        ax332.plot(t[el:-el],xp[2,el:-el],color="tab:orange", alpha=Delta)
        ax332.set_xlabel(r"$t$")
        ax332.set_ylabel(r"$\mathbf{x'}_3$")
        ax332.grid()

        fig3.savefig('plots/lorenz-sgs-xp-del.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

        if Delta == 0.02 or Delta == 0.2:

            if Delta == 0.02:
                linest = '--'
            if Delta == 0.2:
                linest = '-'

            ax3119.plot(t[el:-el],s[0,el:-el]/(Delta**2),linest,color="tab:blue",label=r"$\Delta$ = " + str(Delta))
            ax3119.set_ylabel(r"$\mathbf{s}_1/\Delta^2$")
            ax3119.set_yticks(np.array([-1,-0.5,0,0.5,1]))
            ax3119.grid()

            ax3219.plot(t[el:-el],s[1,el:-el]/(Delta**2),linest,color="tab:green")
            ax3219.set_ylabel(r"$\mathbf{s}_2/\Delta^2$")
            ax3219.grid()

            ax3319.plot(t[el:-el],s[2,el:-el]/(Delta**2),linest,color="tab:orange")
            ax3319.set_xlabel(r"$t$")
            ax3319.set_ylabel(r"$\mathbf{s}_3/\Delta^2$")
            ax3319.grid()

            ax3129.plot(t[el:-el],xp[0,el:-el]/Delta,linest,color="tab:blue")
            ax3129.set_ylabel(r"$\mathbf{x'}_1/\Delta$")
            ax3129.grid()

            ax3229.plot(t[el:-el],xp[1,el:-el]/Delta,linest,color="tab:green")
            ax3229.set_ylabel(r"$\mathbf{x'}_2/\Delta$")
            ax3229.grid()

            ax3329.plot(t[el:-el],xp[2,el:-el]/Delta,linest,color="tab:orange")
            ax3329.set_xlabel(r"$t$")
            ax3329.set_ylabel(r"$\mathbf{x'}_3/\Delta$")
            ax3329.grid()

            ax3119.legend()
            fig39.savefig('plots/lorenz-sgs-xp-del-normalized.png',dpi=300,format='png',transparent=True,bbox_inches='tight')



        ts = np.array([-2.5,2.5,5,10,12.5])
            
        for jj in range(len(ts)):

            indt = np.nonzero(np.round_(t,1) == ts[jj])
            Exptx = np.mean(xp[0,indt])
            Expty = np.mean(xp[1,indt])
            Exptz = np.mean(xp[2,indt])
            Efbartx = np.mean(fbar[0,indt])
            Efbarty = np.mean(fbar[1,indt])
            Efbartz = np.mean(fbar[2,indt])
            Esx = np.mean(s[0,indt])
            Esy = np.mean(s[1,indt])
            Esz = np.mean(s[2,indt])


            # plot x' vs Delta and fbar vs Delta and s vs Delta

            if Delta == 1: # trigger legend for one Delta to avoid printing multiple times
                ax3311.scatter(Delta,Exptx,c='C'+str(jj),label=r"$t$ = " + str(round(ts[jj],1)))
            else:
                ax3311.scatter(Delta,Exptx,c='C'+str(jj))

            ax3321.scatter(Delta,Expty,c='C'+str(jj))
            ax3331.scatter(Delta,Exptz,c='C'+str(jj))

            ax3312.scatter(Delta,Efbartx,c='C'+str(jj))
            ax3322.scatter(Delta,Efbarty,c='C'+str(jj))
            ax3332.scatter(Delta,Efbartz,c='C'+str(jj))

            ax33129.scatter(Delta,Esx,c='C'+str(jj))
            ax33229.scatter(Delta,Esy,c='C'+str(jj))
            ax33329.scatter(Delta,Esz,c='C'+str(jj))

        # plot ||x'||^2 vs Delta and ||fbar||^2 vs Delta and ||s||^2 vs Delta

        Np = len(xbar.T)

        xp_L2 = np.zeros((3))
        fbar_L2 = np.zeros((3))
        s_L2 = np.zeros((3))

        for ii in range(Np):

            xp_L2[0] += xp[0,ii]**2
            xp_L2[1] += xp[1,ii]**2
            xp_L2[2] += xp[1,ii]**2

            fbar_L2[0] += fbar[0,ii]**2
            fbar_L2[1] += fbar[1,ii]**2
            fbar_L2[2] += fbar[1,ii]**2

            s_L2[0] += s[0,ii]**2
            s_L2[1] += s[1,ii]**2
            s_L2[2] += s[1,ii]**2

        xp_L2 = np.sqrt(xp_L2/Np)
        fbar_L2 = np.sqrt(fbar_L2/Np)
        s_L2 = np.sqrt(s_L2/Np)

        ax41.scatter(Delta,xp_L2[0],c='C0')
        ax42.scatter(Delta,xp_L2[1],c='C1')
        ax43.scatter(Delta,xp_L2[2],c='C2')

        ax44.scatter(Delta,fbar_L2[0],c='C0')
        ax45.scatter(Delta,fbar_L2[1],c='C1')
        ax46.scatter(Delta,fbar_L2[2],c='C2')

        ax47.scatter(Delta,s_L2[0],c='C0')
        ax48.scatter(Delta,s_L2[1],c='C1')
        ax49.scatter(Delta,s_L2[2],c='C2')

        ax41.set_ylabel(r"$||{\mathbf{x'}_1}||$")
        ax41.set_xscale("log")
        ax41.set_yscale("log")
        ax42.set_ylabel(r"$||{\mathbf{x'}_2}||$")
        ax42.set_xscale("log")
        ax42.set_yscale("log")
        ax43.set_ylabel(r"$||{\mathbf{x'}_3}||$")
        ax43.set_xscale("log")
        ax43.set_yscale("log")
        ax43.set_xlabel(r"$\Delta$")
        
        ax44.set_ylabel(r"$||\overline{\mathbf{f}}_1||$")
        ax44.set_xscale("log")
        ax44.set_yscale("log")
        ax45.set_ylabel(r"$||\overline{\mathbf{f}}_2||$")
        ax45.set_xscale("log")
        ax45.set_yscale("log")
        ax46.set_ylabel(r"$||\overline{\mathbf{f}}_3||$")
        ax46.set_xscale("log")
        ax46.set_yscale("log")
        ax46.set_xlabel(r"$\Delta$")

        ax47.set_ylabel(r"$||{\mathbf{s}}_1||$")
        ax47.set_xscale("log")
        ax47.set_ylim(-1,1)
        ax48.set_ylabel(r"$||{\mathbf{s}}_2||$")
        ax48.set_xscale("log")
        ax48.set_yscale("log")
        ax49.set_ylabel(r"$||{\mathbf{s}}_3||$")
        ax49.set_xscale("log")
        ax49.set_yscale("log")
        ax49.set_xlabel(r"$\Delta$")

        ax41.grid()

        fig55.savefig('plots/lorenz-Delta-xp-fbar-s-L2-sqrt.png',dpi=300,format='png',transparent=True,bbox_inches='tight')


    ax3311.set_ylabel(r"${\mathbf{x'}_1}$")
    ax3321.set_ylabel(r"${\mathbf{x'}_2}$")
    ax3331.set_ylabel(r"${\mathbf{x'}_3}$")
    ax3331.set_xlabel(r"$\Delta$")
    
    ax3312.set_ylabel(r"$\overline{\mathbf{f}}_1$")
    ax3322.set_ylabel(r"$\overline{\mathbf{f}}_2$")
    ax3332.set_ylabel(r"$\overline{\mathbf{f}}_3$")
    ax3332.set_xlabel(r"$\Delta$")

    ax33129.set_ylabel(r"${\mathbf{s}}_1$")
    ax33129.set_ylim(-1,1)
    ax33229.set_ylabel(r"${\mathbf{s}}_2$")
    ax33329.set_ylabel(r"${\mathbf{s}}_3$")
    ax33329.set_xlabel(r"$\Delta$")

    ax3311.grid()
    ax3321.grid()
    ax3331.grid()
    ax3312.grid()
    ax3322.grid()
    ax3332.grid()
    ax33129.grid()
    ax33229.grid()
    ax33329.grid()

    if Delta == 1:
        ax3311.legend()

    fig33.savefig('plots/lorenz-Delta-xp-fbar-s.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

    #plt.show()












