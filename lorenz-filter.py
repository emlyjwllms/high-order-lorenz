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

def J(xv):
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

def H(xv):
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

def H_alt(xv):
    #H = J(grad(f)).T
    gradf = np.zeros(3)
    gradf[0] = -sigma
    gradf[1] = -1
    gradf[2] = -beta
    H = J(gradf).T
    return H

def filter(t,x_IN,Delta):
    
    dt = t[1]-t[0]
    Ne = len(x_IN)
    el = int(Delta/dt)+1

    Xs = np.zeros(Ne)
    Ys = np.zeros(Ne)
    Zs = np.zeros(Ne)

    # apply filter
    for n in np.arange(el,Ne-el):
        ind = np.arange(int(n-el/2),int(n+el/2))
        I_x = x_IN[ind,0]
        I_y = x_IN[ind,1]
        I_z = x_IN[ind,2]
        E_I_x = np.mean(I_x)
        E_I_y = np.mean(I_y)
        E_I_z = np.mean(I_z)
        Xs[n] = E_I_x
        Ys[n] = E_I_y
        Zs[n] = E_I_z

    x_OUT = np.array([Xs,Ys,Zs]).T

    return x_OUT

def lorenz_sgs(Delta):

    lorenz = np.load('dg_lorenz_dt1000_p3.npz')
    x_DG = lorenz['xh']
    t = lorenz['t']
    dt = t[1]-t[0]

    xbar = filter(t,x_DG.T,Delta)
    fbar = filter(t,f(x_DG),Delta)
    fxbar = f(xbar.T)
    s = fbar - fxbar
    fx = f(x_DG)
    xp = x_DG.T - xbar

    return s, fbar, fx, fxbar, x_DG, xbar, xp, Delta, dt, t


if __name__ == "__main__":

    # params
    sigma = 10
    r = 28
    beta = 8/3

    plots = True

    #widths = np.arange(0.01,1+0.005,0.005)
    widths = np.array([0.025,0.05,0.1,0.25,0.5,0.75,1])
    filter_data = {}

    fig3 = plt.figure(figsize=(20,10))
    ax311 = fig3.add_subplot(3,2,1)
    ax321 = fig3.add_subplot(3,2,3)
    ax331 = fig3.add_subplot(3,2,5)
    ax312 = fig3.add_subplot(3,2,2)
    ax322 = fig3.add_subplot(3,2,4)
    ax332 = fig3.add_subplot(3,2,6)

    for Del in widths:

        s, fbar, fx, fxbar, x_DG, xbar, xp, Delta, dt, t = lorenz_sgs(Del)
        filter_data[Del] = s, fbar, fx, fxbar, x_DG, xbar, xp, Delta, dt, t

        jac = J(xbar)
        hes = H(xbar)

        s_model = 0.5*xp.T*(hes + Delta**2/12 * jac.T * hes * jac)*xp

        # offset to avoid boundary effects
        el = int(Delta/dt)+1

        Delta = round(Delta,3)

        if (Delta == 0.0250 or Delta == 0.25 or Delta == 0.5 or Delta == 1) and plots == True:
            
            # plot filtered states: x, xbar, x'

            fig1 = plt.figure(figsize=(20,10))
            ax1 = fig1.add_subplot(3,3,1)
            ax1.plot(t,x_DG[0,:],label=r"$\mathbf{x}$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax1.plot(t[int(el):-int(el)],xbar[int(el):-int(el),0],label=r"$\overline{\mathbf{x}}$")
            ax1.plot(t[int(el):-int(el)],xp[int(el):-int(el),0],label=r"${\mathbf{x'}}$")
            ax1.set_ylabel(r"$x$")
            ax1.legend(loc='upper left')
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,4)
            ax2.plot(t,x_DG[1,:],label=r"$\mathbf{x}$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax2.plot(t[int(el):-int(el)],xbar[int(el):-int(el),1],label=r"$\overline{\mathbf{x}}$")
            ax2.plot(t[int(el):-int(el)],xp[int(el):-int(el),1],label=r"${\mathbf{x'}}$")
            ax2.set_ylabel(r"$y$")
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,7)
            ax3.plot(t,x_DG[2,:],label=r"$\mathbf{x}$")
            ax3.plot(t[int(el):-int(el)],xbar[int(el):-int(el),2],label=r"$\overline{\mathbf{x}}$")
            ax3.plot(t[int(el):-int(el)],xp[int(el):-int(el),2],label=r"${\mathbf{x'}}$")
            ax3.set_ylabel(r"$z$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            # plot filtered dynamics: f(x), f(xbar), fbar(x)

            ax1 = fig1.add_subplot(3,3,2)
            ax1.plot(t,fx[:,0],label=r"$\mathbf{f}(\mathbf{x})$")
            ax1.plot(t[int(el):-int(el)],fxbar[int(el):-int(el),0],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax1.plot(t[int(el):-int(el)],fbar[int(el):-int(el),0],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax1.set_ylabel(r"$\mathbf{f}_1$")
            ax1.legend(loc='upper left')
            ax1.set_title(r"$\Delta = $" + str(Delta))
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,5)
            ax2.plot(t,fx[:,1],label=r"$\mathbf{f}(\mathbf{x})$")
            ax2.plot(t[int(el):-int(el)],fxbar[int(el):-int(el),1],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax2.plot(t[int(el):-int(el)],fbar[int(el):-int(el),1],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax2.set_ylabel(r"$\mathbf{f}_2$")
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,8)
            ax3.plot(t,fx[:,2],label=r"$\mathbf{f}(\mathbf{x})$")
            ax3.plot(t[int(el):-int(el)],fxbar[int(el):-int(el),2],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax3.plot(t[int(el):-int(el)],fbar[int(el):-int(el),2],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax3.set_ylabel(r"$\mathbf{f}_3$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            # plot subgrid dynamics: s

            ax1 = fig1.add_subplot(3,3,3)
            ax1.plot(t[int(el):-int(el)],s[int(el):-int(el),0])
            ax1.set_ylabel(r"$\mathbf{s}_1$")
            #ax1.set_title("subgrid dynamics")
            ax1.set_yticks(np.array([-1,-0.5,0,0.5,1]))
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,6)
            ax2.plot(t[int(el):-int(el)],s[int(el):-int(el),1])
            ax2.set_ylabel(r"$\mathbf{s}_2$")
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,9)
            ax3.plot(t[int(el):-int(el)],s[int(el):-int(el),2])
            ax3.set_ylabel(r"$\mathbf{s}_3$")
            ax3.set_xlabel(r"$t$")
            ax3.grid()

            plt.savefig('plots/lorenz-filter-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')


            # phase plots for subgrid and perturbation dynamics

            # subgrid s plots
            fig2 = plt.figure(figsize=(10,8))
            ax11 = fig2.add_subplot(3,2,1,projection='3d')
            p = ax11.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=s[int(el):-int(el),0],s=10,cmap=mpl.cm.Blues)
            ax11.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax11.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax11.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            ax11.set_title(r"$\Delta$ = " + str(round(Del,3)))
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax11,label=r"$\mathbf{s}_1$")

            ax21 = fig2.add_subplot(3,2,3,projection='3d')
            p = ax21.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=s[int(el):-int(el),1],s=10,cmap=mpl.cm.Greens)
            ax21.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax21.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax21.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax21,label=r"$\mathbf{s}_2$")

            ax31 = fig2.add_subplot(3,2,5,projection='3d')
            p = ax31.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=s[int(el):-int(el),2],s=5,cmap=mpl.cm.Oranges)
            ax31.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax31.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax31.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax31,label=r"$\mathbf{s}_3$")


            # perturbation x' plots
            ax12 = fig2.add_subplot(3,2,2,projection='3d')
            p = ax12.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=xp[int(el):-int(el),0],s=10,cmap=mpl.cm.Blues)
            ax12.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax12.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax12.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax12,label=r"$\mathbf{x'}_1$")

            ax22 = fig2.add_subplot(3,2,4,projection='3d')
            p = ax22.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=xp[int(el):-int(el),1],s=10,cmap=mpl.cm.Greens)
            ax22.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax22.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax22.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax22,label=r"$\mathbf{x'}_2$")

            ax32 = fig2.add_subplot(3,2,6,projection='3d')
            p = ax32.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=xp[int(el):-int(el),2],s=10,cmap=mpl.cm.Oranges)
            ax32.set_xlabel(r"$\overline{\mathbf{x}}_1$")
            ax32.set_ylabel(r"$\overline{\mathbf{x}}_2$")
            ax32.set_zlabel(r"$\overline{\mathbf{x}}_3$")
            fig2.colorbar(p,shrink=0.5,pad=0.3,ax=ax32,label=r"$\mathbf{x'}_3$")

            plt.savefig('plots/lorenz-sgs-xp-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

            #plt.show()

        print(Delta)
        ax311.plot(t[int(el):-int(el)],s[int(el):-int(el),0],color="tab:blue", alpha=Delta)
        ax311.set_ylabel(r"$\mathbf{s}_1$")
        ax311.set_yticks(np.array([-1,-0.5,0,0.5,1]))
        ax311.grid()

        ax321.plot(t[int(el):-int(el)],s[int(el):-int(el),1],color="tab:green", alpha=Delta)
        ax321.set_ylabel(r"$\mathbf{s}_2$")
        ax321.grid()

        ax331.plot(t[int(el):-int(el)],s[int(el):-int(el),2],color="tab:orange", alpha=Delta)
        ax331.set_xlabel(r"$t$")
        ax331.set_ylabel(r"$\mathbf{s}_3$")
        ax331.grid()

        ax312.plot(t[int(el):-int(el)],x_DG[0,int(el):-int(el)] - xbar[int(el):-int(el),0],color="tab:blue", alpha=Delta)
        ax312.set_ylabel(r"$\mathbf{x'}_1$")
        ax312.grid()

        ax322.plot(t[int(el):-int(el)],x_DG[1,int(el):-int(el)] - xbar[int(el):-int(el),1],color="tab:green", alpha=Delta)
        ax322.set_ylabel(r"$\mathbf{x'}_2$")
        ax322.grid()

        ax332.plot(t[int(el):-int(el)],x_DG[2,int(el):-int(el)] - xbar[int(el):-int(el),2],color="tab:orange", alpha=Delta)
        ax332.set_xlabel(r"$t$")
        ax332.set_ylabel(r"$\mathbf{x'}_3$")
        ax332.grid()

        
        fig3.savefig('plots/lorenz-sgs-xp-del.png',dpi=300,format='png',transparent=True,bbox_inches='tight')


    df = pd.DataFrame(filter_data,index=['s','fbar','fx','fxbar','x_DG','xbar','xp','Delta','dt','t']).T

    df.to_csv('lorenz_filter_sgs.csv')












