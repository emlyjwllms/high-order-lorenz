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
    #print(xv)
    x = xv[0]
    y = xv[1]
    z = xv[2]
    xdot = sigma*(y-x)
    ydot = x*(r-z) - y
    zdot = x*y - beta*z
    return np.array([xdot,ydot,zdot]).T

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

    return s, fbar, fx, fxbar, x_DG, xbar, Delta, dt, t



if __name__ == "__main__":

    # params
    sigma = 10
    r = 28
    beta = 8/3

    plots = True

    widths = np.array([1,0.5,0.25,0.1,0.05,0.025])
    #widths = np.arange(0.01,1+0.005,0.005)
    filter_data = {}

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(3,1,1)
    ax22 = fig.add_subplot(3,1,2)
    ax33 = fig.add_subplot(3,1,3)

    for Del in widths:
        s, fbar, fx, fxbar, x_DG, xbar, Delta, dt, t = lorenz_sgs(Del)
        filter_data[Del] = s, fbar, fx, fxbar, x_DG, xbar, Delta, dt, t

        el = int(Delta/dt)+1

        if (round(Del,3) == 0.250 or round(Del,4) == 0.0250 or round(Del,4) == 0.500 or round(Del,4) == 1) and plots == True:
            Delta = round(Delta,3)
            fig1 = plt.figure(figsize=(20,10))
            ax1 = fig1.add_subplot(3,3,1)
            ax1.plot(t,x_DG[0,:],label=r"$\mathbf{x}$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax1.plot(t[int(el):-int(el)],xbar[int(el):-int(el),0],label=r"$\overline{\mathbf{x}}$")
            ax1.plot(t[int(el):-int(el)],x_DG[0,int(el):-int(el)] - xbar[int(el):-int(el),0],label=r"${\mathbf{x'}}$")
            ax1.set_ylabel(r"$x$")
            ax1.legend(loc='upper left')
            ax1.set_title(r"$\Delta = $" + str(Delta) + ", filtered states")
            #ax1.set_title(r"$\Delta t = $" + str(round(dt,3)) + r", $\Delta = $" + str(Delta))
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,4)
            ax2.plot(t,x_DG[1,:],label=r"$\mathbf{x}$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax2.plot(t[int(el):-int(el)],xbar[int(el):-int(el),1],label=r"$\overline{\mathbf{x}}$")
            ax2.plot(t[int(el):-int(el)],x_DG[1,int(el):-int(el)] - xbar[int(el):-int(el),1],label=r"${\mathbf{x'}}$")
            ax2.set_ylabel(r"$y$")
            #ax2.legend()
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,7)
            ax3.plot(t,x_DG[2,:],label=r"$\mathbf{x}$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax3.plot(t[int(el):-int(el)],xbar[int(el):-int(el),2],label=r"$\overline{\mathbf{x}}$")
            ax3.plot(t[int(el):-int(el)],x_DG[2,int(el):-int(el)] - xbar[int(el):-int(el),2],label=r"${\mathbf{x'}}$")
            ax3.set_ylabel(r"$z$")
            ax3.set_xlabel(r"$t$")
            #ax3.legend()
            ax3.grid()

            #plt.savefig('lorenz-filter-states.png',dpi=300,format='png',transparent=True)

            #plt.show()


            ax1 = fig1.add_subplot(3,3,2)
            ax1.plot(t,fx[:,0],label=r"$\mathbf{f}(\mathbf{x})$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax1.plot(t[int(el):-int(el)],fxbar[int(el):-int(el),0],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax1.plot(t[int(el):-int(el)],fbar[int(el):-int(el),0],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax1.set_ylabel(r"$\mathbf{f}_1$")
            ax1.legend(loc='upper left')
            ax1.set_title("filtered dynamics")
            ax1.grid()

            ax2 = fig1.add_subplot(3,3,5)
            ax2.plot(t,fx[:,1],label=r"$\mathbf{f}(\mathbf{x})$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax2.plot(t[int(el):-int(el)],fxbar[int(el):-int(el),1],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax2.plot(t[int(el):-int(el)],fbar[int(el):-int(el),1],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax2.set_ylabel(r"$\mathbf{f}_2$")
            #ax2.legend()
            ax2.grid()

            ax3 = fig1.add_subplot(3,3,8)
            ax3.plot(t,fx[:,2],label=r"$\mathbf{f}(\mathbf{x})$")
            #ax1.vlines(Tp,-15,15,colors='k',linestyles='dashed',alpha=0.4)
            ax3.plot(t[int(el):-int(el)],fxbar[int(el):-int(el),2],label=r"$\mathbf{f}(\overline{\mathbf{x}})$")
            ax3.plot(t[int(el):-int(el)],fbar[int(el):-int(el),2],label=r"$\overline{\mathbf{f}(\mathbf{x})}$")
            ax3.set_ylabel(r"$\mathbf{f}_3$")
            ax3.set_xlabel(r"$t$")
            #ax3.legend()
            ax3.grid()

            #plt.savefig('lorenz-filterv2.png',dpi=300,format='png',transparent=True)

            #plt.show()

            ax1 = fig1.add_subplot(3,3,3)
            ax1.plot(t[int(el):-int(el)],s[int(el):-int(el),0])
            ax1.set_ylabel(r"$\mathbf{s}_1$")
            ax1.set_title("subgrid dynamics")
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
            #ax3.legend()
            ax3.grid()

            plt.savefig('plots/lorenz-filter-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

            if (round(Del,4) == 0.500):

                # phase plots, xbar components with colored s's
                fig55 = plt.figure(figsize=(7,7))
                ax15 = fig55.add_subplot(1,1,1,projection='3d')
                p = ax15.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=s[int(el):-int(el),0],s=10,cmap=mpl.cm.Blues,label=r"$\mathbf{s}_1$")
                ax15.set_xlabel(r"$\overline{\mathbf{x}}_1$")
                ax15.set_ylabel(r"$\overline{\mathbf{x}}_2$")
                ax15.set_zlabel(r"$\overline{\mathbf{x}}_3$")
                ax15.set_title(r"$\Delta$ = " + str(round(Del,3)))
                ax15.legend()
                fig55.colorbar(p,shrink=0.5,pad=0.1)

                plt.savefig('plots/lorenz-sgs1-s-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

                #plt.show()

                fig555 = plt.figure(figsize=(7,7))
                ax15 = fig555.add_subplot(1,1,1,projection='3d')
                p = ax15.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=s[int(el):-int(el),1],s=10,cmap=mpl.cm.Greens,label=r"$\mathbf{s}_2$")
                ax15.set_xlabel(r"$\overline{\mathbf{x}}_1$")
                ax15.set_ylabel(r"$\overline{\mathbf{x}}_2$")
                ax15.set_zlabel(r"$\overline{\mathbf{x}}_3$")
                ax15.set_title(r"$\Delta$ = " + str(round(Del,3)))
                ax15.legend()
                fig555.colorbar(p,shrink=0.5,pad=0.1)

                plt.savefig('plots/lorenz-sgs2-s-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

                #plt.show()

                fig5555 = plt.figure(figsize=(7,7))
                ax15 = fig5555.add_subplot(1,1,1,projection='3d')
                p = ax15.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=s[int(el):-int(el),2],s=5,cmap=mpl.cm.Oranges,label=r"$\mathbf{s}_3$")
                ax15.set_xlabel(r"$\overline{\mathbf{x}}_1$")
                ax15.set_ylabel(r"$\overline{\mathbf{x}}_2$")
                ax15.set_zlabel(r"$\overline{\mathbf{x}}_3$")
                ax15.set_title(r"$\Delta$ = " + str(round(Del,3)))
                ax15.legend()
                fig5555.colorbar(p,shrink=0.5,pad=0.1)

                plt.savefig('plots/lorenz-sgs3-s-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

                #plt.show()

                # phase plots, xbar components with colored x''s
                fig56 = plt.figure(figsize=(7,7))
                ax15 = fig56.add_subplot(1,1,1,projection='3d')
                p = ax15.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=x_DG[0,int(el):-int(el)] - xbar[int(el):-int(el),0],s=10,cmap=mpl.cm.Blues,label=r"$\mathbf{x'}_1$")
                ax15.set_xlabel(r"$\overline{\mathbf{x}}_1$")
                ax15.set_ylabel(r"$\overline{\mathbf{x}}_2$")
                ax15.set_zlabel(r"$\overline{\mathbf{x}}_3$")
                ax15.set_title(r"$\Delta$ = " + str(round(Del,3)))
                ax15.legend()
                fig55.colorbar(p,shrink=0.5,pad=0.1)

                plt.savefig('plots/lorenz-sgs1-xp-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

                #plt.show()

                fig556 = plt.figure(figsize=(7,7))
                ax15 = fig556.add_subplot(1,1,1,projection='3d')
                p = ax15.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=x_DG[1,int(el):-int(el)] - xbar[int(el):-int(el),1],s=10,cmap=mpl.cm.Greens,label=r"$\mathbf{x'}_2$")
                ax15.set_xlabel(r"$\overline{\mathbf{x}}_1$")
                ax15.set_ylabel(r"$\overline{\mathbf{x}}_2$")
                ax15.set_zlabel(r"$\overline{\mathbf{x}}_3$")
                ax15.set_title(r"$\Delta$ = " + str(round(Del,3)))
                ax15.legend()
                fig55.colorbar(p,shrink=0.5,pad=0.1)

                plt.savefig('plots/lorenz-sgs2-xp-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

                #plt.show()

                fig5556 = plt.figure(figsize=(7,7))
                ax15 = fig5556.add_subplot(1,1,1,projection='3d')
                p = ax15.scatter(xbar[int(el):-int(el),0],xbar[int(el):-int(el),1],xbar[int(el):-int(el),2],c=x_DG[2,int(el):-int(el)] - xbar[int(el):-int(el),2],s=10,cmap=mpl.cm.Oranges,label=r"$\mathbf{x'}_3$")
                ax15.set_xlabel(r"$\overline{\mathbf{x}}_1$")
                ax15.set_ylabel(r"$\overline{\mathbf{x}}_2$")
                ax15.set_zlabel(r"$\overline{\mathbf{x}}_3$")
                ax15.set_title(r"$\Delta$ = " + str(round(Del,3)))
                ax15.legend()
                fig55.colorbar(p,shrink=0.5,pad=0.1)

                plt.savefig('plots/lorenz-sgs3-xp-' + str(Delta) + '.png',dpi=300,format='png',transparent=True,bbox_inches='tight')

                #plt.show()

        if np.isclose(round(Del,3) % 0.05,0):
            
            print(round(Del,3))

            ax.plot(t[int(el):-int(el)],s[int(el):-int(el),0],color="tab:blue", alpha=Del/0.8)
            #ax.set_xlabel(r"$t$")
            ax.set_ylabel(r"$\mathbf{s}_1$")
            ax.set_yticks(np.array([-1,-0.5,0,0.5,1]))
            ax.grid()
            #ax.legend()

            ax22.plot(t[int(el):-int(el)],s[int(el):-int(el),1],color="tab:blue", alpha=Del/0.8)
            #ax22.set_xlabel(r"$t$")
            ax22.set_ylabel(r"$\mathbf{s}_2$")
            ax22.grid()
            #ax22.legend()

            ax33.plot(t[int(el):-int(el)],s[int(el):-int(el),2],color="tab:blue", alpha=Del/0.8)
            ax33.set_xlabel(r"$t$")
            ax33.set_ylabel(r"$\mathbf{s}_3$")
            ax33.grid()
            #ax33.legend()

            #plt.savefig('plots/lorenz-sgs-del.png',dpi=300,format='png',transparent=True,bbox_inches='tight')


    df = pd.DataFrame(filter_data,index=['s','fbar','fx','fxbar','x_DG','xbar','Delta','dt','t']).T

    df.to_csv('lorenz_filter_sgs.csv')












