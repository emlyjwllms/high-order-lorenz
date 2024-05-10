# plotting script for Lorenz attractor

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

porder = 2
h = 1/5
dt = 1/100

# lorenz =  np.load('data/lorenz_dt' + str(int(1/h)) + '.npz')
# x_FE = lorenz['x_FE']
# x_BE = lorenz['x_BE']
# t_h = lorenz['t']
# assert((np.round(t_h[1] - t_h[0],5) == h))
# Nh = len(t_h)

dg_lorenz = np.load('data/dg_lorenz_dt' + str(int(1/dt)) + '_p' + str(porder) + '.npz')

xh = dg_lorenz['xh']
t = dg_lorenz['t']
cs = dg_lorenz['cs']

N = len(t)
xi, w = scipy.special.roots_legendre(porder+1)
phi, dphi = plegendre(xi,porder)

Delta = h

# dictionary with quadrature parameters
quad_par = {'cs': cs, 'w': w, 'phi': phi}

# dictionary with jacobian parameters
jac_par = {'sigma': 10, 'r': 28, 'beta': 8/3}

xbar = filter('x', N, Delta, dt, quad_par, jac_par)


xp = xh - xbar

fx = f(xh,jac_par)

fbar = filter('f(x)', N, Delta, dt, quad_par, jac_par)

fxbar = f(xbar,jac_par)

s = fbar - fxbar

el = int(h/dt)

# lorenz_NN = np.load('data/diffusion-path-NL-h' + str(int(1/h)) + '.npz')
# xbar_NN = lorenz_NN['xbar_NN']

from scipy.spatial import KDTree

xbarr = xbar[:,el:-el].T

tree = KDTree(xbarr)
ind = sorted(tree.query_ball_point(xbarr[850,:], 4))
# print(ind)
print(len(ind))

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1,projection='3d')
ax.plot(*xbarr.T)

sx = np.zeros((len(ind)))
sy = np.zeros((len(ind)))
sz = np.zeros((len(ind)))

# for i in ind:
#     # ax.scatter(*xbarr[i,:].T, c='r')
#     sx[i] = s[0,i]
#     sy[i] = s[1,i]
#     sz[i] = s[2,i]

sr = s[:,el:-el]

ax.scatter(*xbarr[ind,:].T, c='r', depthshade=False)

ax.set_xlabel(r'$\overline{\mathbf{x}}$')
ax.set_ylabel(r'$\overline{\mathbf{y}}$')
ax.set_zlabel(r'$\overline{\mathbf{z}}$')
plt.show()

plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
plt.scatter(xbarr[ind,0],sr[0,ind])
plt.xlabel(r"$\overline{\mathbf{x}}$")
plt.ylabel(r"${\mathbf{s}}_{x}$")
plt.grid(True)

plt.subplot(3,3,2)
plt.scatter(xbarr[ind,1],sr[0,ind])
plt.xlabel(r"$\overline{\mathbf{y}}$")
plt.ylabel(r"${\mathbf{s}}_{x}$")
plt.grid(True)

plt.subplot(3,3,3)
plt.scatter(xbarr[ind,2],sr[0,ind])
plt.xlabel(r"$\overline{\mathbf{z}}$")
plt.ylabel(r"${\mathbf{s}}_{x}$")
plt.grid(True)

plt.subplot(3,3,4)
plt.scatter(xbarr[ind,0],sr[1,ind])
plt.xlabel(r"$\overline{\mathbf{x}}$")
plt.ylabel(r"${\mathbf{s}}_{y}$")
plt.grid(True)

plt.subplot(3,3,5)
plt.scatter(xbarr[ind,1],sr[1,ind])
plt.xlabel(r"$\overline{\mathbf{y}}$")
plt.ylabel(r"${\mathbf{s}}_{y}$")
plt.grid(True)

plt.subplot(3,3,6)
plt.scatter(xbarr[ind,2],sr[1,ind])
plt.xlabel(r"$\overline{\mathbf{z}}$")
plt.ylabel(r"${\mathbf{s}}_{y}$")
plt.grid(True)

plt.subplot(3,3,7)
plt.scatter(xbarr[ind,0],sr[2,ind])
plt.xlabel(r"$\overline{\mathbf{x}}$")
plt.ylabel(r"${\mathbf{s}}_{z}$")
plt.grid(True)

plt.subplot(3,3,8)
plt.scatter(xbarr[ind,1],sr[2,ind])
plt.xlabel(r"$\overline{\mathbf{y}}$")
plt.ylabel(r"${\mathbf{s}}_{z}$")
plt.grid(True)

plt.subplot(3,3,9)
plt.scatter(xbarr[ind,2],sr[2,ind])
plt.xlabel(r"$\overline{\mathbf{z}}$")
plt.ylabel(r"${\mathbf{s}}_{z}$")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
counts,bins = np.histogram(sr[0,ind])
plt.stairs(counts,bins)
plt.title(r"${\mathbf{s}}_{x}$")

plt.subplot(1,3,2)
counts,bins = np.histogram(sr[1,ind])
plt.stairs(counts,bins)
plt.title(r"${\mathbf{s}}_{y}$")

plt.subplot(1,3,3)
counts,bins = np.histogram(sr[2,ind])
plt.stairs(counts,bins)
plt.title(r"${\mathbf{s}}_{z}$")

plt.show()


# plot_3d({r"$\overline{\mathbf{x}}$": xbar[:,el:-el].T},r"$\overline{\mathbf{x}}$",r"$\overline{\mathbf{y}}$",r"$\overline{\mathbf{z}}$",None,save=False)


# plot_3(t[el:-el],{r"${\mathbf{x}}$": xh[:,el:-el].T,r"$\overline{\mathbf{x}}$": xbar[:,el:-el].T,r"${\mathbf{x}'}$": xp[:,el:-el].T},r"t",r"$x$",r"$y$",r"$z$",None,save=False)
# plot_3(t[el:-el],{r"${\mathbf{s}}$": s[:,el:-el].T},r"t",r"$x$",r"$y$",r"$z$",None,save=False)




if 0:
    diffusivity_type = "spd"

    lorenz_NN_BEM = np.load('data/diffusion-path-NL-BEM-' + diffusivity_type + '-h' + str(int(1/h)) + '.npz')
    xbar_NN_BEM = lorenz_NN_BEM['xbar_NN']

    # interpolate for plotting

    # xbar_NN_t = interp_3(t,t_h,xbar_NN).T
    x_FE_t = interp_3(t,t_h,x_FE).T
    x_BE_t = interp_3(t,t_h,x_BE).T
    xbar_NN_BEM_t = interp_3(t,t_h,xbar_NN_BEM).T


    # plot trajectory paths
    plot_3(t[el:-el],{r"$\overline{\mathbf{x}}_{NN}$": xbar_NN_BEM_t[el:-el,:], r"${\mathbf{x}}_{BE}$": x_BE_t[el:-el,:], r"$\overline{\mathbf{x}}_{DG}$": xbar[:,el:-el].T},r"t",r"$x$",r"$y$",r"$z$",'plots/paths-NL-BEM-' + diffusivity_type + '-h' + str(int(1/h)) + '.png',save=True)

    # plot 3D
    plot_3d({r"$\overline{\mathbf{x}}_{NN}$": xbar_NN_BEM, r"${\mathbf{x}}_{BE}$": x_BE, r"$\overline{\mathbf{x}}_{DG}$": xbar[:,el:-el].T},r"$x$",r"$y$",r"$z$",'plots/nn-3d-coarse-nl-bem-' + diffusivity_type + '-h' + str(int(1/h)) + '.png',save=True)

    # time mean

    xbar_mean = np.mean(xbar[:,el:-el],1)
    xbar_mean_cumulative = cumulative_mean(t,xbar,el)

    # xbar_NN_mean = np.mean((xbar_NN[el:-el,:]).transpose(),1)
    # xbar_NN_mean_cumulative = cumulative_mean(t_h,xbar_NN.T,0).T

    xbar_NN_BEM_mean = np.mean((xbar_NN_BEM[el:-el,:]).transpose(),1)
    xbar_NN_BEM_mean_cumulative = cumulative_mean(t_h,xbar_NN_BEM.T,0).T

    x_FE_mean = np.mean((x_FE[el:-el,:]).transpose(),1)
    x_FE_mean_cumulative = cumulative_mean(t_h,x_FE.T,0).T

    x_BE_mean = np.mean((x_BE[el:-el,:]).transpose(),1)
    x_BE_mean_cumulative = cumulative_mean(t_h,x_BE.T,0).T


    # interpolate for plotting

    # xbar_NN_mean_cumulative_t = interp_3(t,t_h,xbar_NN_mean_cumulative).T
    xbar_NN_BEM_mean_cumulative_t = interp_3(t,t_h,xbar_NN_BEM_mean_cumulative).T
    x_FE_mean_cumulative_t = interp_3(t,t_h,x_FE_mean_cumulative).T
    x_BE_mean_cumulative_t = interp_3(t,t_h,x_BE_mean_cumulative).T


    plot_3(t,{r"$\overline{\mathbf{x}}_{NN}^{avg}$": np.ones((N,3))*xbar_NN_BEM_mean.T, r"${\mathbf{x}}_{BE}^{avg}$": np.ones((N,3))*x_BE_mean.T, r"$\overline{\mathbf{x}}_{DG}^{avg}$": np.ones((N,3))*xbar_mean.T},r"t",r"$x$",r"$y$",r"$z$",'plots/paths-NL-BEM-mean-' + diffusivity_type + '-h' + str(int(1/h)) + '.png',save=True)

    plot_3(t[el:-el],{r"$\overline{\mathbf{x}}_{NN}^{avg,t}$": xbar_NN_BEM_mean_cumulative_t[el:-el,:], r"${\mathbf{x}}_{BE}^{avg,t}$": x_BE_mean_cumulative_t[el:-el,:], r"$\overline{\mathbf{x}}_{DG}^{avg,t}$": xbar_mean_cumulative.T},r"t",r"$x$",r"$y$",r"$z$",'plots/paths-NL-BEM-mean-cumulative-' + diffusivity_type + '-h' + str(int(1/h)) + '.png',save=True)

