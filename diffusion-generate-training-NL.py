# Lorenz attractor nonlinear solution - training data generation

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from lorenz_functions import *

porder = 2
h = 1/1000
dt = 1/10000

lorenz = np.load('data/dg_lorenz_dt' + str(int(1/dt)) + '_p' + str(porder) + '.npz')

xh = lorenz['xh'].T
cs = lorenz['cs']
t = lorenz['t']
assert(dt == np.round(t[1]-t[0],5))
dt = np.round(t[1]-t[0],5)

N = len(t)
xi, w = scipy.special.roots_legendre(porder+1)
phi, dphi = plegendre(xi,porder)

Delta = h
K = int(h/dt)

# dictionary with quadrature parameters
quad_par = {'cs': cs, 'w': w, 'phi': phi}


# dictionary with jacobian parameters
jac_par = {'sigma': 10, 'r': 28, 'beta': 8/3}

xbar = filter('x',N,Delta,dt,quad_par,jac_par)

xp = xh.T - xbar

fx = f(xh,jac_par)

fbar = filter('f(x)',N,Delta,dt,quad_par,jac_par)

fxbar = f(xbar,jac_par)

s = fbar - fxbar

print('loop through elements')
for n in range(0,N-1-K):

    x_bar_n_0 = xbar[:,n]
    x_bar_n_h = xbar[:,n+K]

    if n == 0:
        x_bar_0 = x_bar_n_0
        x_bar_h = x_bar_n_h

    else:
        x_bar_0 = np.vstack((x_bar_0,x_bar_n_0))
        x_bar_h = np.vstack((x_bar_h,x_bar_n_h))

print('save data')
np.savez('data/diffusion-training-data-NL-h' + str(int(1/h)), x_bar_0=x_bar_0, x_bar_h=x_bar_h, h=h)