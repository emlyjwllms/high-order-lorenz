# OLD

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np

lorenz = np.load('stochastic_lorenz_dt100.npz')
E_x_EM = lorenz['E_x_EM']
E_x_MIL = lorenz['E_x_MIL']
E_x_RK = lorenz['E_x_RK']
t = lorenz['t']

# dg_lorenz_0 = np.load('dg_lorenz_dt1000_p0.npz')
# xh_0 = dg_lorenz_0['xh']

dg_lorenz = np.load('dg_stochastic_lorenz_dt100_p3.npz')
E_xh = dg_lorenz['E_xh']

fig1 = plt.figure(figsize=(7,10))
ax1 = fig1.add_subplot(3,1,1)
ax1.plot(t,E_x_EM[:,0].T,label="Euler-Maruyama")
ax1.plot(t,E_x_MIL[:,0].T,label="Milstein")
ax1.plot(t,E_x_RK[:,0].T,label="RK")
ax1.set_title("x")
ax1.legend()
ax1.grid()

ax2 = fig1.add_subplot(3,1,2)
ax2.plot(t,E_x_EM[:,1],label="Euler-Maruyama")
ax2.plot(t,E_x_MIL[:,1],label="Milstein")
ax2.plot(t,E_x_RK[:,1],label="RK")
ax2.set_title("y")
ax2.legend()
ax2.grid()

ax3 = fig1.add_subplot(3,1,3)
ax3.plot(t,E_x_EM[:,2],label="Euler-Maruyama")
ax3.plot(t,E_x_MIL[:,2],label="Milstein")
ax3.plot(t,E_x_RK[:,2],label="RK")
ax3.set_title("z")
ax3.legend()
ax3.grid()

plt.savefig('stochastic-lorenz-em-mil-rk.png',dpi=300,format='png')
plt.show()

fig2 = plt.figure(figsize=(12,4))
ax1 = fig2.add_subplot(1,3,1)
ax1.plot(t,E_xh[0,:].T,zorder=2)#,label="DG")
ax1.plot(t,E_x_RK[:,0].T,zorder=1)#,label="RK")
ax1.set_ylabel("x")
ax1.set_xlabel("t")
#ax1.legend()
ax1.grid()

ax2 = fig2.add_subplot(1,3,2)
ax2.plot(t,E_xh[1,:].T,zorder=2)#,label="DG")
ax2.plot(t,E_x_RK[:,1],zorder=1)#,label="RK")
ax2.set_ylabel("y")
ax2.set_xlabel("t")
#ax2.legend()
ax2.grid()

ax3 = fig2.add_subplot(1,3,3)
ax3.plot(t,E_xh[2,:].T,zorder=2,label="DG")
ax3.plot(t,E_x_RK[:,2],zorder=1,label="RK")
ax3.set_ylabel("z")
ax3.set_xlabel("t")
ax3.legend()
ax3.grid()

plt.savefig('stochastic-lorenz-dg-rk.png',dpi=300,format='png')
plt.show()












