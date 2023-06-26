# plotting script for Lorenz attractor

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np

lorenz = np.load('lorenz_dt1000.npz')
x_FE = lorenz['x_FE']
x_BE = lorenz['x_BE']
x_RK = lorenz['x_RK']
t = lorenz['t']

dg_lorenz_0 = np.load('dg_lorenz_dt1000_p0.npz')
xh_0 = dg_lorenz_0['xh']

dg_lorenz = np.load('dg_lorenz_dt1000_p3.npz')
xh = dg_lorenz['xh']

fig1 = plt.figure(figsize=(7,10))
ax1 = fig1.add_subplot(3,1,1)
ax1.plot(t,x_FE[:,0],label="FE")
ax1.plot(t,x_BE[:,0],label="BE")
ax1.plot(t,x_RK[:,0],label="RK")
ax1.set_title("x")
ax1.legend()
ax1.grid()

ax2 = fig1.add_subplot(3,1,2)
ax2.plot(t,x_FE[:,1],label="FE")
ax2.plot(t,x_BE[:,1],label="BE")
ax2.plot(t,x_RK[:,1],label="RK")
ax2.set_title("y")
ax2.legend()
ax2.grid()

ax3 = fig1.add_subplot(3,1,3)
ax3.plot(t,x_FE[:,2],label="FE")
ax3.plot(t,x_BE[:,2],label="BE")
ax3.plot(t,x_RK[:,2],label="RK")
ax3.set_title("z")
ax3.legend()
ax3.grid()

plt.savefig('lorenz-fe-be-rk.png',dpi=300,format='png')
plt.show()

fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1.plot(*x_RK.T,label="RK")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.legend()

plt.savefig('rk-lorenz.png',dpi=300,format='png')
plt.show()

fig1 = plt.figure(figsize=(7,10))
ax1 = fig1.add_subplot(3,1,1)
ax1.plot(t,xh_0[0,:],label="DG, p = 0")
ax1.plot(t,x_BE[:,0],'k--',label="BE")
ax1.set_title("x")
ax1.legend()
ax1.grid()

ax2 = fig1.add_subplot(3,1,2)
ax2.plot(t,xh_0[1,:],label="DG, p = 0")
ax2.plot(t,x_BE[:,1],'k--',label="BE")
ax2.set_title("y")
ax2.legend()
ax2.grid()

ax3 = fig1.add_subplot(3,1,3)
ax3.plot(t,xh_0[2,:],label="DG, p = 0")
ax3.plot(t,x_BE[:,2],'k--',label="BE")
ax3.set_title("z")
ax3.legend()
ax3.grid()

plt.savefig('dg-be-lorenz.png',dpi=300,format='png')
plt.show()

plt.plot(t,xh[0,:],label="x")
plt.plot(t,xh[1,:],label="y")
plt.plot(t,xh[2,:],label="z")
plt.xlabel("t")
plt.legend()
plt.grid()
plt.savefig('dg-lorenz.png',dpi=300,format='png')
plt.show()

fig = plt.figure(figsize=(7,7))
ax1 = fig.add_subplot(1,1,1,projection='3d')
ax1.plot(*xh,label="DG")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.legend()

plt.savefig('dg-lorenz-3d.png',dpi=300,format='png')
plt.show()