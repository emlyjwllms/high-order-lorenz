# train SDE diffusion

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm

training_data = np.load('diffusion-training-data.npz')
x_tilde_n = training_data['x_tilde_n']
x_tilde_np1 = training_data['x_tilde_np1']

print(x_tilde_n.shape)

x_tilde_n_train = x_tilde_n[::2,:]
x_tilde_np1_train = x_tilde_np1[::2,:]

porder = 2

lorenz = np.load('dg_lorenz_dt100_p' + str(porder) + '.npz')

x_n = lorenz['xh'].T
cs = lorenz['cs']
t = lorenz['t']
dt = t[1]-t[0]
 

# # regularization
# lamb = 0
# order = 2

# # Carry out training
# def loss(w):
#    L = 0
#    for i in range(N):
#       L +=  # + lamb*norm(w,ord=order)
#    L = -1/N * L
#    return L

# w0 = np.ones(N)
# w = minimize(loss,w0).x

# there are python ML packages that can split the data into training and testing sets, maybe consider using that instead of the way I split it up?

# Compute training accuracy


# Compute testing accuracy of the predictions of your model

x_tilde_n_test = x_tilde_n[1::2,:]
x_tilde_np1_test = x_tilde_np1[1::2,:]










