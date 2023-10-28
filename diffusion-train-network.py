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
x_ns = training_data['x_n']
x_np1s = training_data['x_np1']
dts = training_data['dt_train']
alpha_train = training_data['alpha_train']
lambda_max = 0.95
mu_train = alpha_train*np.sqrt(2*lambda_max)

x_n_train = x_ns[::2,:]
x_np1_train = x_np1s[::2,:]
dt_train = dts[::2]

N = np.shape(x_n_train)[0]

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

# we won't need the above function if we use a built-in NN code
# also, there are python ML packages that can split the data into training and testing sets, maybe consider using that instead of the way I split it up?

# Compute training accuracy


# Compute testing accuracy of the predictions of your model

x_n_test = x_ns[1::2,:]
x_np1_test = x_np1s[1::2,:]
dt_test = dts[1::2]

N = np.shape(x_n_test)[0]










