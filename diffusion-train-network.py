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
x_EMs_n_data = training_data['x_EMs_n']
x_EMs_np1_data = training_data['x_EMs_np1']
dt_train = training_data['dt_train']

x_EMs_n_train = x_EMs_n_data[::2,:]
x_EMs_np1_train = x_EMs_np1_data[::2,:]
dt_train = dt_train[::2,:]

N = np.shape(x_EMs_n_train)[0]

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

x_EMs_n_test = x_EMs_n_data[1::2,:]
x_EMs_np1_test = x_EMs_np1_data[1::2,:]
dt_test = dt_train[1::2,:]

N = np.shape(x_EMs_n_test)[0]










