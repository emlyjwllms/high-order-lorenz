# train SDE diffusion for nonlinear solution
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from lorenz_functions import *


import tensorflow as tf
from Network_DietrichBased_NL_BEM import (
                                    SDEIdentification,
                                    ModelBuilder,
                                    SDEApproximationNetwork
                                  )
 
################################ Load data ####################################

porder = 2
h = 1/100
dt = 1/1000

training_data = np.load('data/diffusion-training-data-NL-h' + str(int(1/h)) + '.npz')
x_bar_0 = training_data['x_bar_0']
x_bar_h = training_data['x_bar_h']
assert(h == training_data['h'])
h = training_data['h']

lorenz = np.load('data/dg_lorenz_dt' + str(int(1/dt)) + '_p' + str(porder) + '.npz')

xh = lorenz['xh'].T
t = lorenz['t']
x_n = xh[0:-int(h/dt)-1,:]
cs = lorenz['cs']
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

# SPECIFY IF YOU WANT TO TRAIN AND SAVE THE RESULTS
train = True

######################### Network parameters ##################################

n_layers = 2 #Number of hidden layers
n_dim_per_layer = 5 #Neurons per layer

n_dimensions = 3 #Spatial dimension 

ACTIVATIONS = tf.nn.sigmoid #Activation function
VALIDATION_SPLIT = .5 # 80% for training, 20% for testing
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
N_EPOCHS = 100

# sigma matrix 
diffusivity_type = "spd"

tf.random.set_seed(1)

# define the neural network model we will use for identification
encoder = ModelBuilder.diff_network(
                                    n_input_dimensions=n_dimensions, #int(2*n_dimensions), #Takes xn and bar_xn
                                    n_output_dimensions=n_dimensions,
                                    n_layers=n_layers,
                                    n_dim_per_layer=n_dim_per_layer,
                                    name="diff_net",
                                    activation=ACTIVATIONS,
                                    diffusivity_type=diffusivity_type)
# encoder.summary()

# dictionary with jacobian parameters
jac_par = {'sigma': 10, 'r': 28, 'beta': 8/3}

file_path = 'Trained_Dietrich_NL_BEM'
file_path += '/' + diffusivity_type + '_h' + str(int(1/h)) + '/'
file_path += f'HL{n_layers}_'
file_path += f'N{n_dim_per_layer}_'
file_path += 'SIGMOID_'
file_path += 'LR1e-1_'
file_path += f'BS{BATCH_SIZE}_'
file_path += f'EP{N_EPOCHS}/'

if train:

    n_pts = x_bar_h.shape[0]

    step_sizes = np.zeros(n_pts) + h

    model = SDEApproximationNetwork(sde_model=encoder,
                                    step_size=h,
                                    jac_par=jac_par,
                                    method="euler",
                                    diffusivity_type=diffusivity_type)

    model.compile(optimizer=tf.keras.optimizers.Adamax())

    sde_i = SDEIdentification(model=model)

    bar_xn = x_bar_0
    bar_xn1 = x_bar_h

    hist = sde_i.train_model(bar_xn, bar_xn1, step_size=step_sizes,
                            validation_split=VALIDATION_SPLIT,
                            n_epochs=N_EPOCHS,
                            batch_size=BATCH_SIZE)

    plt.figure(16,figsize=(6,4))
    # plt.title(r"$\Sigma$")
    plt.plot(hist.history["loss"], label='Training')
    plt.plot(hist.history["val_loss"], label='Validation')
    plt.ylim([np.min(hist.history["loss"])*1.1, np.max(hist.history["loss"])])
    #plt.ylim([-5,50])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_path + 'plots/training-NL-BEM-h' + str(int(1/h)) + '.png')
    plt.show()

    model.save_weights(file_path, overwrite=True, save_format=None, options=None)


################## TEST

model = SDEApproximationNetwork(sde_model=encoder,
                                step_size=h,
                                jac_par=jac_par,
                                method="euler",
                                diffusivity_type=diffusivity_type)

model.load_weights(file_path).expect_partial()

sde_i = SDEIdentification(model=model)

TA = 10
TB = 5
ht = np.arange(-TB,TA+TB+1,h)
Nh = len(ht)

# initial conditions
xbar_NN = np.zeros((Nh,3))

# take as first nonzero value of filtered DG
xbar_NN[0,:] = xbar[:,K+1].T

def resid(xbar_n1,xbar_n,dW):
    sigma = sde_i.sample_sigma(xbar_n1)[0]
    return (xbar_n1 - xbar_n - f(xbar_n1,jac_par)*h - np.matmul(sigma,dW))

# time integration
for n in range(Nh-1):
    dW = np.sqrt(h)*np.random.randn(3) # Brownian increment
    xbar_NN[n+1,:] = scipy.optimize.root(resid,xbar_NN[n,:],args=(xbar_NN[n,:],dW)).x


print("saving data")
np.savez('data/diffusion-path-NL-BEM-' + diffusivity_type + '-h' + str(int(1/h)) + '.npz', t=t, xbar_NN=xbar_NN )