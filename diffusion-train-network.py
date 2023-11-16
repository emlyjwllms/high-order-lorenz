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

import tensorflow as tf
from Network_DietrichBased import (
                                    SDEIdentification,
                                    ModelBuilder,
                                    SDEApproximationNetwork
                                  )
 
################################ Load data ####################################

training_data = np.load('diffusion-training-data.npz')
x_tilde_n = training_data['x_tilde_n']
x_tilde_np1 = training_data['x_tilde_np1']

porder = 2

lorenz = np.load('dg_lorenz_dt100_p' + str(porder) + '.npz')

x_n = lorenz['xh'].T
x_n = x_n[0:-1,:]
cs = lorenz['cs']
t = lorenz['t']
dt = t[1]-t[0]

# Prepare the input tensor
for n_point, nominal_point in enumerate(x_n):
    
    xn_matrix = np.tile(nominal_point,(1000,1))
    
    slice_start = int(n_point*1000)
    slice_end = int(slice_start + 1000)
    
    x_tilde_n_slice = x_tilde_n[slice_start:slice_end,:]
    x_tilde_n1_slice = x_tilde_np1[slice_start:slice_end,:]
    
    new_block = np.hstack((xn_matrix,x_tilde_n_slice,x_tilde_n1_slice))
    
    if n_point == 0:
        
        full_data = new_block
        
    else:
        
        full_data = np.vstack((full_data,new_block))

n_pts = x_tilde_n.shape[0]

step_sizes = np.zeros(n_pts) + dt

#%%
######################### Network parameters ##################################

n_layers = 3 #Number of hidden layers
n_dim_per_layer = 100 #Neurons per layer

n_dimensions = 3 #Spatial dimension 

ACTIVATIONS = tf.nn.leaky_relu #Activation function
VALIDATION_SPLIT = .2 # 80% for training, 20% for testing
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
N_EPOCHS = 100

# use diagonal sigma matrix
diffusivity_type = "diagonal"

# define the neural network model we will use for identification
encoder = ModelBuilder.diff_network(
                                    n_input_dimensions=int(2*n_dimensions), #Takes xn and tilde_xn
                                    n_output_dimensions=n_dimensions,
                                    n_layers=n_layers,
                                    n_dim_per_layer=n_dim_per_layer,
                                    name="diff_net",
                                    activation=ACTIVATIONS,
                                    diffusivity_type=diffusivity_type)
encoder.summary()

#dictionary with jacobian parameters
jac_par = {'sigma': 10, 'r': 28, 'beta': 8/3}

model = SDEApproximationNetwork(sde_model=encoder,
                                step_size=dt,
                                jac_par=jac_par,
                                method="euler",
                                diffusivity_type=diffusivity_type)

model.compile(optimizer=tf.keras.optimizers.Adamax())

sde_i = SDEIdentification(model=model)

xn = full_data[:,0:3]
tilde_xn = full_data[:,3:6]
tilde_xn1 = full_data[:,6:9]

hist = sde_i.train_model(xn, tilde_xn, tilde_xn1, step_size=step_sizes,
                         validation_split=VALIDATION_SPLIT,
                         n_epochs=N_EPOCHS,
                         batch_size=BATCH_SIZE)
#%%

fig, hist_axes = plt.subplots(1, 1, figsize=(10, 5))
hist_axes.clear()
hist_axes.set_title(r"Training Results for Diagonal $\Sigma$")
hist_axes.plot(hist.history["loss"], label='Loss')
hist_axes.plot(hist.history["val_loss"], label='Validation')
hist_axes.set_ylim([np.min(hist.history["loss"])*1.1, np.max(hist.history["loss"])])
hist_axes.set_xlabel("Epoch")
hist_axes.legend()
fig.savefig('FirstTrainingDiagonal.pdf')

#%%

file_path = 'Trained_Dietrich'
file_path += '/' + diffusivity_type + '/'
file_path += f'HL{n_layers}_'
file_path += f'N{n_dim_per_layer}_'
file_path += 'LReLU_'
file_path += 'LR1e-1_'
file_path += f'BS{BATCH_SIZE}_'
file_path += f'EP{N_EPOCHS}/'

#%%

# For using the model, you would do:

# sde_i.sample_tilde_xn1(xn, tilde_xn, step_size, jac_par, diffusivity_type)


#provided you have xn, tilde_xn and step_size  


















