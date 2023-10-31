# File to train the MLP diffusion network

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from MLP_diffusion_network import MLP_diff_NN


# Load the training data
data = np.load('diffusion-training-data.npz')
x_n = data['x_n']
x_np1 = data['x_np1']
dt_train = data['dt_train']
alpha_train = data['alpha_train']

# Prepare the input and output for the network
# Combining x_n and dt_train for the input
input_data = np.hstack([x_n, dt_train.reshape(-1, 1)])

# Convert alpha_train to a 3D vector
output_data = alpha_train[:, np.newaxis] * np.array([1, 1, 1])


# Create an instance of the network
n_hlayers = 8
model = MLP_diff_NN(n_hlayers)

# Optimizer
optimizer = Adam()

# Training loop
EPOCHS = 1000
BATCH_SIZE = 128

for epoch in range(EPOCHS):
    for i in range(0, len(input_data), BATCH_SIZE):
        # Get mini-batch data
        batch_input = input_data[i:i+BATCH_SIZE]
        batch_output = output_data[i:i+BATCH_SIZE]

        # Compute gradients and apply them
        with tf.GradientTape() as tape:
            predictions = model(batch_input)
            loss = model.compute_loss(predictions, batch_output)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Print epoch loss
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.numpy()}")

# Save the weights for later
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.save(f'MLP_weights/checkpoint_hlayers_{n_hlayers}_BS_{BATCH_SIZE}_EP_{EPOCHS}/trained_MLP_model')





