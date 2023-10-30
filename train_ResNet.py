# File to train the ResNet network

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from ResNet_network import ResNetModel

with np.load('diffusion-training-data.npz') as data:
    x_n = data['x_n']
    x_np1 = data['x_np1']
    dt_train = data['dt_train']
    alpha_train = data['alpha_train']

x_train = np.hstack([x_n, dt_train.reshape(-1, 1)])

y_train = x_np1

# Convert numpy data to TensorFlow tensors
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=1024).batch(32)  # Shuffle should be larger than the data size

# Instantiate and compile model
model = ResNetModel()
model.compile(optimizer=Adam(learning_rate=0.001))

# Train the model
model.train(dataset, epochs=10)

# Save weights for later
model.model.save_weights("ResNet_weights/first_try.h5")
