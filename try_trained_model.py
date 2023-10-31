# File to test trained models

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf
from MLP_diffusion_network import MLP_diff_NN

# CAREFUL, the following three parameters 
# must exactly match the ones used to train the model
# Otherwhise, the weights cannot be loaded anyways
n_hlayers = 8
BATCH_SIZE = 128
EPOCHS = 1000 

# Recreate the model architecture
model = MLP_diff_NN(n_hlayers)

# Load the weights
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(tf.train.latest_checkpoint(f'MLP_weights/checkpoint_hlayers_{n_hlayers}_BS_{BATCH_SIZE}_EP_{EPOCHS}/trained_MLP_model'))

# Sample tensor with shape (4,)
sample_tensor = tf.constant([0.902, 0.902, 0.902, 0.009])

# Reshape to (1, 4) to denote batch size of 1
input_tensor = tf.expand_dims(sample_tensor, axis=0)

# Pass it to the model
predictions = model(input_tensor)

output = tf.squeeze(predictions)
print(output)


