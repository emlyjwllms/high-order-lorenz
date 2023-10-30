# Implementation of a ResNet to perform a step of Euler-Maruyama scheme

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from diffusion-generate-training import dfdx

class ResNetModel:

    def __init__(self):
        
        # Define input layer
        inputs = layers.Input(shape=(4,))  # 3 for x0 and 1 for delta_t
        
        # Save inputs for later usage
        self.x0 = inputs[:, :3]
        self.dt = inputs[:, 3:4]

        # Some residual layers (this attempts to learn the residual function)
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)

        # Compute drift term explicitly
        
        J = self.tf_jacobian_function(self.x0)
        
        # Reshape x0 to [batch_size, 3, 1]
        x0_reshaped = tf.reshape(self.x0, [-1, 3, 1])
        
        # Multiply the tensors
        drift = tf.matmul(J, x0_reshaped)
        
        # Reshape the result back to [batch_size, 3]
        drift = tf.reshape(drift, [-1, 3])
        
        drift_term = drift*self.dt
        
        # Network for sigma_theta
        sigma_theta_layer = layers.Dense(256, activation="relu")(x)
        sigma_theta_layer = layers.Dense(128, activation="relu")(sigma_theta_layer)
        sigma_theta_layer = layers.Dense(64, activation="relu")(sigma_theta_layer)
        sigma_theta_layer = layers.Dense(32, activation="relu")(sigma_theta_layer)
        sigma_theta_diag = layers.Dense(3, activation="relu")(sigma_theta_layer)
        
        # Save diffusion matrix to use in the loss function
        self.sigma_theta = sigma_theta_diag
        
        # Generate random noise of shape [batch_size, 1]
        random_noise = tf.random.normal(shape=(tf.shape(inputs)[0], 1), mean=0, stddev=tf.sqrt(self.dt))
        
        diffusion_term = sigma_theta_diag * random_noise

        # Apply identity map + residual function
        outputs = self.x0 + drift_term + diffusion_term

        self.model = Model(inputs=inputs, outputs=outputs)
        
    def compile(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=self.loss_function)

    def train(self, x_train, y_train, epochs, batch_size):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x):
        return self.model.predict(x)
    
    def tf_jacobian_function(self,x_tensor):
        
        # Wrap the Jacobian function with tf.numpy_function
        J_tensor = tf.numpy_function(dfdx, [x_tensor], tf.float32)
        
        # Set the shape of the output tensor
        batch_size = tf.shape(x_tensor)[0]
        
        J_tensor.set_shape((batch_size, 3, 3))
        
        return J_tensor
    
    def loss_function(self,y_true,y_pred):
    
        # Invert the diagonal values
        sigma_inv_diag = 1.0 / sigma_theta

        x = # i'll figure out in a bit
        mu = # i'll figure out in a bit
        
        diff = x - mu

        term1 = tf.reduce_sum(tf.multiply(diff, sigma_inv_diag*diff)), axis=-1)
        
        det_sigma = tf.reduce_prod(sigma_theta_diag, axis=-1)
        
        term2 = tf.math.log(det_sigma)
        
        # Construct the loss
        loss = -0.5 * 3 * tf.math.log(2.0 * np.pi) - 0.5 * term2 - 0.5 * term1

        return tf.reduce_mean(loss)


