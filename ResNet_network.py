# Implementation of a ResNet to perform a step of Euler-Maruyama scheme

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

import importlib

# had to use this because the module has '-' in its name lol
dfdx = importlib.import_module('diffusion-generate-training').dfdx

class ResNetModel:

    def __init__(self):
        
        # Define input layer
        inputs = layers.Input(shape=(4,))  # 3 for x0 and 1 for delta_t
        
        # Save inputs for later usage
        self.x0 = inputs[:, :3]
        self.dt = inputs[:, 3:4]

        # Compute drift term explicitly
        
        J = self.tf_dfdx(self.x0)
        
        # Reshape x0 to [batch_size, 3, 1]
        x0_reshaped = tf.reshape(self.x0, [-1, 3, 1])
        
        # Multiply the tensors
        drift = tf.matmul(J, x0_reshaped)
        
        # Reshape the result back to [batch_size, 3]
        drift = tf.reshape(drift, [-1, 3])
        
        # Compute the drift term and save it as member to use in the loss function
        self.drift_term = drift*self.dt
        
        # Network for sigma_theta
        sigma_theta_layer = layers.Dense(256, activation="relu")(inputs)
        sigma_theta_layer = layers.Dense(128, activation="relu")(sigma_theta_layer)
        sigma_theta_layer = layers.Dense(64, activation="relu")(sigma_theta_layer)
        sigma_theta_layer = layers.Dense(32, activation="relu")(sigma_theta_layer)
        sigma_theta_diag = layers.Dense(3, activation="relu")(sigma_theta_layer)
        
        # Save diffusion matrix to use in the loss function
        self.sigma_theta = sigma_theta_diag
        
        # Generate random noise of shape [batch_size, 1]
        random_noise = tf.random.normal(shape=(tf.shape(inputs)[0], 1), mean=0, stddev=tf.sqrt(self.dt))
        
        diffusion_term = sigma_theta_diag * random_noise

        # Apply identity map + drift + residual function
        outputs = self.x0 + self.drift_term + diffusion_term

        self.model = Model(inputs=inputs, outputs=outputs)
        
    def compile(self, optimizer):
        
        self.model.compile(optimizer=optimizer, loss=self.loss_function)

    def train(self, dataset, epochs):
        
        self.model.fit(dataset, epochs=epochs)

    def predict(self, x):
        
        return self.model.predict(x)
    
    
    def loss_function(self,y_true,y_pred):
    
        # Invert the diagonal values
        sigma_inv_diag = 1.0 / (self.dt*self.sigma_theta)

        x = y_true
        mu = self.x0 + self.drift_term
        
        diff = x - mu

        term1 = tf.reduce_sum(tf.multiply(diff, sigma_inv_diag*diff), axis=-1)
        
        det_sigma = tf.reduce_prod(self.sigma_theta, axis=-1)
        
        term2 = tf.math.log(self.dt*det_sigma)
        
        # Construct the loss
        loss = -0.5 * 3 * tf.math.log(2.0 * np.pi) - 0.5 * term2 - 0.5 * term1

        return tf.reduce_mean(loss)
    
    def tf_dfdx(self,xv):
        
        # Jacobian matrix using TF operations
        
        sigma_tensor = tf.constant(10.0, dtype=tf.float32)
        r_tensor = tf.constant(28.0, dtype=tf.float32)
        beta_tensor = tf.constant(8.0/3.0, dtype=tf.float32)
    
        x, y, z = xv[:, 0], xv[:, 1], xv[:, 2]
    
        # Create the Jacobian matrices
        Jx = [tf.negative(sigma_tensor)*tf.ones_like(x), sigma_tensor*tf.ones_like(x), tf.constant(0.0, dtype=tf.float32)*tf.ones_like(x)]
        Jy = [r_tensor - z, tf.constant(-1.0, dtype=tf.float32)*tf.ones_like(y), tf.negative(x)]
        Jz = [y, x, tf.negative(beta_tensor)*tf.ones_like(z)]
    
        J = tf.stack([Jx, Jy, Jz], axis=2)
    
        # Ensure the J matrix has the expected shape of (batch_size, 3, 3)
        J = tf.transpose(J, perm=[2, 0, 1])
    
        return J


