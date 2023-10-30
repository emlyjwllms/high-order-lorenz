# Implementation of a Multilayer Perceptron NN with multiple hidden layers
# and ReLU activation function

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import tensorflow as tf

class MLP_diff_NN(tf.Module):
    
    #constructor: set up the architecture
    def __init__(self, n_hlayers, name=None):
        
        # call base class constructor passing name
        super().__init__(name=name)
        
        # set the number of neurons in each internal layer
        # it is done such that the last hidden layer contains 32 neurons
        # and then each layer contains twice as much as the next one
        self.neurons_per_layer = np.zeros(n_hlayers,dtype=int)
        
        for i in range(len(self.neurons_per_layer)):
            
            self.neurons_per_layer[i] = int(2**(n_hlayers + 5 - (i + 1)))
            
        # creates the hidden layer    
        self.h_layers = []
        for i in range(n_hlayers):
            
            neurons = self.neurons_per_layer[i]
            activation = tf.nn.relu # use ReLU activation
            layer = tf.keras.layers.Dense(neurons,activation)
            self.h_layers.append(layer)
            
        assert(len(self.h_layers) == n_hlayers), "Hidden layers construction failed"
        
        self.n_hlayers = n_hlayers
        
        # creates that output layer
        self.out = tf.keras.layers.Dense(3) # output dimension is three for Lorentz Eqs.
    
    # overloads operator()        
    def __call__(self, X):
        
        # perform sequential multiplications
        for i in range(self.n_hlayers):
            
            X = self.h_layers[i](X)
            
        return self.out(X)
    
    # define loss function, in this case the average of the squared error
    def compute_loss(self, predicted_X1, true_X1):
        
        return tf.reduce_mean(tf.square(predicted_X1 - true_X1))

    # define the gradient using TF automatic diff
    def compute_gradients(model, train_X0, train_X1):
        
        with tf.GradientTape() as tape:
            
            predicted_X1 = self.call(train_X0)
            
            loss = self.compute_loss(predicted_X1, train_X1)
            
        return tape.gradient(loss, self.trainable_variables), loss

    
    
    
        
