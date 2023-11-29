import tensorflow as tf
from tensorflow.keras import layers

import keras
import keras.backend as K

import tensorflow_probability as tfp

import sys
import numpy as np

import json

tfd = tfp.distributions

# For numeric stability, set the default floating-point dtype to float64
tf.keras.backend.set_floatx('float64')
NUMBER_TYPE = tf.float64  # or tf.float32

STD_MIN_VALUE = 1e-13  # the minimal number that the diffusivity models can have

class ModelBuilder:
    """
    Constructs neural network models with specified topology.
    """
    DIFF_TYPES = ["diagonal", "triangular", "spd"]

    @staticmethod
    def diff_network(n_input_dimensions,
                     n_output_dimensions,
                     n_layers,
                     n_dim_per_layer,
                     name,
                     diffusivity_type="diagonal",
                     activation="tanh",
                     dtype=tf.float64,
                     ):
        def make_tri_matrix(z):
            # first, make all eigenvalues positive by changing the diagonal to positive values
            z = tfp.math.fill_triangular(z)
            z2 = tf.linalg.diag(tf.linalg.diag_part(z))
            z = z - z2 + tf.abs(z2)  # this ensures the values on the diagonal are positive
            return z

        def make_spd_matrix(z):
            z = make_tri_matrix(z)
            return tf.linalg.matmul(z, tf.linalg.matrix_transpose(z))

        # initialize with small (not zero!) values so that it does not dominate the drift
        # estimation at the beginning of training
        small_init = 1e-2
        # small_init = 0.3
        initializer = tf.keras.initializers.RandomUniform(minval=-small_init, maxval=small_init, seed=None)
        
        inputs = layers.Input((n_input_dimensions,), dtype=dtype, name=name + '_inputs')

        #Network for Sigma matrix
        gp_x = inputs
        for i in range(n_layers):
            gp_x = layers.Dense(n_dim_per_layer,
                                activation=activation,
                                dtype=dtype,
                                kernel_initializer=initializer,
                                bias_initializer=initializer,
                                name=name + "_std_hidden_{}".format(i))(gp_x)
        if diffusivity_type == "diagonal":
            gp_output_std = layers.Dense(n_output_dimensions,
                                         kernel_initializer=initializer,
                                         bias_initializer=initializer,
                                         activation=lambda x: tf.nn.softplus(x) + STD_MIN_VALUE,
                                         name=name + "_output_std", dtype=dtype)(gp_x)
        elif diffusivity_type == "triangular":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the lower triangular matrix with positive eigenvalues on the diagonal.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_cholesky", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_tri_matrix)(gp_output_tril)
        elif diffusivity_type == "spd":
            # the dimension of std should be N*(N+1)//2, for one of the Cholesky factors L of the covariance,
            # so that we can create the SPD matrix C using C = L @ L.T to be used later.
            gp_output_tril = layers.Dense((n_output_dimensions * (n_output_dimensions + 1) // 2),
                                          activation="linear",
                                          kernel_initializer=initializer,
                                          bias_initializer=initializer,
                                          name=name + "_output_spd", dtype=dtype)(gp_x)
            gp_output_std = layers.Lambda(make_spd_matrix)(gp_output_tril)
            # gp_output_std = layers.Lambda(lambda L: tf.linalg.matmul(L, tf.transpose(L)))(gp_output_tril)
        else:
            raise ValueError(f"Diffusivity type {diffusivity_type} not supported. Use one of {ModelBuilder.DIFF_TYPES}.")

        gp = tf.keras.Model(inputs,
                            gp_output_std,
                            name=name + "_gaussian_process")
        return gp

    @staticmethod
    def define_normal_distribution(xn,
                                   tilde_xn,
                                   step_size_,
                                   jac_par,
                                   sigma_theta, 
                                   diffusivity_type):
        
        #Define constant tensors for using in Jacobian computation
        sigma = tf.constant([jac_par['sigma']], dtype=tf.float64)
        r = tf.constant([jac_par['r']], dtype=tf.float64)
        beta = tf.constant([jac_par['beta']], dtype=tf.float64)
        
        #Get individual coordinates of xn
        if len(xn.shape) == 1:
            x = tf.reshape(xn[0], [1])
            y = tf.reshape(xn[1], [1])
            z = tf.reshape(xn[2], [1])
        else:
            x, y, z = xn[:,0], xn[:,1], xn[:,2]
        
        # Compute Jacobian entries evaluated at xn
        df1dx = -sigma * tf.ones_like(x)
        df1dy = sigma * tf.ones_like(x)
        df1dz = tf.zeros_like(x)
        df2dx = r - z
        df2dy = -tf.ones_like(x)
        df2dz = -x
        df3dx = y
        df3dy = x
        df3dz = -beta * tf.ones_like(x)
                
        # Creates Jacobian tensor
        J = tf.stack([
            tf.stack([df1dx, df1dy, df1dz], axis=1),
            tf.stack([df2dx, df2dy, df2dz], axis=1),
            tf.stack([df3dx, df3dy, df3dz], axis=1)
        ], axis=1)
        
        #Compute drift term
        drift_ = tf.linalg.matvec(J, tilde_xn)
        
        #Compute mean
        mean = tilde_xn + drift_ * step_size_
        
        # #Compute sigma_theta*tilde_xn
        # sigma_tilde_xn = tf.linalg.matvec(sigma_theta, tilde_xn)
        
        #Compute diffusion matrix: (sigma_theta*tilde_xn)*(sigma_theta*tilde_xn)^T
        # diff_matrix = tf.matmul(sigma_tilde_xn[:, tf.newaxis], sigma_tilde_xn[:, tf.newaxis], transpose_b=True)
        diff_matrix = sigma_theta

        if diffusivity_type == "diagonal":
            approx_normal = tfd.MultivariateNormalDiag(
                loc=(mean),
                scale_diag=tf.math.sqrt(step_size_) * diff_matrix,
                name="approx_normal"
            )
        elif diffusivity_type == "triangular":
            # form the normal distribution with a lower triangular matrix
            approx_normal = ModelBuilder.define_normal_distribution_triangular(tilde_xn, step_size_, drift_, diff_matrix)
        elif diffusivity_type == "spd":
            # form the normal distribution with SPD matrix
            approx_normal = ModelBuilder.define_normal_distribution_spd(tilde_xn, step_size_, drift_, diff_matrix)
        else:
            raise ValueError(
                f"Diffusivity type <{diffusivity_type}> not supported. Use one of {ModelBuilder.DIFF_TYPES}.")
        return approx_normal

    @staticmethod
    def define_normal_distribution_triangular(yn_, step_size_, drift_, diffusivity_tril_):
        """
        Construct a normal distribution with the Euler-Maruyama template, in tensorflow.

        Parameters
        ----------
        yn_ current points (batch_size x dimension)
        step_size_ step sizes per point (batch_size x 1)
        drift_ estimated drift at yn_
        diffusivity_spd_ estimated diffusivity matrix at yn_ (batch_size x n_dim x n_dim)

        Returns
        -------
        tfd.MultivariateNormalTriL object

        """
        # a cumbersome way to multiply the step size scalar with the batch of matrices...
        # better use tfp.bijectors.FillScaleTriL()
        tril_step_size = tf.math.sqrt(step_size_)
        n_dim = K.shape(yn_)[-1]
        full_shape = n_dim * n_dim
        step_size_matrix = tf.broadcast_to(tril_step_size, [K.shape(step_size_)[0], full_shape])
        step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

        # now form the normal distribution
        approx_normal = tfd.MultivariateNormalTriL(
            loc=(yn_ + step_size_ * drift_),
            scale_tril=tf.multiply(step_size_matrix, diffusivity_tril_),
            name="approx_normal"
        )
        return approx_normal

    @staticmethod
    def define_normal_distribution_spd(yn_, step_size_, drift_, diffusivity_spd_):
        """
        Construct a normal distribution with the Euler-Maruyama template, in tensorflow.

        Parameters
        ----------
        yn_ current points (batch_size x dimension)
        step_size_ step sizes per point (batch_size x 1)
        drift_ estimated drift at yn_
        diffusivity_spd_ estimated diffusivity matrix at yn_ (batch_size x n_dim x n_dim)

        Returns
        -------
        tfd.MultivariateNormalTriL object

        """
        # a cumbersome way to multiply the step size scalar with the batch of matrices...
        # TODO: REFACTOR with diffusivity_type=="triangular"
        spd_step_size = tf.math.sqrt(step_size_)  # NO square root because we use cholesky below?
        n_dim = K.shape(yn_)[-1]
        full_shape = n_dim * n_dim
        step_size_matrix = tf.broadcast_to(spd_step_size, [K.shape(step_size_)[0], full_shape])
        step_size_matrix = tf.reshape(step_size_matrix, (-1, n_dim, n_dim))

        # multiply with the step size
        covariance_matrix = tf.multiply(step_size_matrix, diffusivity_spd_)
        # square the matrix so that the cholesky decomposition does not change the eienvalues
        covariance_matrix = tf.linalg.matmul(covariance_matrix, tf.linalg.matrix_transpose(covariance_matrix))
        # perform cholesky to get the lower trianular matrix needed for MultivariateNormalTriL
        covariance_matrix = tf.linalg.cholesky(covariance_matrix)

        # now form the normal distribution
        approx_normal = tfd.MultivariateNormalTriL(
            loc=(yn_ + step_size_ * drift_),
            scale_tril=covariance_matrix,
            name="approx_normal"
        )
        return approx_normal
    
class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    @staticmethod
    def __log(message, flush=True):
        sys.stdout.write(message)
        if flush:
            sys.stdout.flush()

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        LossAndErrorPrintingCallback.__log(
            "\rThe average loss for epoch {} is {:7.10f} ".format(
                epoch, logs["loss"]
            )
        )

class SDEIdentification:
    """
    Wrapper class that can be used for SDE identification.
    Needs a "tf.keras.Model" like the SDEApproximationNetwork or VAEModel to work.
    """

    def __init__(self, model=None, path=None):
        
        if model == None:
            
            assert(path is not None), "Model not provided. Please, provide a fresh model or path to trained model"
            
            #Load SDEApp dictionary
            with open(path + 'SDEApp_data.json', 'r') as infile:
                SDEApp = json.load(infile)
                
            step_size = SDEApp['step_size'] 
            jac_par = SDEApp['jac_par']
            method = SDEApp['method']
            diffusivity_type = SDEApp['diffusivity_type']
            
            #Load inner model (diff_network)
            loaded_diff_network = tf.keras.models.load_model(path + 'diff_network/')
            
            #Construct outer model (SDEApproximationNetwork)
            SDEApp_model = SDEApproximationNetwork(loaded_diff_network, step_size, jac_par, method, diffusivity_type)
            
            # #Load outer model weights
            # SDEApp_model.load_weights(path + 'SDEApp_model/')
            
            self.model = SDEApp_model
            
        else:
            
         self.model = model

    def train_model(self, xn, tilde_xn, tilde_xn1, step_size, validation_split=0.1, n_epochs=100, batch_size=1000, callbacks=[]):
        print(f"training for {n_epochs} epochs with {int(xn.shape[0] * (1 - validation_split))} data points"
              f", validating with {int(xn.shape[0] * validation_split)}")

        features = np.column_stack([xn, tilde_xn])

        full_dataset = np.column_stack([features, tilde_xn1])
        
        if step_size is not None:
            full_dataset = np.column_stack([full_dataset, step_size])

        if len(callbacks) == 0:
            callbacks.append(LossAndErrorPrintingCallback())

        hist = self.model.fit(x=full_dataset,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              verbose=0,
                              validation_split=validation_split,
                              callbacks=callbacks)
        return hist
    
    def eval_sigma(self, xn, tilde_xn):
        
        sigma_theta = self.model.call_xn(xn, tilde_xn)
        return K.eval(sigma_theta)
    
    def sample_tilde_xn1(self,
                         xn,
                         tilde_xn,
                         step_size,
                         jac_par,
                         diffusivity_type):
        """
        Use the neural network to sample a path with the Euler Maruyama scheme.
        """
        sigma_theta = self.model.call_xn(xn, tilde_xn)

        approx_normal = ModelBuilder.define_normal_distribution(xn,
                                                                tilde_xn,
                                                                step_size,
                                                                jac_par,
                                                                sigma_theta,
                                                                diffusivity_type)

        tilde_xn1 = approx_normal.sample()
        return keras.backend.eval(tilde_xn1)
    
    def save_model(self,path):
        
        SDEApp = {}
        SDEApp['step_size'] = self.model.step_size
        SDEApp['jac_par'] =  self.model.jac_par
        SDEApp['method'] = self.model.method
        SDEApp['diffusivity_type'] = self.model.diffusivity_type
        
        #Save dictionary with model data
        with open(path + 'SDEApp_data.json', 'w') as outfile:
            json.dump(SDEApp, outfile)
        
        #Save weights of outer model (SDEApproximationNetwork)       
        self.model.save(path + 'SDEApp_model/SDEApp_model_weights.h5')
        
        #Save inner model (diff_network)
        self.model.sde_model.save(path + 'diff_network')
        
        

class SDEApproximationNetwork(tf.keras.Model):
    """
    A neural network sde_model that uses a given
    sde_model network to predict Sigma matrix
    of our linearized Lorenz, and trains it using EM scheme
    based loss functions.
    """
    VALID_METHODS = ["euler"]

    def __init__(self,
                 sde_model: tf.keras.Model,
                 step_size,
                 jac_par,
                 method="euler",
                 diffusivity_type="diagonal",
                 scale_per_point=1e-3,
                 **kwargs):
        super().__init__(**kwargs)
        self.sde_model = sde_model
        self.step_size = step_size
        self.jac_par = jac_par
        self.method = method
        self.diffusivity_type = diffusivity_type

        SDEApproximationNetwork.verify(self.method)

    @staticmethod
    def verify(method):
        if not (method in SDEApproximationNetwork.VALID_METHODS):
            raise ValueError(method + " is not a valid method. Use any of" + SDEApproximationNetwork.VALID_METHODS)

    def get_config(self):
        return {
            "sde_model": self.sde_model,
            "step_size": self.step_size,
            "method": self.method,
            "diffusivity_type": self.diffusivity_type
        }

    @staticmethod
    def euler_maruyama_pdf(xn, tilde_xn, tilde_xn1, step_size, jac_par, model_, diffusivity_type="diagonal"):
        """
        This implies a very simple sde_model, essentially just a Gaussian process
        on x_n that predicts the drift and diffusivity.
        Returns log P(y(n+1) | y(n)) for the Euler-Maruyama scheme.

        Parameters
        ----------
        ynp1_ next point in time.
        yn_ current point in time.
        step_size_ step size in time.
        model_ sde_model that returns a (drift, diffusivity) tuple.
        parameters_ parameters of the model at yn_. Default: None.
        diffusivity_type defines which type of diffusivity matrix will be used. See ModelBuilder.DIFF_TYPES.

        Returns
        -------
        logarithm of p(ynp1_ | yn_) under the Euler-Maruyama scheme.

        """
        
        sigma_theta = model_(tf.concat([xn, tilde_xn], axis=1)) #Call to the model defined in ModelBuilder.diff_network

        approx_normal = ModelBuilder.define_normal_distribution(xn,
                                                                tilde_xn,
                                                                step_size,
                                                                jac_par,
                                                                sigma_theta,
                                                                diffusivity_type)
        return approx_normal.log_prob(tilde_xn1)
    
    @staticmethod
    def split_inputs(inputs, step_size=None):
        
        n_total = inputs.shape[1]
        
        if step_size is not None:
            # Subtract one for the step_size at the end
            n_each = (n_total - 1) // 3
            x_n, tilde_xn, tilde_xn1, step_size = tf.split(inputs, num_or_size_splits=[n_each, n_each, n_each, 1], axis=1)
        else:
            n_each = n_total // 3
            x_n, tilde_xn, tilde_xn1 = tf.split(inputs, num_or_size_splits=[n_each, n_each, n_each], axis=1)
    
        return x_n, tilde_xn, tilde_xn1, step_size

    def call_xn(self, xn, tilde_xn):
        """
        Can be used to evaluate the drift and diffusivity
        of the sde_model. This is different than the "call" method
        because it only expects "x_k", not "x_{k+1}" as well.

        Parameters
        ----------
        inputs_xn input points: xn and tilde_xn

        Returns evaluated the diffusivity of the sde_model
        -------
        """
        assert(len(xn.shape) == len(tilde_xn.shape)), "Shape dimension mismatch between xn and tilde_xn"
        if len(xn.shape) == 1:
            xn = xn.reshape((1,3))
            tilde_xn = tilde_xn.reshape((1,3))
            arguments = tf.concat([xn, tilde_xn], axis=1)
            #print(arguments.shape)
        else:
            arguments = tf.concat([xn, tilde_xn], axis=1)
        return self.sde_model(arguments)

    def call(self, inputs):
        """
        Expects the input tensor to contain all of (xn, tilde_xn, tilde_xn1, step_size).
        """
        xn, tilde_xn, tilde_xn1, step_size = SDEApproximationNetwork.split_inputs(inputs, self.step_size)

        if self.method == "euler":
            log_prob = SDEApproximationNetwork.euler_maruyama_pdf(xn, tilde_xn, tilde_xn1, step_size, self.jac_par, self.sde_model,
                                                                  diffusivity_type=self.diffusivity_type)

        else:
            raise ValueError(self.method + " not available")

        sample_distortion = -tf.reduce_mean(log_prob, axis=-1)
        distortion = tf.reduce_mean(sample_distortion)

        loss = distortion

        # correct the loss so that it converges to zero regardless of dimension
        loss = loss + 2 * np.log(2 * np.pi) / np.log(10) * xn.shape[1]

        self.add_loss(loss)
        self.add_metric(distortion, name="distortion", aggregation="mean")

        return self.sde_model(tf.concat([xn, tilde_xn], axis=1))
    