"""
Brown CSCI 1470/2470 Deep Learning homework 5
"""
from utils import load_weights

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Concatenate
from tensorflow import exp, sqrt, square

__all__ = ['VAE', 'CVAE', 'get_model']


class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 128  # H_d
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        ############################################################################################
        # Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code
        self.encoder = Sequential()
        self.encoder.add(Flatten())
        self.encoder.add(Dense(self.hidden_dim * 4, 'relu'))
        self.encoder.add(Dense(self.hidden_dim * 2, 'relu'))
        self.encoder.add(Dense(self.hidden_dim, 'relu'))
        self.mu_layer = Dense(self.latent_size)
        self.logvar_layer = Dense(self.latent_size)

        ############################################################################################
        # Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        self.decoder = Sequential()
        self.decoder.add(Input(self.latent_size, ))
        self.decoder.add(Dense(self.hidden_dim, 'relu'))
        self.decoder.add(Dense(self.hidden_dim * 2, 'relu'))
        self.decoder.add(Dense(self.hidden_dim * 4, 'relu'))
        self.decoder.add(Dense(self.input_size, 'sigmoid'))

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    @tf.function
    def call(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)

        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_shape = x.shape
        ############################################################################################
        # Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to reconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        x, mu, logvar = self.encode(x)
        x_hat = self.decode(x, x_shape)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        x = reparametrize(mu, logvar)

        return x, mu, logvar

    def decode(self, x, shape):
        x = self.decoder(x)
        x = tf.reshape(x, shape)
        return x


class CVAE(tf.keras.Model):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C
        self.hidden_dim = 512  # H_d
        self.lr = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        ############################################################################################
        # Define a FC encoder as described in the notebook that transforms the image--after  #
        # flattening and now adding our one-hot class vector (N, H*W + C)--into a hidden_dimension #               #
        # (N, H_d) feature space, and a final two layers that project that feature space           #
        # to posterior mu and posterior log-variance estimates of the latent space (N, Z)          #
        ############################################################################################
        # Replace "pass" statement with your code
        self.flatten = Flatten()
        self.concat = Concatenate()
        self.encoder = Sequential()
        self.encoder.add(Input(self.input_size + self.num_classes, ))
        self.encoder.add(Dense(self.hidden_dim, 'relu'))
        self.encoder.add(Dense(self.hidden_dim, 'relu'))
        self.encoder.add(Dense(self.hidden_dim, 'relu'))
        self.mu_layer = Dense(self.latent_size)
        self.logvar_layer = Dense(self.latent_size)

        ############################################################################################
        # Define a fully-connected decoder as described in the notebook that transforms the  #
        # latent space (N, Z + C) to the estimated images of shape (N, 1, H, W).                   #
        ############################################################################################
        # Replace "pass" statement with your code
        self.decoder = Sequential()
        self.decoder.add(Input(self.latent_size + self.num_classes, ))
        self.decoder.add(Dense(self.hidden_dim, 'relu'))
        self.decoder.add(Dense(self.hidden_dim, 'relu'))
        self.decoder.add(Dense(self.hidden_dim, 'relu'))
        self.decoder.add(Dense(self.input_size, 'sigmoid'))

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    @tf.function
    def call(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)

        Returns:
        - x_hat: Reconstruced input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with Z latent space dimension
        """
        x_shape = x.shape
        ############################################################################################
        # Implement the forward pass by following these steps                                #
        # (1) Pass the concatenation of input batch and one hot vectors through the encoder model  #
        # to get posterior mu and logvariance                                                      #
        # (2) Reparametrize to compute the latent vector z                                         #
        # (3) Pass concatenation of z and one hot vectors through the decoder to reconstruct x    #
        ############################################################################################
        # Replace "pass" statement with your code
        x = self.flatten(x)
        x = self.concat([x, c])
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        x = reparametrize(mu, logvar)
        x = self.concat([x, c])
        x = self.decoder(x)
        x_hat = tf.reshape(x, x_shape)

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar


@tf.function
def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    ################################################################################################
    # Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code
    z = tf.random.normal(mu.shape)
    z = mu + z * exp(logvar * 0.5)
    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def get_model(cfg):
    if cfg.is_cvae:
        model = CVAE(cfg.input_size, latent_size=cfg.latent_size)
    else:
        model = VAE(cfg.input_size, latent_size=cfg.latent_size)
    if cfg.load_weights:
        load_weights(model, cfg.is_cvae)

    return model
