import tensorflow as tf
from tensorflow import exp, sqrt, square


@tf.function
def bce_function(x_hat, x):
    """
    Computes the reconstruction loss of the VAE.

    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)

    Returns:
    - reconstruction_loss: Tensor containing the scalar loss for the reconstruction loss term.
    """
    bce_fn = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.SUM,
    )
    reconstruction_loss = bce_fn(x, x_hat) * x.shape[
        -1]  # Sum over all loss terms for each data point. This looks weird, but we need this to work...
    return reconstruction_loss


@tf.function
def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).
    Returned loss is the average loss per sample in the current batch.

    Inputs:
    - x_hat: Reconstructed input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension

    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    ################################################################################################
    # Compute negative variational lowerbound loss as described in the notebook              #
    ################################################################################################
    # Replace "pass" statement with your code
    loss = bce_function(x_hat, x) / x.shape[0]
    loss += -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - square(mu) - exp(logvar), axis=-1))
    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss
