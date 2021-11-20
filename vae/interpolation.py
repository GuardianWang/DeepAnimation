from vae.model import VAE

import tensorflow as tf
import math


def encode(model: VAE, data: tf.Tensor):
    return model.encode(data)[0]


def circle_sampling(latent: tf.Tensor, radius: float, center_vec, dir_vec, sample_rate: int):
    """
    Sampling along a circle.
    :param latent:
        Latent representation of shape [batch_size, latent_size]
    :param radius:
    :param center_vec:
        [1, latent_size] or [sample_rate, latent_size]
    :param dir_vec:
        [1, latent_size] or [sample_rate, latent_size]
    :param sample_rate:
    :return:
    """

    # [sample_rate, 1]
    thetas = tf.linspace(0, tf.constant(math.pi), sample_rate)[:, tf.newaxis]
    # unit vector
    center_vec = tf.math.l2_normalize(center_vec, axis=1)
    dir_vec = tf.math.l2_normalize(dir_vec, axis=1)
    # [sample_rate, latent_size]
    deltas = (radius - radius * tf.cos(thetas)) * center_vec + radius * tf.sin(thetas) * dir_vec
    # [sample_rate * batch_size, latent_size]
    deltas = tf.tile(deltas, [latent.shape[0], 1])
    # [sample_rate * batch_size, latent_size]
    deltas = deltas[:, tf.newaxis]

    # [batch_size * sample_rate, latent_size]
    latent = tf.repeat(latent, [sample_rate], axis=0)
    latent += deltas

    return latent


def decode(model: VAE, latent: tf.Tensor, shape: tf.Tensor):
    return model.decode(latent, shape)


def get_sample(model: VAE, data: tf.Tensor, radius: float, center_vec, dir_vec, sample_rate: int):
    latent = encode(model, data)
    latent = circle_sampling(latent, radius, center_vec, dir_vec, sample_rate)
    decode_shape = [latent.shape[0]] + data.shape[1:]
    # [batch_size * sample_rate, c, h, w]
    return decode(model, latent, decode_shape)
