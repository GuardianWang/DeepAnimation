from vae.model import VAE

import numpy as np
import tensorflow as tf
from PIL import Image
import math
import os
from more_itertools import pairwise


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


def split_img(samples: np.ndarray, batch_size):
    for start, end in pairwise(range(0, samples.shape[0] + 1, batch_size)):
        images = np.squeeze(samples[start: end], 1)
        images = np.clip(images, 0, 1) * 255
        images = images.astype(np.uint8)
        yield images


def save_gray_img(samples: np.ndarray, batch_size, img_folder='.'):
    """

    :param samples: [batch_size * sample_rate, 1, h, w]
    :param batch_size:
    :param img_folder:
    :return:
    """
    for n_img, img_frames in enumerate(split_img(samples, batch_size)):
        img_sub_folder = os.path.join(img_folder, f"image{n_img:03d}")
        os.makedirs(img_sub_folder, exist_ok=True)
        frames = []
        for n_frame, frame in enumerate(img_frames):
            # [h, w]
            frame = Image.fromarray(frame)
            frames.append(frame)
            file = os.path.join(img_sub_folder, f"{n_frame:03d}.png")
            frame.save(file)
        gif_file = os.path.join(img_sub_folder, f"gif{n_img:03d}.gif")
        frames[0].save(gif_file, save_all=True, append_images=frames[1:])


def test_circle_sampling(model: VAE, data: tf.Tensor):
    batch_size = data.shape[0]
    radius = 0.1
    center_vec = tf.one_hot([0], model.latent_size)
    dir_vec = tf.one_hot([1], model.latent_size)
    samples = get_sample(model, data, radius, center_vec, dir_vec, 10).numpy()
    save_gray_img(samples, batch_size)
