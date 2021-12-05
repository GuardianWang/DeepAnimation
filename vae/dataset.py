import tensorflow as tf
import numpy as np


def load_mnist(batch_size, buffer_size=1024):
    """
    Load and preprocess MNIST dataset from tf.keras.datasets.mnist.

    Inputs:
    - batch_size: An integer value of batch size.
    - buffer_size: Buffer size for random sampling in tf.images.Dataset.shuffle().

    Returns:
    - train_dataset: A tf.images.Dataset instance of MNIST dataset. Batching and shuffling are already supported.
    """
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train / 255.0
    x_train = np.expand_dims(x_train, axis=1)  # [batch_sz, channel_sz, height, width]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return train_dataset
