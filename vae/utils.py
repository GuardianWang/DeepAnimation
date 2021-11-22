import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


@tf.function
def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset
    Returns:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when
    the ground truth label for image i is j, and targets[i, :j] &
    targets[i, j + 1:] are equal to 0
    """
    return tf.one_hot(labels, class_size)


def save_model_weights(model, args):
    """
    Save trained VAE model weights to model_ckpts/

    Inputs:
    - model: Trained VAE model.
    - cfg: All arguments.
    """
    model_flag = "cvae" if args.is_cvae else "vae"
    output_dir = os.path.join("model_ckpts", model_flag)
    output_path = os.path.join(output_dir, model_flag)
    os.makedirs("model_ckpts", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(output_path)


def show_vae_images(model, latent_size):
    """
    Call this only if the model is VAE!
    Generate 10 images from random vectors.
    Show the generated images from your trained VAE.
    Image will be saved to outputs/show_vae_images.pdf

    Inputs:
    - model: Your trained model.
    - latent_size: Latent size of your model.
    """
    # Generated images from vectors of random values.
    z = tf.random.normal(shape=[10, latent_size])
    samples = model.decoder(z).numpy()

    # Visualize
    fig = plt.figure(figsize=(10, 1))
    gspec = gridspec.GridSpec(1, 10)
    gspec.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")

    # Save the generated images
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "show_vae_images.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def show_vae_interpolation(model, latent_size):
    """
    Call this only if the model is VAE!
    Generate interpolation between two .
    Show the generated images from your trained VAE.
    Image will be saved to outputs/show_vae_interpolation.pdf

    Inputs:
    - model: Your trained model.
    - latent_size: Latent size of your model.
    """

    def show_interpolation(images):
        """
        A helper to visualize the interpolation.
        """
        images = tf.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
        sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
        sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

        fig = plt.figure(figsize=(sqrtn, sqrtn))
        gs = gridspec.GridSpec(sqrtn, sqrtn)
        gs.update(wspace=0.05, hspace=0.05)
        for i, img in enumerate(images):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(tf.reshape(img, [sqrtimg, sqrtimg]))

        # Save the generated images
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", "show_vae_interpolation.pdf")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    S = 12
    z0 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)  # [S, latent_size]
    z1 = tf.random.normal(shape=[S, latent_size], dtype=tf.dtypes.float32)
    w = tf.linspace(0, 1, S)
    w = tf.cast(tf.reshape(w, (S, 1, 1)), dtype=tf.float32)  # [S, 1, 1]
    z = tf.transpose(w * z0 + (1 - w) * z1, perm=[1, 0, 2])
    z = tf.reshape(z, (S * S, latent_size))  # [S, S, latent_size]
    x = model.decoder(z)  # [S*S, 1, 28, 28]
    show_interpolation(x)


def show_cvae_images(model, latent_size):
    """
    Call this only if the model is CVAE!
    Conditionally generate 10 images for each digit.
    Show the generated images from your trained CVAE.
    Image will be saved to outputs/show_cvae_images.pdf

    Inputs:
    - model: Your trained model.
    - latent_size: Latent size of your model.
    """
    # Conditionally generated images from vectors of random values.
    num_generation = 100
    num_classes = 10
    num_per_class = num_generation // num_classes
    c = tf.eye(num_classes)  # [one hot labels for 0-9]
    z = []
    labels = []
    for label in range(num_classes):
        curr_c = c[label]
        curr_c = tf.broadcast_to(curr_c, [num_per_class, len(curr_c)])
        curr_z = tf.random.normal(shape=[num_per_class, latent_size])
        curr_z = tf.concat([curr_z, curr_c], axis=-1)
        z.append(curr_z)
        labels.append([label] * num_per_class)
    z = np.concatenate(z)
    labels = np.concatenate(labels)
    samples = model.decoder(z).numpy()

    # Visualize
    rows = num_classes
    cols = num_generation // rows

    fig = plt.figure(figsize=(cols, rows))
    gspec = gridspec.GridSpec(rows, cols)
    gspec.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gspec[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(sample.reshape(28, 28), cmap="Greys_r")

    # Save the generated images
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "show_cvae_images.pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_weights(model, is_cvae):
    """
    Load the trained model's weights.

    Inputs:
    - model: Your untrained model instance.

    Returns:
    - model: Trained model.
    """
    num_classes = 10
    inputs = tf.zeros([1, 1, 28, 28])  # Random data sample
    labels = tf.constant([[0]])
    if is_cvae:
        weights_path = os.path.join("model_ckpts", "cvae", "cvae")
        one_hot_vec = one_hot(labels, num_classes)
        _ = model(inputs, one_hot_vec)
        model.load_weights(weights_path)
    else:
        weights_path = os.path.join("model_ckpts", "vae", "vae")
        _ = model(inputs)
        model.load_weights(weights_path)
    return model
