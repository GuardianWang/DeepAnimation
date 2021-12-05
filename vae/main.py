from .utils import *
from .model import *
from .dataset import *
from .config import *
from .loss import *
from .interpolation import test_circle_sampling

import numpy as np
from tqdm import tqdm
from PIL import Image


def train_vae(model, train_loader, cfg, is_cvae=False):
    """
    Train your VAE with one epoch.

    Inputs:
    - base_model: Your VAE instance.
    - train_loader: A tf.images.Dataset of MNIST dataset.
    - cfg: All arguments.
    - is_cvae: A boolean flag for Conditional-VAE. If your base_model is a Conditional-VAE,
    set is_cvae=True. If it's a Vanilla-VAE, set is_cvae=False.

    Returns:
    - total_loss: Sum of loss values of all batches.
    """
    mean_loss = 0
    n_prev = 0
    n_curr = 0
    n_batches = len(train_loader)
    pbar = tqdm(total=n_batches)
    total_loss = 0
    for n_batch, (data, labels) in enumerate(train_loader):
        if is_cvae:
            one_hot_labels = one_hot_labels(labels, 10)
        with tf.GradientTape() as tape:
            output = model(data, one_hot_labels) if is_cvae else model(data)
            loss = loss_function(output[0], data, output[1], output[2])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss.numpy()
        pbar.update()
        n_curr += data.shape[0]
        mean_loss = mean_loss * (n_prev / n_curr) + loss.numpy() * (data.shape[0] / n_curr)
        n_prev += data.shape[0]
        pbar.set_postfix_str(f"[batch {n_batch + 1}/{n_batches}]"
                             f"[batch loss: {loss:.2f}]"
                             f"[epoch loss: {mean_loss:.2f}]"
                             )
    return total_loss


def main(cfg):
    # Load MNIST dataset
    train_dataset = load_mnist(cfg.batch_size)

    # Get an instance of VAE
    if cfg.is_cvae:
        model = CVAE(cfg.input_size, latent_size=cfg.latent_size)
    else:
        model = VAE(cfg.input_size, latent_size=cfg.latent_size)

    # Load trained weights
    if cfg.load_weights:
        model = load_weights(model, cfg.is_cvae)

    # Train VAE
    for epoch_id in range(cfg.num_epochs):
        total_loss = train_vae(model, train_dataset, cfg, is_cvae=cfg.is_cvae)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss/len(train_dataset):.6f}")

    # Visualize results
    if cfg.visualize:
        if cfg.is_cvae:
            show_cvae_images(model, cfg.latent_size)
        else:
            show_vae_images(model, cfg.latent_size)
            show_vae_interpolation(model, cfg.latent_size)

    # Optional: Save VAE/CVAE base_model for debugging/testing.
    if cfg.save_weights:
        save_model_weights(model, cfg)


def test_sampling(cfg):
    model = get_model(cfg)
    train_dataset = load_mnist(cfg.batch_size)
    for data, _ in train_dataset:
        test_circle_sampling(model, data[:2], "image")
        break


def test_unseen(cfg):
    model = get_model(cfg)
    img_path = "../doc/unseen.png"
    img = Image.open(img_path).convert('L')
    img.thumbnail((28, 28), Image.ANTIALIAS)  # gray
    img = np.array(img).astype(np.float32) / 255.
    img = img[np.newaxis, np.newaxis, ...]
    img = tf.convert_to_tensor(img)
    test_circle_sampling(model, img, "image")


if __name__ == "__main__":
    config = parse_arguments()
    # main(config)
    test_unseen(config)
