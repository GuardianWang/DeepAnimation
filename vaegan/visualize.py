import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io


def vis_generate_images(model, data_input=None, save=True, plot=False, **kwargs):
    fig_imgs_sz = [4, 4]
    n_imgs = fig_imgs_sz[0] * fig_imgs_sz[1] // 2
    if data_input is not None:
        n_imgs = min(n_imgs, data_input.shape[0])
        data_input = data_input[:n_imgs]
        imgs = model.generate(x=data_input, training=False)
        imgs = merge_img_pairs(data_input, imgs)
    else:
        imgs = model.random_generate(n_imgs)
        imgs = 0.5 * (imgs + 1.)

    vis_save(imgs, fig_imgs_sz[0], fig_imgs_sz[1], save, "vaegan_outputs", plot, **kwargs)


def vis_save(imgs, sub_r, sub_c, save=True, save_dir="", plot=False, **kwargs):
    n_imgs = imgs.shape[0]
    fig, axes = plt.subplots(sub_r, sub_c)
    for n in range(n_imgs):
        i = n // sub_c
        j = n % sub_c
        ax = axes[i][j]
        ax.imshow(imgs[n])
        ax.axis('off')

    if save:
        p = Path(save_dir)
        p.mkdir(exist_ok=True)
        p = p / f"epoch_{kwargs['epoch']:04d}_batch_{kwargs['batch']:04d}"
        plt.savefig(str(p))
    if 'writer' in kwargs:
        writer = kwargs['writer']
        tensorboard_img = plot_to_image(fig)
        with writer.as_default():
            tf.summary.image("Training data", tensorboard_img, step=kwargs['epoch'])
    if plot:
        plt.show()

    plt.close(fig)


def vis_vae_images(model, data_input, save=True, plot=False, **kwargs):
    fig_imgs_sz = [4, 4]
    n_imgs = fig_imgs_sz[0] * fig_imgs_sz[1] // 2
    n_imgs = min(n_imgs, data_input.shape[0])
    data_input = data_input[:n_imgs]
    is_image = len(data_input.shape) == 4

    if is_image:
        imgs = model(data_input, training=False)
    else:  # latent vector
        imgs = model.decode(data_input, training=False)

    imgs = merge_img_pairs(data_input, imgs)
    vis_save(imgs, fig_imgs_sz[0], fig_imgs_sz[1], save, "vae_outputs", plot, **kwargs)


def merge_img_pairs(input_data, output_data):
    x = tf.concat((input_data, output_data), 0)
    x = 0.5 * (x + 1.)
    return x.numpy()


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


if __name__ == "__main__":
    from vaegan.model import VAEGAN
    vaegan = VAEGAN()
    vis_generate_images(vaegan)
