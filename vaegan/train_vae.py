from vaegan.model import VAE
from vaegan.dataset import load_toy_dataset, make_dataset
from vaegan.loss import *
from vaegan.visualize import *
from vaegan.utils import *

import tensorflow as tf

from tqdm import tqdm, trange


def train_batch(model: VAE, data):
    with tf.GradientTape() as e_tape:
        latent_img, mu_img, logvar_img = model.encode(data[0])
        reconstruct_img = model.decode(latent_img)

        e_kl = kl_loss(mu_img, logvar_img)
        e_c = vae_content_loss(data[0], reconstruct_img)
        e_loss = e_kl + e_c

    grad_e = e_tape.gradient(e_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grad_e, model.trainable_variables))

    losses = dict()
    losses["e_kl"] = e_kl
    losses["e_content"] = e_c

    return losses


def train_epoch(model, data, **kwargs):
    pbar = tqdm(total=len(data))
    for i, batch_data in enumerate(data):
        losses = train_batch(model, batch_data)

        losses = {k: v.numpy() for k, v in losses.items()}
        losses.update(kwargs["epoch_info"])
        pbar.update()
        pbar.set_postfix(losses)

        if i % 20 == 0:
            vis_vae_images(model, batch_data[1],
                           epoch=kwargs["epoch_info"]["cur_epoch"], batch=i + 1)


def train(data_path):
    vae = VAE(latent_size=512)
    data = make_dataset(data_path)
    n_epochs = 100
    # load_weights(vae, [1, 112, 112, 3], name='vae', epoch=0, batch=0)
    for i in trange(n_epochs):
        epoch_info = {'cur_epoch': i + 1}
        train_epoch(vae, data, epoch_info=epoch_info)
        save_weights(vae, name="vae", epoch=0, batch=0)


if __name__ == "__main__":
    frame_dir = r"C:/Users/zichu/Downloads/icons/target_svg/pngs"
    # images = load_toy_dataset()
    train(frame_dir)
