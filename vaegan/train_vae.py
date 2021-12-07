import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpu, [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)])
    except RuntimeError as e:
        print(e)

from vaegan.model import VAE
from vaegan.dataset import load_toy_dataset, make_dataset
from vaegan.loss import *
from vaegan.visualize import *
from vaegan.utils import *
from vaegan.optimizers import *

from tqdm import tqdm, trange
from pathlib import Path
from datetime import datetime


@tf.function
def train_batch(model: VAE, data, **kwargs):
    opt = kwargs["optimizer"]
    with tf.GradientTape() as e_tape:
        latent_img, mu_img, logvar_img = model.encode(data[0])
        reconstruct_img = model.decode(latent_img)

        e_kl = kl_loss(mu_img, logvar_img)
        e_c = vae_content_loss(data[0], reconstruct_img)
        e_loss = e_kl + e_c

    grad_e = e_tape.gradient(e_loss, model.trainable_variables)
    opt.apply_gradients(zip(grad_e, model.trainable_variables))

    losses = dict()
    losses["e_kl"] = e_kl
    losses["e_content"] = e_c

    return losses


def train_epoch(model, data, **kwargs):
    pbar = tqdm(total=len(data))
    loss_handler = LossHandler()
    train_writer = kwargs['writer']
    for i, batch_data in enumerate(data):
        losses = train_batch(model, batch_data, **kwargs)

        losses = {k: v.numpy() for k, v in losses.items()}
        loss_handler.update(losses, batch_data[0].shape[0])
        losses.update(kwargs["epoch_info"])
        pbar.update()
        pbar.set_postfix(losses)

        if i % 50 == 0:
            vis_vae_images(model, batch_data[0],
                           epoch=kwargs["epoch"], batch=i + 1, writer=train_writer, plot=False)

    with train_writer.as_default():
        for k, v in loss_handler.items():
            tf.summary.scalar(k, v, step=kwargs['epoch'])


def train(data_path):
    vae = VAE(latent_size=512)
    data = make_dataset(data_path)
    n_epochs = 3000
    opt = get_adam()
    train_writer = get_writer()

    # load_weights(vae, [1, 112, 112, 3], name='vae', epoch=0, batch=0)
    for i in range(n_epochs):
        epoch_info = {'epoch': f"{i + 1}/{n_epochs}"}
        train_epoch(vae, data, epoch_info=epoch_info,
                    optimizer=opt, epoch=i + 1, writer=train_writer)

        if i % 50 == 0:
            save_weights(vae, name="vae", epoch=i + 1, batch=0)


def get_writer():
    p = Path("logs")
    p.mkdir(parents=True, exist_ok=True)
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    p = p / cur_time / "train"
    train_writer = tf.summary.create_file_writer(str(p))

    return train_writer


if __name__ == "__main__":
    frame_dir = "../pngs"
    # images = load_toy_dataset()
    train(frame_dir)
