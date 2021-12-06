from vaegan.model import VAEGAN, VGG
from vaegan.dataset import load_toy_dataset, make_dataset
from vaegan.loss import *
from vaegan.visualize import *

import tensorflow as tf

from tqdm import tqdm, trange


# @tf.function
def train_batch(model: VAEGAN, data, content_model):
    # diff = data[0] - data[1]
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as e_tape:
        latent_img, mu_img, logvar_img = model.encode_image(data[1])
        fake_images = model.generate(x=latent_img, m=None)
        fake_images_z = model.random_generate(data[0].shape[0])

        d_real = model.discriminate(data[0])
        d_fake = model.discriminate(fake_images)
        d_fake_z = model.discriminate(fake_images_z)

        e_kl = kl_loss(mu_img, logvar_img)
        eg_c = generator_content_loss(data[0], fake_images, content_model)

        g_d = generator_discriminator_loss(d_fake, expect_true=True)
        g_d_z = generator_discriminator_loss(d_fake_z, expect_true=True)

        d_loss_real = generator_discriminator_loss(d_real, expect_true=True)
        d_loss_fake = generator_discriminator_loss(d_fake, expect_true=False)
        d_loss_fake_z = generator_discriminator_loss(d_fake_z, expect_true=False)

        e_loss = e_kl + eg_c
        g_loss = 1e-4 * eg_c + 0.5 * (g_d + g_d_z)  # be within tape
        d_loss = 0.5 * d_loss_real + 0.25 * (d_loss_fake + d_loss_fake_z)

    grad_e = e_tape.gradient(e_loss, model.encoder_trainable_params)
    grad_g = g_tape.gradient(g_loss, model.generator.trainable_variables)
    grad_d = d_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.optim_e.apply_gradients(zip(grad_e, model.encoder_trainable_params))
    model.optim_g.apply_gradients(zip(grad_g, model.generator.trainable_variables))
    model.optim_d.apply_gradients((zip(grad_d, model.discriminator.trainable_variables)))

    losses = dict()
    losses["e_kl"] = e_kl
    losses["eg_content"] = eg_c
    losses["g_dis"] = g_d
    losses["g_dis_z"] = g_d_z
    losses["d_real"] = d_loss_real
    losses["d_fake"] = d_loss_fake
    losses["d_fake_z"] = d_loss_fake_z

    return losses


def train_epoch(model, data, content_model, **kwargs):
    pbar = tqdm(total=len(data))
    for i, batch_data in enumerate(data):
        losses = train_batch(model, batch_data, content_model)

        losses = {k: v.numpy() for k, v in losses.items()}
        losses.update(kwargs["epoch_info"])
        pbar.update()
        pbar.set_postfix(losses)

        if i % 20 == 0:
            vis_generate_images(model, batch_data[1],
                                epoch=kwargs["epoch_info"]["cur_epoch"], batch=i + 1)


def train(data_path):
    vaegan = VAEGAN()
    vgg = VGG()
    data = make_dataset(data_path)
    n_epochs = 100
    for i in trange(n_epochs):
        epoch_info = {'cur_epoch': i + 1}
        train_epoch(vaegan, data, vgg, epoch_info=epoch_info)


if __name__ == "__main__":
    frame_dir = r"C:/Users/zichu/Downloads/icons/target_svg/pngs"
    # images = load_toy_dataset()
    train(frame_dir)
