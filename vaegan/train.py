from vaegan.model import VAEGAN, VGG
from vaegan.dataset import load_test_dataset
from vaegan.loss import *

import tensorflow as tf

from tqdm import tqdm, trange


@tf.function
def train_batch(model: VAEGAN, data, content_model):
    diff = data[0] - data[1]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = model.generate(x=data[1], m=diff)
        d_real = model.discriminate(data[0])
        d_fake = model.discriminate(fake_images)

        g_c = generator_content_loss(data[0], fake_images, content_model)
        g_d = generator_disguise_discriminator(d_fake)
        d_loss = discriminator_loss(d_real, d_fake)
        g_loss = g_c + g_d  # be within tape

    grad_g = gen_tape.gradient(g_loss, model.generator_encoder_trainable_vars)
    grad_d = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.optim_g.apply_gradients(zip(grad_g, model.generator_encoder_trainable_vars))
    model.optim_d.apply_gradients((zip(grad_d, model.discriminator.trainable_variables)))

    losses = dict()
    losses["g_content"] = g_c
    losses["g_discriminator"] = g_d
    losses["d"] = d_loss

    return losses


def train_epoch(model, data, content_model):
    pbar = tqdm(total=len(data))
    for batch_data in data:
        losses = train_batch(model, batch_data, content_model)
        losses = {k: v.numpy() for k, v in losses.items()}
        pbar.update()
        pbar.set_postfix(losses)


def train(model, data, content_model):
    n_epochs = 100
    for i in trange(n_epochs):
        train_epoch(model, data, content_model)


if __name__ == "__main__":
    from pathlib import Path
    from vaegan.dataset import make_dataset
    frame_dir = r"C:/Users/zichu/Downloads/icons/target_svg/pngs"
    p = Path(frame_dir)
    ds = make_dataset()
    vaegan = VAEGAN()
    vgg = VGG()
    images = load_test_dataset()
    train(vaegan, ds, vgg)
