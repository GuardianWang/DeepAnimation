from vaegan.model import VAEGAN, VGG
from vaegan.dataset import load_test_dataset
from vaegan.loss import *

import tensorflow as tf


def train_batch(model: VAEGAN, data, content_model):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = model.generate(x=data, m=data)
        d_real = model.discriminate(data)
        d_fake = model.discriminate(fake_images)

        g_c = generator_content_loss(data, fake_images, content_model)
        g_d = generator_disguise_discriminator(d_fake)
        d_loss = discriminator_loss(d_real, d_fake)
        g_loss = g_c + g_d  # be within tape

    grad_g = gen_tape.gradient(g_loss, model.generator_encoder_trainable_vars)
    grad_d = disc_tape.gradient(d_loss, model.discriminator.trainable_variables)
    model.optim_g.apply_gradients(zip(grad_g, model.generator_encoder_trainable_vars))
    model.optim_d.apply_gradients((zip(grad_d, model.discriminator.trainable_variables)))


def train_epoch(model, data, content_model):
    for batch_data in data:
        train_batch(model, batch_data, content_model)


if __name__ == "__main__":
    vaegan = VAEGAN()
    vgg = VGG()
    images = load_test_dataset()
    train_epoch(vaegan, images, vgg)
