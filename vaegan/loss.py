import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error, binary_crossentropy


def generator_content_loss(real, fake, content_model):
    real_o = content_model(real)
    fake_o = content_model(fake)
    loss = tf.reduce_mean([mean_absolute_error(x1, x2) for x1, x2 in zip(real_o, fake_o)])
    return loss


def generator_disguise_discriminator(x):
    return tf.reduce_mean(binary_crossentropy(tf.ones_like(x), x, from_logits=True))


def discriminator_loss(real, fake):
    l_real = binary_crossentropy(tf.ones_like(real), real, from_logits=True)
    l_fake = binary_crossentropy(tf.zeros_like(fake), fake, from_logits=True)
    return tf.reduce_mean(l_real + l_fake)
