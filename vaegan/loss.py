import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error, binary_crossentropy, BinaryCrossentropy, Reduction
from tensorflow import sigmoid


@tf.function
def generator_content_loss(real, fake, content_model):

    bce_fn = BinaryCrossentropy(
        from_logits=True,
        reduction=Reduction.SUM,
    )
    real_o = content_model(real)
    fake_o = content_model(fake)
    # real_o, fake_o = sigmoid(real_o), sigmoid(fake_o)
    # loss = bce_fn(real_o, fake_o) * real_o.shape[-1] / real_o.shape[0]
    loss = tf.reduce_mean(mean_absolute_error(real_o, fake_o))
    return loss


@tf.function
def generator_discriminator_loss(x, expect_true=True):
    label = tf.ones_like(x) if expect_true else tf.zeros_like(x)
    return tf.reduce_mean(binary_crossentropy(label, x, from_logits=True))


@tf.function
def generator_disguise_discriminator(x):
    return tf.reduce_mean(binary_crossentropy(tf.ones_like(x), x, from_logits=True))


@tf.function
def discriminator_loss(real, fake):
    l_real = binary_crossentropy(tf.ones_like(real), real, from_logits=True)
    l_fake = binary_crossentropy(tf.zeros_like(fake), fake, from_logits=True)
    return tf.reduce_mean(l_real), tf.reduce_mean(l_fake)


@tf.function
def kl_loss(mu, logvar):
    loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1))
    return loss
