from vaegan.model import VAE
from vaegan.utils import load_weights
from vaegan.dataset import make_dataset
from vae.interpolation import circle_sampling, save_img

import tensorflow as tf


def load_model(model, input_shape=[1, 112, 112, 3], path='../', **kwargs):
    load_weights(model, input_shape, path=path, **kwargs)


def slerp(z1, z2, n):
    """
    https://github.com/JeremyCCHsu/tf-vaegan/blob/7f782c8cfeaa6bebf041812f3e2a32f16407534c/model/vaegan.py#L267
    """

    norm1 = tf.math.l2_normalize(z1, 1)
    norm2 = tf.math.l2_normalize(z2, 1)
    theta = tf.expand_dims(tf.einsum("ij,ij->i", norm1, norm2), -1)
    z1 = tf.repeat(z1, [n], axis=0)
    z2 = tf.repeat(z2, [n], axis=0)

    a = tf.reshape(tf.linspace(0., 1., n), [1, n])

    sin_inv = 1. / tf.sin(theta)
    a1 = tf.reshape(tf.sin((1. - a) * theta) * sin_inv, [-1, 1]) 
    a2 = tf.reshape(tf.sin(a * theta) * sin_inv, [-1, 1])
    z = a1 * z1 + a2 * z2

    return z


def test_sampling(model: VAE, data: tf.Tensor, img_folder='./images', sample_func='slerp'):
    """
    Tests circle sampler.
    """
    batch_size = data.shape[0]
    sample_rate = 50

    latent, _, _ = model.encode(data, training=False)

    if sample_func == 'circle':
        radius = 40
        ones = tf.random.normal([1, model.latent_size // 2])
        zeros = tf.zeros([1, model.latent_size // 2])
        center_vec = tf.concat([ones, zeros], -1)
        dir_vec = tf.concat([zeros, ones], -1)
        latent = circle_sampling(latent, radius, center_vec, dir_vec, sample_rate)
    elif sample_func == 'slerp':
        latent = slerp(latent[:-1], latent[1:], sample_rate)
        batch_size -= 1
    else:
        raise NotImplementedError

    # [batch_size * sample_rate, h, w, c]
    samples = (0.5 * (model.decode(latent, training=False) + 1.)).numpy()
    save_img(samples, batch_size, img_folder, is_gray=False)


if __name__ == '__main__':
    frame_dir = "../../pngs"
    vae = VAE(512)
    ds = make_dataset(frame_dir, 3)
    data = next(ds.take(1).as_numpy_iterator())[0]
    load_model(vae, name='vae', epoch=2751, batch=0)
    test_sampling(vae, data)

