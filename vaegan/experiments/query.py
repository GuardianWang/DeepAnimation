from vaegan.model import VAE
from vaegan.dataset import make_dataset
from vaegan.visualize import plot_by_paths
from vaegan.experiments.path_sampling import load_model

from pathlib import Path
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from joblib import dump, load
from sklearn.metrics import pairwise_distances
import re


def hash(data):
    # ahash
    ave = tf.reduce_mean(data, axis=1, keepdims=True)
    h = data >= ave
    return h.numpy()


def hash_images(model, ds, frame_dir, fmt):
    """Hash all png images and make a dataset"""
    p = Path(frame_dir)
    files = np.array(list(map(str, p.glob(fmt))))
    hashes = []
    for data in tqdm(ds):
        _, mu, _ = model.encode(data[0], training=False)
        h = hash(mu)
        hashes.append(h)
    hashes = np.concatenate(hashes, 0)
    all_data = {
        'hash': hashes,
        'files': files
    }
    dump(all_data, 'latents/hash.joblib')

    return all_data


def get_query(model: VAE, image):
    """image to hash"""
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image, [112, 112])
    image = image * (1. / 127.5) - 1.
    image = tf.expand_dims(image, 0)
    _, mu, _ = model.encode(image, training=False)
    return hash(mu)


def nearest(query, data, k=1):
    """Gets indices of closest images in dataset"""
    metric = 'hamming'
    d = pairwise_distances(query, data, metric=metric, n_jobs=-1)[0]
    top = np.argpartition(d, k)[:k]
    # true rank
    d = pairwise_distances(query, data[top], metric=metric)[0]
    top = top[np.argsort(d)]
    return top


def nearest_png_files(query, k=1):
    """Get a list of file paths to closest images in dataset"""
    dataset = load('latents/hash.joblib')
    top = nearest(query, dataset['hash'], k)
    files = dataset['files'][top]

    return files


def nearest_img_files(files, fmt='png'):
    """Get a list of file paths to closest images (type as specified) in dataset """
    def file_png2svg(x):
        p = Path(x)
        n = p.name
        n = re.sub(r"\.png", ".svg", n)
        p = p.parent.parent
        p = p / 'transformed_svgs' / n
        return str(p)

    def file_png2gif(x):
        p = Path(x)
        n = p.name
        n = re.sub(r"-\d+\.png", ".gif", n)
        p = p.parent.parent
        p = p / 'icons' / 'icon_gif' / n
        return str(p)

    if fmt == 'png':
        pass
    elif fmt == 'svg':
        files = list(map(lambda x: file_png2svg(x), files))
    elif fmt == 'gif':
        files = list(map(lambda x: file_png2gif(x), files))

    return files


def img2dataset(img, model, k=1, fmt='png'):
    """
    Given an image tensor, return the top k similar image path.
    :param fmt: "svg" or 'png' or 'gif'
    """
    img = get_query(model, img)
    files = nearest_png_files(img, k)
    files = nearest_img_files(files, fmt=fmt)
    return files


def test_nearest_dataset(model: VAE):
    # load a test png image
    # frame_dir = "../../pngs"
    frame_dir = r"C:\Users\zichu\Downloads"
    ds = make_dataset(frame_dir, batch_size=1, fmt='zoom_draw.png')
    img = tf.squeeze((ds.take(1).as_numpy_iterator().next()[0] + 1.) * 127.5, 0)
    files = img2dataset(img, model, k=10)
    plot_by_paths(files)


def load_vae():
    frame_dir = "../../pngs"
    vae = VAE(512)
    load_model(vae, name='vae', epoch=2751, batch=0)
    return vae


if __name__ == '__main__':
    batch_size = 128
    fmt = '*-0.png'
    vae = load_vae()
    # ds = make_dataset(frame_dir, batch_size, shuffle=False, fmt=fmt)

    # hash_images(vae, ds, frame_dir, fmt)
    test_nearest_dataset(vae)
