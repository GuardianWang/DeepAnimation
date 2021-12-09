from vaegan.model import VAE
from vaegan.utils import load_weights
from vaegan.dataset import make_dataset

try:
    from cuml.manifold import TSNE
except:
    from sklearn.manifold import TSNE

import numpy as np
from tqdm import tqdm
from pathlib import Path


def get_latents(model: VAE, ds, save=True):
    latents = []
    for data in tqdm(ds):
        _, mu, _ = model.encode(data[0], training=False)
        latents.append(mu.numpy())

    latents = np.concatenate(latents, 0)
    if save:
        p = Path('latents')
        p.mkdir(parents=True, exist_ok=True)
        p = p / 'latents.npy'
        np.save(str(p), latents)
    return latents


def tsne_fit(latents=None, load=False):
    if load:
        latents = np.load('latents/latents.npy')


if __name__ == '__main__':
    frame_dir = "../../pngs"
    vae = VAE(512)
    batch_size = 128
    ds = make_dataset(frame_dir, batch_size, shuffle=False, fmt='*.png')
    get_latents(vae, ds)
