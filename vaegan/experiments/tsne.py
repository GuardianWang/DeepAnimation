from vaegan.model import VAE
from vaegan.dataset import make_dataset
from vaegan.experiments.path_sampling import load_model

try:
    from cuml.manifold import TSNE
except:
    from sklearn.manifold import TSNE

import numpy as np
from tqdm import tqdm
from pathlib import Path
from joblib import dump, load
import matplotlib.pyplot as plt


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


def tsne_fit(latents=None, load_tsne=True, save_tsne=True, save_transformed=True):
    if latents is None:
        latents = np.load('latents/latents.npy')
    if load_tsne:
        tsne = load('latents/tsne.joblib')
    else:
        tsne = TSNE()
        tsne.fit(latents)
        if save_tsne:
            dump(tsne, 'latents/tsne.joblib')

    latents = tsne.embedding_
    if save_transformed:
        np.save('latents/tsne_latents.npy', latents)
    return latents


def draw_tsne(latents=None):
    if latents is None:
        latents = np.load('latents/tsne_latents.npy')
    # 10 frames each gif
    labels = np.repeat(np.arange(latents.shape[0] // 10), 10)
    plt.scatter(latents[:, 0], latents[:, 1], s=3, c=labels)
    plt.colorbar()
    plt.title("t-SNE visualization of embeddings")
    plt.axis('off')
    plt.savefig('latents/vis.png', dpi=500)
    plt.show()


if __name__ == '__main__':
    frame_dir = "../../pngs"
    vae = VAE(512)
    load_model(vae, name='vae', epoch=2751, batch=0)
    batch_size = 128
    ds = make_dataset(frame_dir, batch_size, shuffle=False, fmt='*.png')
    z = get_latents(vae, ds)
    z = tsne_fit(latents=z, load_tsne=False)
    draw_tsne(z)
