import matplotlib.pyplot as plt
from pathlib import Path


def vis_generate_images(model, data_input=None, save=True, plot=False, **kwargs):
    fig_imgs_sz = [4, 4]
    n_imgs = fig_imgs_sz[0] * fig_imgs_sz[1]
    if data_input is not None:
        n_imgs = min(n_imgs, data_input.shape[0])
        imgs = model.generate(x=data_input[:n_imgs], training=False)
    else:
        imgs = model.random_generate(n_imgs)

    imgs = 0.5 * (imgs + 1.)
    vis_save(imgs.numpy(), fig_imgs_sz[0], fig_imgs_sz[1], save, "vaegan_outputs", plot, **kwargs)


def vis_save(imgs, sub_r, sub_c, save=True, save_dir="", plot=False, **kwargs):

    fig, axes = plt.subplots(sub_r, sub_c)
    for i in range(sub_r):
        for j in range(sub_c):
            ax = axes[i][j]
            ax.imshow(imgs[i])
            plt.axis('off')

    if save:
        p = Path(save_dir)
        p.mkdir(exist_ok=True)
        p = p / f"epoch_{kwargs['epoch']:04d}_batch_{kwargs['batch']:04d}"
        plt.savefig(str(p))
    if plot:
        plt.show()

    plt.close(fig)


def vis_vae_images(model, data_input, save=True, plot=False, **kwargs):
    fig_imgs_sz = [4, 4]
    n_imgs = fig_imgs_sz[0] * fig_imgs_sz[1]
    n_imgs = min(n_imgs, data_input.shape[0])
    data_input = data_input[:n_imgs]
    is_image = len(data_input.shape) == 4

    if is_image:
        imgs = model(data_input, training=False)
    else:  # latent vector
        imgs = model.decode(data_input, training=False)

    imgs = 0.5 * (imgs + 1.)
    vis_save(imgs.numpy(), fig_imgs_sz[0], fig_imgs_sz[1], save, "vae_outputs", plot, **kwargs)


if __name__ == "__main__":
    from vaegan.model import VAEGAN
    vaegan = VAEGAN()
    vis_generate_images(vaegan)
