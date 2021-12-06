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

    # white is 0
    imgs = 0.5 * (-imgs + 1.)

    for i in range(n_imgs):
        plt.subplot(4, 4, i + 1)
        plt.imshow(imgs[i].numpy())
        plt.axis('off')

    if save:
        p = Path("vaegan_outputs")
        p.mkdir(exist_ok=True)
        p = p / f"epoch_{kwargs['epoch']:04d}_batch_{kwargs['batch']:04d}"
        plt.savefig(str(p))
    if plot:
        plt.show()


if __name__ == "__main__":
    from vaegan.model import VAEGAN
    vaegan = VAEGAN()
    vis_generate_images(vaegan)
