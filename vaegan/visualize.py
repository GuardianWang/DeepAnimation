import matplotlib.pyplot as plt


def vis_generate_images(model, data_input=None):
    fig_imgs_sz = [4, 4]
    n_imgs = fig_imgs_sz[0] * fig_imgs_sz[1]
    if data_input is not None:
        n_imgs = min(n_imgs, data_input.shape[0])
        imgs = model.generate(x=data_input[:n_imgs], training=False)
    else:
        imgs = model.random_generate(n_imgs)
    fig = plt.figure()

    for i in range(n_imgs):
        plt.subplot(4, 4, i + 1)
        plt.imshow(imgs[i].numpy())
        plt.axis('off')

    plt.savefig(f'image_at_epoch.png')
    # plt.show()


if __name__ == "__main__":
    from vaegan.model import VAEGAN
    vaegan = VAEGAN()
    vis_generate_images(vaegan)
