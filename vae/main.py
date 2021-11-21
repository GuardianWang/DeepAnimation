from utils import *
from model import *
from dataset import *
from config import *
from loss import *
from interpolation import test_circle_sampling

from tqdm import tqdm


def train_vae(model, train_loader, args, is_cvae=False):
    """
    Train your VAE with one epoch.

    Inputs:
    - model: Your VAE instance.
    - train_loader: A tf.data.Dataset of MNIST dataset.
    - args: All arguments.
    - is_cvae: A boolean flag for Conditional-VAE. If your model is a Conditional-VAE,
    set is_cvae=True. If it's a Vanilla-VAE, set is_cvae=False.

    Returns:
    - total_loss: Sum of loss values of all batches.
    """
    mean_loss = 0
    mean_acc = 0
    n_prev = 0
    n_curr = 0
    n_batches = len(train_loader)
    pbar = tqdm(total=n_batches)
    total_loss = 0
    for n_batch, (data, labels) in enumerate(train_loader):
        if is_cvae:
            one_hot_labels = one_hot_labels(labels, 10)
        with tf.GradientTape() as tape:
            output = model(data, one_hot_labels) if is_cvae else model(data)
            loss = loss_function(output[0], data, output[1], output[2])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss.numpy()
        pbar.update()
        n_curr += data.shape[0]
        mean_loss = mean_loss * (n_prev / n_curr) + loss.numpy() * (data.shape[0] / n_curr)
        n_prev += data.shape[0]
        pbar.set_postfix_str(f"[batch {n_batch + 1}/{n_batches}]"
                             f"[batch loss: {loss:.2f}]"
                             f"[epoch loss: {mean_loss:.2f}]"
                             )
    return total_loss


def main(args):
    # Load MNIST dataset
    train_dataset = load_mnist(args.batch_size)

    # Get an instance of VAE
    if args.is_cvae:
        model = CVAE(args.input_size, latent_size=args.latent_size)
    else:
        model = VAE(args.input_size, latent_size=args.latent_size)

    # Load trained weights
    #if args.load_weights:
    #    model = load_weights(model)

    # Train VAE
    for epoch_id in range(args.num_epochs):
        total_loss = train_vae(model, train_dataset, args, is_cvae=args.is_cvae)
        print(f"Train Epoch: {epoch_id} \tLoss: {total_loss/len(train_dataset):.6f}")

    # sampling
    for data, _ in train_dataset:
        test_circle_sampling(model, data[:2], "image")
        break

    # Visualize results
    if args.is_cvae:
        show_cvae_images(model, args.latent_size)
    else:
        show_vae_images(model, args.latent_size)
        show_vae_interpolation(model, args.latent_size)

    # Optional: Save VAE/CVAE model for debugging/testing.
    save_model_weights(model, args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
