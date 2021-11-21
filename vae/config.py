import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_cvae", action="store_true")
    parser.add_argument("--load_weights", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--latent_size", type=int, default=128)
    parser.add_argument("--input_size", type=int, default=28*28)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args([])

    return args
