import argparse
import constants
from network.network import RedditNetwork
from util.set_seed import set_seed

def main():

    set_seed(seed=constants.SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    network = RedditNetwork(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size)

    network.train()


if __name__ == '__main__':
    main()