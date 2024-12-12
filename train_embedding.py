import argparse
import constants
from model.embedding import EmbeddingType
from model.loss import LossType
from network.network import RedditNetwork
from util.set_seed import set_seed

def main():

    set_seed(seed=constants.SEED)

    parser = argparse.ArgumentParser()

    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--loss_type", type=int, default=1)

    parser.add_argument("--embedding_type", type=int, default=1)
    parser.add_argument("--negative_sample", type=int, default=5)

    parser.add_argument("--hidden_feats", type=int, default=128)

    args = parser.parse_args()

    network = RedditNetwork(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        # load=args.load_model,
        loss_type=LossType(args.loss_type),
        embedding_type=EmbeddingType(args.embedding_type),
        negative_samples=args.negative_sample,
        hidden_feats=args.hidden_feats,
    )

    network.train()


if __name__ == '__main__':
    main()