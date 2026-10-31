"""Run the scalar MLP-versus-softmax MNIST protocol with the PyTorch network and softmax model."""

from ml_scratch.pure.mlp_mnist import experiment as protocol
from ml_scratch.pytorch.softmax_mnist.model import SoftmaxRegression, byte_images

from .model import ClassificationNetwork


def run_experiment(directory=None, seed=protocol.SEED, **options):
    options.setdefault("checkpoint", protocol.default_checkpoint("pytorch"))
    return protocol.run_with(
        ClassificationNetwork, SoftmaxRegression, byte_images, directory, seed, **options
    )


def main():
    protocol.main(run_experiment, "PyTorch", "pytorch")


if __name__ == "__main__":
    main()
