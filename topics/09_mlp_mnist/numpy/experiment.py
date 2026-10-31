"""Run the scalar MLP-versus-softmax MNIST protocol with the NumPy network and softmax model."""

from ml_scratch.numpy.softmax_mnist.model import SoftmaxRegression, byte_images
from ml_scratch.pure.mlp_mnist import experiment as protocol

from .model import ClassificationNetwork


def run_experiment(directory=None, seed=protocol.SEED, **options):
    options.setdefault("checkpoint", protocol.default_checkpoint("numpy"))
    return protocol.run_with(
        ClassificationNetwork, SoftmaxRegression, byte_images, directory, seed, **options
    )


def main():
    protocol.main(run_experiment, "NumPy", "numpy")


if __name__ == "__main__":
    main()
