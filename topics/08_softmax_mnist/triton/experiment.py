"""Run the scalar MNIST protocol (validation search, retraining, one test evaluation) on Triton."""

from ml_scratch.pure.softmax_mnist import experiment as protocol

from .model import SoftmaxRegression, byte_images


def run_experiment(directory=None, seed=protocol.SEED, **options):
    return protocol.run_with(SoftmaxRegression, byte_images, directory, seed, **options)


def main():
    protocol.main(run_experiment, "Triton")


if __name__ == "__main__":
    main()
