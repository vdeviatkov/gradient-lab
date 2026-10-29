"""Run the scalar protocol (checks, step-size sweep, spiral training) on the NumPy backend."""

from ml_scratch.pure.backpropagation import experiment as protocol

from . import gradient_check
from .model import FeedForwardNetwork


def run_experiment(seed=protocol.SEED, epochs=protocol.EPOCHS):
    return protocol.run_with(FeedForwardNetwork, gradient_check, seed, epochs)


def main():
    protocol.main(run_experiment, "NumPy")


if __name__ == "__main__":
    main()
