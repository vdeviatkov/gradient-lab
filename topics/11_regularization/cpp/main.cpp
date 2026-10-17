#include "ml_scratch/mnist.hpp"
#include "ml_scratch/neural_network.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260914;
constexpr std::size_t class_count = 10;
constexpr std::size_t hidden_units = 128;
// A deliberately small training split: 118k parameters on 2000 images overfits hard, which is the
// regime where these techniques are supposed to matter.
constexpr std::size_t train_size = 2'000;
constexpr std::size_t validation_size = 2'000;
constexpr std::size_t test_size = 5'000;
constexpr std::size_t epochs = 30;
constexpr std::size_t batch_size = 64;

std::string mnist_directory() {
    if (const char* override_path = std::getenv("ML_SCRATCH_MNIST_DIR")) {
        return override_path;
    }
    return "data/mnist";
}

struct Splits {
    ml_scratch::LabeledDataset train;
    ml_scratch::LabeledDataset validation;
    ml_scratch::LabeledDataset test;
    std::size_t feature_count;
};

struct Outcome {
    double train_loss;
    double train_accuracy;
    double validation_accuracy;
    double test_accuracy;
    std::size_t best_epoch;
    std::size_t epochs_run;
    double seconds;
};

std::vector<ml_scratch::DenseLayer> build_layers(const Splits& splits,
                                                 const ml_scratch::Normalization normalization,
                                                 const double dropout_rate) {
    return {{splits.feature_count, hidden_units, ml_scratch::Activation::rectified_linear,
             normalization, dropout_rate},
            {hidden_units, hidden_units, ml_scratch::Activation::rectified_linear, normalization,
             dropout_rate},
            {hidden_units, class_count, ml_scratch::Activation::identity}};
}

Outcome run(const Splits& splits, const ml_scratch::InitializationConfig& initialization,
            const ml_scratch::Normalization normalization, const double dropout_rate,
            const ml_scratch::RegularizationConfig& regularization, const double learning_rate,
            const std::size_t patience) {
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = learning_rate;
    config.max_epochs = epochs;
    config.batch_size = batch_size;
    config.seed = seed;
    config.regularization = regularization;
    config.early_stopping.patience = patience;

    ml_scratch::FeedForwardNetwork network{build_layers(splits, normalization, dropout_rate),
                                           ml_scratch::Loss::softmax_cross_entropy, seed,
                                           initialization};
    const auto start = std::chrono::steady_clock::now();
    const auto result = network.fit(splits.train, splits.validation, config);
    const auto finish = std::chrono::steady_clock::now();

    return {result.loss_per_epoch.back(),
            network.accuracy(splits.train),
            network.accuracy(splits.validation),
            network.accuracy(splits.test),
            result.best_epoch,
            result.epochs,
            std::chrono::duration<double>(finish - start).count()};
}

void print_row(const std::string& label, const Outcome& outcome, const std::size_t width = 28) {
    std::string padded = label;
    padded.resize(width, ' ');
    std::cout << "  " << padded << std::setw(12) << outcome.train_loss << std::setw(10)
              << outcome.train_accuracy << std::setw(12) << outcome.validation_accuracy
              << std::setw(10) << outcome.test_accuracy << std::setw(10)
              << outcome.train_accuracy - outcome.validation_accuracy << std::setw(9)
              << std::setprecision(1) << outcome.seconds << std::setprecision(4) << '\n';
}

void print_header(const std::string& first_column) {
    std::string padded = first_column;
    padded.resize(28, ' ');
    std::cout << "  " << padded << "  train loss  train val. accuracy     test      gap  seconds\n";
}

} // namespace

int main() {
    const std::string directory = mnist_directory();
    Splits splits;
    try {
        const auto training_data = ml_scratch::load_mnist(directory + "/train-images-idx3-ubyte",
                                                          directory + "/train-labels-idx1-ubyte",
                                                          train_size + validation_size);
        const auto test_data =
            ml_scratch::load_mnist(directory + "/t10k-images-idx3-ubyte",
                                   directory + "/t10k-labels-idx1-ubyte", test_size);
        splits.train.assign(training_data.samples.begin(),
                            training_data.samples.begin() +
                                static_cast<std::ptrdiff_t>(train_size));
        splits.validation.assign(training_data.samples.begin() +
                                     static_cast<std::ptrdiff_t>(train_size),
                                 training_data.samples.end());
        splits.test = test_data.samples;
        splits.feature_count = training_data.rows * training_data.columns;
    } catch (const std::exception& error) {
        std::cerr << "could not load MNIST from " << directory << ": " << error.what() << "\n\n"
                  << "Download it first (the data is not committed):\n"
                  << "    ./scripts/download_mnist.sh\n"
                  << "or point ML_SCRATCH_MNIST_DIR at a directory holding the uncompressed IDX "
                     "files.\n";
        return 1;
    }

    const ml_scratch::RegularizationConfig none{};
    std::cout << std::fixed << std::setprecision(4)
              << "C++ initialization, regularization and normalization on MNIST (seed=" << seed
              << ")\n"
              << "network " << splits.feature_count << " -> " << hidden_units << " -> "
              << hidden_units << " -> " << class_count << " with ReLU hidden layers\n"
              << "splits: train=" << splits.train.size()
              << ", validation=" << splits.validation.size() << ", test=" << splits.test.size()
              << "; " << epochs << " epochs, batch " << batch_size << "\n"
              << "the training split is deliberately small, so the unregularized network can and "
                 "does memorize it\n\n";

    std::cout << "weight initialization (plain training, learning rate 0.10)\n";
    print_header("scheme");
    const Outcome automatic = run(splits, {ml_scratch::Initialization::automatic, 0.0},
                                  ml_scratch::Normalization::none, 0.0, none, 0.10, 0);
    print_row("automatic (He for ReLU)", automatic);
    const Outcome glorot = run(splits, {ml_scratch::Initialization::glorot_uniform, 0.0},
                               ml_scratch::Normalization::none, 0.0, none, 0.10, 0);
    print_row("Glorot", glorot);
    const Outcome tiny = run(splits, {ml_scratch::Initialization::fixed_uniform, 0.001},
                             ml_scratch::Normalization::none, 0.0, none, 0.10, 0);
    print_row("fixed uniform 0.001", tiny);
    const Outcome large = run(splits, {ml_scratch::Initialization::fixed_uniform, 1.0},
                              ml_scratch::Normalization::none, 0.0, none, 0.10, 0);
    print_row("fixed uniform 1.0", large);
    const Outcome zeros = run(splits, {ml_scratch::Initialization::zeros, 0.0},
                              ml_scratch::Normalization::none, 0.0, none, 0.10, 0);
    print_row("all zeros", zeros);

    std::cout << "\nregularization (automatic initialization, learning rate 0.10)\n";
    print_header("penalty");
    const Outcome unregularized = automatic;
    print_row("none", unregularized);
    const Outcome l2_small =
        run(splits, {}, ml_scratch::Normalization::none, 0.0, {0.0, 1e-3}, 0.10, 0);
    print_row("L2 1e-3", l2_small);
    const Outcome l2_large =
        run(splits, {}, ml_scratch::Normalization::none, 0.0, {0.0, 1e-2}, 0.10, 0);
    print_row("L2 1e-2", l2_large);
    const Outcome l1_small =
        run(splits, {}, ml_scratch::Normalization::none, 0.0, {1e-4, 0.0}, 0.10, 0);
    print_row("L1 1e-4", l1_small);
    const Outcome dropout = run(splits, {}, ml_scratch::Normalization::none, 0.3, none, 0.10, 0);
    print_row("dropout 0.3", dropout);
    const Outcome dropout_heavy =
        run(splits, {}, ml_scratch::Normalization::none, 0.5, none, 0.10, 0);
    print_row("dropout 0.5", dropout_heavy);

    std::cout << "\nnormalization (automatic initialization)\n";
    print_header("scheme");
    print_row("none, lr 0.10", unregularized);
    const Outcome batch_norm =
        run(splits, {}, ml_scratch::Normalization::batch, 0.0, none, 0.10, 0);
    print_row("batch norm, lr 0.10", batch_norm);
    const Outcome layer_norm =
        run(splits, {}, ml_scratch::Normalization::layer, 0.0, none, 0.10, 0);
    print_row("layer norm, lr 0.10", layer_norm);
    // Normalization is supposed to widen the range of workable learning rates, so the same rate
    // that breaks a plain network is the interesting comparison.
    const Outcome plain_fast = run(splits, {}, ml_scratch::Normalization::none, 0.0, none, 1.00, 0);
    print_row("none, lr 1.00", plain_fast);
    const Outcome batch_fast =
        run(splits, {}, ml_scratch::Normalization::batch, 0.0, none, 1.00, 0);
    print_row("batch norm, lr 1.00", batch_fast);

    std::cout << "\nearly stopping (patience 5, automatic initialization, learning rate 0.10)\n";
    print_header("configuration");
    const Outcome stopped = run(splits, {}, ml_scratch::Normalization::none, 0.0, none, 0.10, 5);
    print_row("restored best epoch", stopped);
    std::cout << "  best epoch " << stopped.best_epoch << " of " << stopped.epochs_run
              << " run (budget " << epochs << "); without it the run would have kept the epoch-"
              << unregularized.epochs_run << " parameters at test accuracy "
              << unregularized.test_accuracy << '\n';

    const bool zeros_failed = zeros.test_accuracy < 0.2;
    const bool scaling_mattered = automatic.test_accuracy > large.test_accuracy &&
                                  automatic.test_accuracy > tiny.test_accuracy;
    const bool overfitting_happened =
        unregularized.train_accuracy > 0.99 &&
        unregularized.train_accuracy - unregularized.validation_accuracy > 0.05;
    const bool a_regularizer_closed_the_gap =
        std::min({l2_small.train_accuracy - l2_small.validation_accuracy,
                  l2_large.train_accuracy - l2_large.validation_accuracy,
                  l1_small.train_accuracy - l1_small.validation_accuracy,
                  dropout.train_accuracy - dropout.validation_accuracy,
                  dropout_heavy.train_accuracy - dropout_heavy.validation_accuracy}) <
        unregularized.train_accuracy - unregularized.validation_accuracy;
    const bool normalization_survived_the_large_rate =
        batch_fast.test_accuracy > plain_fast.test_accuracy;
    const bool early_stopping_selected_an_earlier_epoch = stopped.best_epoch < epochs;

    std::cout << "\nsummary\n"
              << "  zero initialization reached test accuracy " << zeros.test_accuracy
              << ", against " << automatic.test_accuracy << " for the automatic scheme\n"
              << "  the unregularized network reached train accuracy "
              << unregularized.train_accuracy << " against validation "
              << unregularized.validation_accuracy << ", a gap of "
              << unregularized.train_accuracy - unregularized.validation_accuracy << '\n'
              << "  the best regularizer on test accuracy was "
              << (dropout_heavy.test_accuracy > dropout.test_accuracy ? "dropout 0.5"
                                                                      : "dropout 0.3")
              << " at " << std::max(dropout.test_accuracy, dropout_heavy.test_accuracy)
              << ", against " << unregularized.test_accuracy << " unregularized\n"
              << "  at learning rate 1.00 a plain network scored " << plain_fast.test_accuracy
              << " and a batch-normalized one " << batch_fast.test_accuracy << '\n'
              << "  early stopping restored epoch " << stopped.best_epoch << " at test accuracy "
              << stopped.test_accuracy << ", against " << unregularized.test_accuracy
              << " for the full " << epochs << "-epoch run\n";

    if (!zeros_failed || !scaling_mattered || !overfitting_happened ||
        !a_regularizer_closed_the_gap || !normalization_survived_the_large_rate ||
        !early_stopping_selected_an_earlier_epoch) {
        std::cerr << "experiment failed its criteria: zeros=" << zeros.test_accuracy
                  << ", automatic=" << automatic.test_accuracy
                  << ", gap=" << unregularized.train_accuracy - unregularized.validation_accuracy
                  << ", plain fast=" << plain_fast.test_accuracy
                  << ", batch fast=" << batch_fast.test_accuracy
                  << ", best epoch=" << stopped.best_epoch << '\n';
        return 1;
    }
    return 0;
}
