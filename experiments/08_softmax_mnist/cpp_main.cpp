#include "ml_scratch/mnist.hpp"
#include "ml_scratch/softmax_regression.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260911;
constexpr std::size_t class_count = 10;
constexpr std::size_t validation_size = 6'000;
constexpr std::size_t sweep_epochs = 8;
constexpr std::size_t final_epochs = 30;

struct Candidate {
    double learning_rate;
    double l2_regularization;
};

std::string mnist_directory() {
    if (const char* override_path = std::getenv("ML_SCRATCH_MNIST_DIR")) {
        return override_path;
    }
    return "data/mnist";
}

void print_confusion(const ml_scratch::ConfusionCounts& confusion) {
    std::cout << "  actual \\ predicted";
    for (std::size_t label = 0; label < class_count; ++label) {
        std::cout << std::setw(6) << label;
    }
    std::cout << '\n';
    for (std::size_t actual = 0; actual < class_count; ++actual) {
        std::cout << "  " << std::setw(17) << actual;
        for (std::size_t predicted = 0; predicted < class_count; ++predicted) {
            std::cout << std::setw(6) << confusion[actual][predicted];
        }
        std::cout << '\n';
    }
}

} // namespace

int main() {
    const std::string directory = mnist_directory();
    ml_scratch::MnistSplit training_data;
    ml_scratch::MnistSplit test_data;
    try {
        training_data = ml_scratch::load_mnist(directory + "/train-images-idx3-ubyte",
                                               directory + "/train-labels-idx1-ubyte");
        test_data = ml_scratch::load_mnist(directory + "/t10k-images-idx3-ubyte",
                                           directory + "/t10k-labels-idx1-ubyte");
    } catch (const std::exception& error) {
        std::cerr << "could not load MNIST from " << directory << ": " << error.what() << "\n\n"
                  << "Download it first (the data is not committed):\n"
                  << "    ./scripts/download_mnist.sh\n"
                  << "or point ML_SCRATCH_MNIST_DIR at a directory holding the uncompressed IDX "
                     "files.\n";
        return 1;
    }

    if (training_data.samples.size() <= validation_size) {
        std::cerr << "training split is too small to hold out a validation set\n";
        return 1;
    }
    // The last samples of the official training file become the validation split; the official
    // 10000-image test file is touched exactly once, at the end.
    const std::size_t train_size = training_data.samples.size() - validation_size;
    const ml_scratch::LabeledDataset train{training_data.samples.begin(),
                                           training_data.samples.begin() +
                                               static_cast<std::ptrdiff_t>(train_size)};
    const ml_scratch::LabeledDataset validation{training_data.samples.begin() +
                                                    static_cast<std::ptrdiff_t>(train_size),
                                                training_data.samples.end()};
    const ml_scratch::LabeledDataset& test = test_data.samples;
    const std::size_t feature_count = training_data.rows * training_data.columns;

    std::cout << "C++ softmax regression on MNIST (seed=" << seed << ")\n"
              << "images " << training_data.rows << "x" << training_data.columns << " = "
              << feature_count << " features, " << class_count
              << " classes; pixels rescaled from 0..255 into [0, 1]\n"
              << "splits: train=" << train.size() << ", validation=" << validation.size()
              << ", test=" << test.size() << '\n';

    const auto counts = ml_scratch::class_counts(validation, class_count);
    std::cout << "validation class counts:";
    for (const std::size_t count : counts) {
        std::cout << ' ' << count;
    }
    std::cout << "\n\n";

    std::cout << std::fixed << std::setprecision(4)
              << "hyperparameter search on the validation split (" << sweep_epochs
              << " epochs, batch=64)\n";
    const std::vector<Candidate> candidates{
        {0.10, 0.0}, {0.10, 1e-4}, {0.50, 0.0}, {0.50, 1e-4}, {0.50, 1e-3}, {1.00, 1e-4},
    };

    Candidate best{0.0, 0.0};
    double best_validation = -1.0;
    for (const Candidate& candidate : candidates) {
        ml_scratch::SoftmaxTrainingConfig config;
        config.learning_rate = candidate.learning_rate;
        config.l2_regularization = candidate.l2_regularization;
        config.max_epochs = sweep_epochs;
        config.batch_size = 64;
        config.seed = seed;

        ml_scratch::SoftmaxRegression model{feature_count, class_count};
        const auto result = model.fit(train, config);
        const double validation_accuracy = model.accuracy(validation);
        std::cout << "  learning rate=" << std::setw(6) << candidate.learning_rate
                  << ", L2=" << std::setw(8) << candidate.l2_regularization
                  << ": train loss=" << result.loss_per_epoch.back()
                  << ", train accuracy=" << model.accuracy(train)
                  << ", validation accuracy=" << validation_accuracy << '\n';
        if (validation_accuracy > best_validation) {
            best_validation = validation_accuracy;
            best = candidate;
        }
    }
    std::cout << "  selected learning rate=" << best.learning_rate
              << ", L2=" << best.l2_regularization << " by validation accuracy\n\n";

    ml_scratch::SoftmaxTrainingConfig config;
    config.learning_rate = best.learning_rate;
    config.l2_regularization = best.l2_regularization;
    config.max_epochs = final_epochs;
    config.batch_size = 64;
    config.seed = seed;

    ml_scratch::SoftmaxRegression model{feature_count, class_count};
    const auto start = std::chrono::steady_clock::now();
    const auto result = model.fit(train, config);
    const auto finish = std::chrono::steady_clock::now();
    const double training_seconds = std::chrono::duration<double>(finish - start).count();

    std::cout << "final training run (" << final_epochs << " epochs, " << model.parameter_count()
              << " parameters)\n";
    for (std::size_t epoch :
         {std::size_t{1}, std::size_t{5}, std::size_t{10}, std::size_t{20}, final_epochs}) {
        std::cout << "  epoch " << std::setw(2) << epoch
                  << ": training loss=" << result.loss_per_epoch[epoch - 1] << '\n';
    }
    std::cout << "  wall-clock training time=" << std::setprecision(2) << training_seconds << " s ("
              << training_seconds / static_cast<double>(final_epochs) << " s per epoch)\n\n"
              << std::setprecision(4);

    // Everything above used train and validation only. The test split is evaluated here, once.
    const auto test_metrics = model.evaluate(test);
    const auto baseline_counts = ml_scratch::class_counts(train, class_count);
    std::size_t majority = 0;
    for (std::size_t label = 1; label < class_count; ++label) {
        if (baseline_counts[label] > baseline_counts[majority]) {
            majority = label;
        }
    }
    const auto test_counts = ml_scratch::class_counts(test, class_count);
    const double baseline_accuracy =
        static_cast<double>(test_counts[majority]) / static_cast<double>(test.size());

    std::cout << "held-out results\n"
              << "  majority-class baseline (always " << majority
              << ") : test accuracy=" << baseline_accuracy << '\n'
              << "  softmax regression                : train accuracy=" << model.accuracy(train)
              << ", validation accuracy=" << model.accuracy(validation)
              << ", test accuracy=" << test_metrics.accuracy << '\n'
              << "  macro precision=" << test_metrics.macro_precision
              << ", macro recall=" << test_metrics.macro_recall
              << ", macro F1=" << test_metrics.macro_f1 << "\n\n"
              << "per-class test recall\n";
    for (std::size_t label = 0; label < class_count; ++label) {
        std::cout << "  digit " << label
                  << ": precision=" << test_metrics.per_class_precision[label]
                  << ", recall=" << test_metrics.per_class_recall[label]
                  << ", F1=" << test_metrics.per_class_f1[label] << '\n';
    }

    const auto confusion = model.confusion_matrix(test);
    std::cout << "\ntest confusion matrix\n";
    print_confusion(confusion);

    std::size_t worst_actual = 0;
    std::size_t worst_predicted = 1;
    std::size_t worst_count = 0;
    for (std::size_t actual = 0; actual < class_count; ++actual) {
        for (std::size_t predicted = 0; predicted < class_count; ++predicted) {
            if (actual != predicted && confusion[actual][predicted] > worst_count) {
                worst_count = confusion[actual][predicted];
                worst_actual = actual;
                worst_predicted = predicted;
            }
        }
    }
    std::cout << "\nmost frequent confusion: " << worst_actual << " predicted as "
              << worst_predicted << ", " << worst_count << " times\n";

    const bool beat_baseline = test_metrics.accuracy > baseline_accuracy;
    // A linear model on raw MNIST pixels is well established around 92%; anything below 90% would
    // mean the optimizer or the preprocessing is broken rather than that the model is weak.
    const bool reached_expected_accuracy = test_metrics.accuracy >= 0.90;
    const bool balanced_across_digits = test_metrics.macro_f1 >= 0.90;
    const bool loss_decreased = result.loss_per_epoch.back() < result.loss_per_epoch.front();

    if (!beat_baseline || !reached_expected_accuracy || !balanced_across_digits ||
        !loss_decreased) {
        std::cerr << "experiment failed its criteria: test accuracy=" << test_metrics.accuracy
                  << ", macro F1=" << test_metrics.macro_f1 << ", baseline=" << baseline_accuracy
                  << '\n';
        return 1;
    }
    return 0;
}
