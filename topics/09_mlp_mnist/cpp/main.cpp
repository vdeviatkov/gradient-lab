#include "ml_scratch/mnist.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/softmax_regression.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

// Identical to experiment 08 so the two models can be compared directly.
constexpr std::uint32_t seed = 20260911;
constexpr std::size_t class_count = 10;
constexpr std::size_t validation_size = 6'000;
constexpr std::size_t search_epochs = 5;
constexpr std::size_t final_epochs = 25;
constexpr std::size_t batch_size = 64;

struct Candidate {
    std::size_t hidden_units;
    ml_scratch::Activation activation;
    double learning_rate;
    const char* name;
};

std::string mnist_directory() {
    if (const char* override_path = std::getenv("ML_SCRATCH_MNIST_DIR")) {
        return override_path;
    }
    return "data/mnist";
}

void print_confusion(const std::vector<std::vector<std::size_t>>& confusion) {
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

std::vector<ml_scratch::DenseLayer> build_layers(const Candidate& candidate,
                                                 const std::size_t feature_count) {
    return {{feature_count, candidate.hidden_units, candidate.activation},
            {candidate.hidden_units, class_count, ml_scratch::Activation::identity}};
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

    const std::size_t train_size = training_data.samples.size() - validation_size;
    const ml_scratch::LabeledDataset train{training_data.samples.begin(),
                                           training_data.samples.begin() +
                                               static_cast<std::ptrdiff_t>(train_size)};
    const ml_scratch::LabeledDataset validation{training_data.samples.begin() +
                                                    static_cast<std::ptrdiff_t>(train_size),
                                                training_data.samples.end()};
    const ml_scratch::LabeledDataset& test = test_data.samples;
    const std::size_t feature_count = training_data.rows * training_data.columns;

    std::cout << std::fixed << std::setprecision(4) << "C++ MLP on MNIST (seed=" << seed << ")\n"
              << "same data, preprocessing, splits and seed as experiment 08, so the softmax "
                 "baseline below is measured under identical conditions\n"
              << "splits: train=" << train.size() << ", validation=" << validation.size()
              << ", test=" << test.size() << "\n\n";

    const std::vector<Candidate> candidates{
        {64, ml_scratch::Activation::rectified_linear, 0.10, "64 ReLU,  lr 0.10"},
        {64, ml_scratch::Activation::hyperbolic_tangent, 0.10, "64 tanh,  lr 0.10"},
        {128, ml_scratch::Activation::rectified_linear, 0.10, "128 ReLU, lr 0.10"},
        {128, ml_scratch::Activation::rectified_linear, 0.50, "128 ReLU, lr 0.50"},
    };

    std::cout << "architecture search on the validation split (" << search_epochs
              << " epochs, batch=" << batch_size << ")\n";
    Candidate best = candidates.front();
    double best_validation = -1.0;
    for (const Candidate& candidate : candidates) {
        ml_scratch::NetworkTrainingConfig config;
        config.learning_rate = candidate.learning_rate;
        config.max_epochs = search_epochs;
        config.batch_size = batch_size;
        config.seed = seed;

        ml_scratch::FeedForwardNetwork network{build_layers(candidate, feature_count),
                                               ml_scratch::Loss::softmax_cross_entropy, seed};
        const auto result = network.fit(train, config);
        const double validation_accuracy = network.accuracy(validation);
        std::cout << "  " << candidate.name << ": parameters=" << std::setw(6)
                  << network.parameter_count() << ", train loss=" << result.loss_per_epoch.back()
                  << ", validation accuracy=" << validation_accuracy << '\n';
        if (validation_accuracy > best_validation) {
            best_validation = validation_accuracy;
            best = candidate;
        }
    }
    std::cout << "  selected " << best.name << " by validation accuracy\n\n";

    // The epoch loop is driven here rather than inside fit, so validation accuracy can be measured
    // after every epoch and the best model checkpointed. Each epoch advances the shuffle seed.
    ml_scratch::FeedForwardNetwork network{build_layers(best, feature_count),
                                           ml_scratch::Loss::softmax_cross_entropy, seed};
    const std::string checkpoint_path = "data/mlp_mnist_best.checkpoint";
    double best_epoch_validation = -1.0;
    std::size_t best_epoch = 0;
    std::vector<double> loss_history;

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t epoch = 1; epoch <= final_epochs; ++epoch) {
        ml_scratch::NetworkTrainingConfig config;
        config.learning_rate = best.learning_rate;
        config.max_epochs = 1;
        config.batch_size = batch_size;
        config.seed = static_cast<std::uint32_t>(seed + epoch);

        const auto result = network.fit(train, config);
        loss_history.push_back(result.loss_per_epoch.back());
        const double validation_accuracy = network.accuracy(validation);
        if (validation_accuracy > best_epoch_validation) {
            best_epoch_validation = validation_accuracy;
            best_epoch = epoch;
            network.save(checkpoint_path);
        }
        if (epoch == 1 || epoch % 5 == 0) {
            std::cout << "  epoch " << std::setw(2) << epoch
                      << ": training loss=" << result.loss_per_epoch.back()
                      << ", validation accuracy=" << validation_accuracy << '\n';
        }
    }
    const auto finish = std::chrono::steady_clock::now();
    const double training_seconds = std::chrono::duration<double>(finish - start).count();
    std::cout << "  best validation accuracy " << best_epoch_validation << " at epoch "
              << best_epoch << ", checkpointed to " << checkpoint_path << '\n'
              << "  wall-clock training time=" << std::setprecision(2) << training_seconds << " s ("
              << training_seconds / static_cast<double>(final_epochs) << " s per epoch)\n\n"
              << std::setprecision(4);

    // Restoring the checkpoint is how the evaluated model is obtained, so the round trip is
    // exercised for real rather than only in a unit test.
    const auto restored = ml_scratch::FeedForwardNetwork::load(checkpoint_path);
    const bool checkpoint_matches =
        restored.parameters() == network.parameters() || best_epoch != final_epochs;
    const double restored_validation = restored.accuracy(validation);

    ml_scratch::SoftmaxTrainingConfig linear_config;
    linear_config.learning_rate = 0.5;
    linear_config.max_epochs = 30;
    linear_config.batch_size = batch_size;
    linear_config.seed = seed;
    ml_scratch::SoftmaxRegression linear{feature_count, class_count};
    static_cast<void>(linear.fit(train, linear_config));

    const auto mlp_confusion = restored.confusion_matrix(test);
    const auto linear_confusion = linear.confusion_matrix(test);
    const auto mlp_metrics = ml_scratch::multiclass_metrics(mlp_confusion);
    const auto linear_metrics = ml_scratch::multiclass_metrics(linear_confusion);

    std::cout << "held-out results (test split evaluated once, from the restored checkpoint)\n"
              << "  softmax regression (" << std::setw(6) << linear.parameter_count()
              << " params): test accuracy=" << linear_metrics.accuracy
              << ", macro F1=" << linear_metrics.macro_f1 << '\n'
              << "  MLP                (" << std::setw(6) << restored.parameter_count()
              << " params): test accuracy=" << mlp_metrics.accuracy
              << ", macro F1=" << mlp_metrics.macro_f1 << '\n'
              << "  restored checkpoint validation accuracy=" << restored_validation
              << " (best during training " << best_epoch_validation << ")\n"
              << "  error rate fell from " << 1.0 - linear_metrics.accuracy << " to "
              << 1.0 - mlp_metrics.accuracy << ", a "
              << 100.0 * (1.0 - (1.0 - mlp_metrics.accuracy) / (1.0 - linear_metrics.accuracy))
              << "% reduction\n\n";

    std::cout << "per-digit test F1\n  digit   linear      MLP\n";
    for (std::size_t label = 0; label < class_count; ++label) {
        std::cout << "  " << std::setw(5) << label << std::setw(9)
                  << linear_metrics.per_class_f1[label] << std::setw(9)
                  << mlp_metrics.per_class_f1[label] << '\n';
    }

    std::cout << "\nMLP test confusion matrix\n";
    print_confusion(mlp_confusion);

    // The linear model's structured failures are the specific thing a hidden layer should fix.
    const std::vector<std::pair<std::size_t, std::size_t>> watched{{5, 3}, {4, 9}, {7, 9}, {8, 3}};
    std::cout << "\nconfusions that limited the linear model\n";
    for (const auto& [actual, predicted] : watched) {
        std::cout << "  " << actual << " predicted as " << predicted << ": linear "
                  << linear_confusion[actual][predicted] << ", MLP "
                  << mlp_confusion[actual][predicted] << '\n';
    }

    const bool mlp_beat_linear = mlp_metrics.accuracy > linear_metrics.accuracy;
    const bool checkpoint_restored = restored_validation == best_epoch_validation;
    const bool loss_decreased = loss_history.back() < loss_history.front();

    if (!mlp_beat_linear || !checkpoint_restored || !checkpoint_matches || !loss_decreased) {
        std::cerr << "experiment failed its criteria: MLP accuracy=" << mlp_metrics.accuracy
                  << ", linear accuracy=" << linear_metrics.accuracy
                  << ", restored validation=" << restored_validation << '\n';
        return 1;
    }
    return 0;
}
