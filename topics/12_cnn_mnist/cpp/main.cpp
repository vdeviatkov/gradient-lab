#include "ml_scratch/convolution.hpp"
#include "ml_scratch/mnist.hpp"
#include "ml_scratch/neural_network.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260915;
constexpr std::size_t class_count = 10;
constexpr std::size_t train_size = 10'000;
constexpr std::size_t validation_size = 2'000;
constexpr std::size_t test_size = 5'000;
constexpr std::size_t epochs = 15;
constexpr std::size_t batch_size = 32;
constexpr double learning_rate = 0.001;

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
    std::size_t rows;
    std::size_t columns;
};

// Moves every image by (dy, dx), filling the vacated border with zeros. Nothing is retrained: the
// point is to see how a model already fitted on centered digits copes with a small displacement.
ml_scratch::LabeledDataset shift(const ml_scratch::LabeledDataset& dataset, const std::size_t rows,
                                 const std::size_t columns, const int dy, const int dx) {
    ml_scratch::LabeledDataset shifted;
    shifted.reserve(dataset.size());
    for (const auto& sample : dataset) {
        std::vector<double> moved(sample.features.size(), 0.0);
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const int source_row = static_cast<int>(row) - dy;
                const int source_column = static_cast<int>(column) - dx;
                if (source_row < 0 || source_row >= static_cast<int>(rows) || source_column < 0 ||
                    source_column >= static_cast<int>(columns)) {
                    continue;
                }
                moved[row * columns + column] =
                    sample.features[static_cast<std::size_t>(source_row) * columns +
                                    static_cast<std::size_t>(source_column)];
            }
        }
        shifted.push_back({std::move(moved), sample.label});
    }
    return shifted;
}

struct Report {
    std::string label;
    std::size_t parameters;
    double train_accuracy;
    double validation_accuracy;
    double test_accuracy;
    double shifted_accuracy;
    double seconds;
};

void print_header() {
    std::cout << "  model                     parameters    train     val.     test  shifted  "
                 "seconds\n";
}

void print_report(const Report& report) {
    std::string label = report.label;
    label.resize(24, ' ');
    std::cout << "  " << label << std::setw(12) << report.parameters << std::setw(9)
              << report.train_accuracy << std::setw(9) << report.validation_accuracy << std::setw(9)
              << report.test_accuracy << std::setw(9) << report.shifted_accuracy << std::setw(9)
              << std::setprecision(1) << report.seconds << std::setprecision(4) << '\n';
}

ml_scratch::NetworkTrainingConfig training_config() {
    ml_scratch::NetworkTrainingConfig config;
    config.optimizer.kind = ml_scratch::OptimizerKind::adam;
    config.learning_rate = learning_rate;
    config.max_epochs = epochs;
    config.batch_size = batch_size;
    config.seed = seed;
    return config;
}

Report run_dense(const std::string& label, const Splits& splits,
                 const ml_scratch::LabeledDataset& shifted,
                 const std::vector<ml_scratch::DenseLayer>& layers) {
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, seed};
    const auto start = std::chrono::steady_clock::now();
    static_cast<void>(network.fit(splits.train, training_config()));
    const auto finish = std::chrono::steady_clock::now();
    return {label,
            network.parameter_count(),
            network.accuracy(splits.train),
            network.accuracy(splits.validation),
            network.accuracy(splits.test),
            network.accuracy(shifted),
            std::chrono::duration<double>(finish - start).count()};
}

Report run_convolutional(const std::string& label, const Splits& splits,
                         const ml_scratch::LabeledDataset& shifted,
                         const std::vector<ml_scratch::ConvLayerSpec>& layers) {
    ml_scratch::ConvolutionalNetwork network{
        {1, splits.rows, splits.columns}, layers, ml_scratch::Loss::softmax_cross_entropy, seed};
    const auto start = std::chrono::steady_clock::now();
    static_cast<void>(network.fit(splits.train, training_config()));
    const auto finish = std::chrono::steady_clock::now();
    return {label,
            network.parameter_count(),
            network.accuracy(splits.train),
            network.accuracy(splits.validation),
            network.accuracy(splits.test),
            network.accuracy(shifted),
            std::chrono::duration<double>(finish - start).count()};
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
        splits.rows = training_data.rows;
        splits.columns = training_data.columns;
    } catch (const std::exception& error) {
        std::cerr << "could not load MNIST from " << directory << ": " << error.what() << "\n\n"
                  << "Download it first (the data is not committed):\n"
                  << "    ./scripts/download_mnist.sh\n"
                  << "or point ML_SCRATCH_MNIST_DIR at a directory holding the uncompressed IDX "
                     "files.\n";
        return 1;
    }

    const auto shifted_test = shift(splits.test, splits.rows, splits.columns, 2, 2);
    const std::size_t pixels = splits.rows * splits.columns;

    std::cout << std::fixed << std::setprecision(4)
              << "C++ convolutional network on MNIST (seed=" << seed << ")\n"
              << "every model uses Adam at " << learning_rate << ", batch " << batch_size << ", "
              << epochs << " epochs, and the same seed, so only the architecture differs\n"
              << "splits: train=" << splits.train.size()
              << ", validation=" << splits.validation.size() << ", test=" << splits.test.size()
              << "\n"
              << "the `shifted` column evaluates the same trained model on the test images moved "
                 "two pixels down and right\n\n";

    std::cout << "architectures\n";
    print_header();
    const Report linear = run_dense("linear softmax", splits, shifted_test,
                                    {{pixels, class_count, ml_scratch::Activation::identity}});
    print_report(linear);

    const Report mlp = run_dense("MLP 128 ReLU", splits, shifted_test,
                                 {{pixels, 128, ml_scratch::Activation::rectified_linear},
                                  {128, class_count, ml_scratch::Activation::identity}});
    print_report(mlp);

    const Report cnn =
        run_convolutional("CNN 8c3-16c3, max pool", splits, shifted_test,
                          {ml_scratch::convolution(8, 3, ml_scratch::Activation::rectified_linear),
                           ml_scratch::max_pooling(2),
                           ml_scratch::convolution(16, 3, ml_scratch::Activation::rectified_linear),
                           ml_scratch::max_pooling(2), ml_scratch::flatten(),
                           ml_scratch::dense(class_count, ml_scratch::Activation::identity)});
    print_report(cnn);

    std::cout << "\nablations on the convolutional network\n";
    print_header();
    const Report average =
        run_convolutional("average pooling", splits, shifted_test,
                          {ml_scratch::convolution(8, 3, ml_scratch::Activation::rectified_linear),
                           ml_scratch::average_pooling(2),
                           ml_scratch::convolution(16, 3, ml_scratch::Activation::rectified_linear),
                           ml_scratch::average_pooling(2), ml_scratch::flatten(),
                           ml_scratch::dense(class_count, ml_scratch::Activation::identity)});
    print_report(average);

    const Report no_pooling = run_convolutional(
        "no pooling, stride 2", splits, shifted_test,
        {ml_scratch::convolution(8, 3, ml_scratch::Activation::rectified_linear, 2),
         ml_scratch::convolution(16, 3, ml_scratch::Activation::rectified_linear, 2),
         ml_scratch::flatten(), ml_scratch::dense(class_count, ml_scratch::Activation::identity)});
    print_report(no_pooling);

    const Report single =
        run_convolutional("one 8-filter layer", splits, shifted_test,
                          {ml_scratch::convolution(8, 3, ml_scratch::Activation::rectified_linear),
                           ml_scratch::max_pooling(2), ml_scratch::flatten(),
                           ml_scratch::dense(class_count, ml_scratch::Activation::identity)});
    print_report(single);

    const double cnn_drop = cnn.test_accuracy - cnn.shifted_accuracy;
    const double mlp_drop = mlp.test_accuracy - mlp.shifted_accuracy;
    std::cout << "\nsummary\n"
              << "  the CNN matched or beat the MLP with " << std::setprecision(1)
              << static_cast<double>(mlp.parameters) / static_cast<double>(cnn.parameters)
              << "x fewer parameters (" << cnn.parameters << " against " << mlp.parameters << ")\n"
              << std::setprecision(4) << "  under a two-pixel shift the CNN lost " << cnn_drop
              << " accuracy and the MLP lost " << mlp_drop << '\n'
              << "  the CNN cost " << std::setprecision(1) << cnn.seconds / mlp.seconds
              << "x the MLP's training time for those parameters\n"
              << std::setprecision(4);

    const bool cnn_is_competitive = cnn.test_accuracy >= mlp.test_accuracy;
    const bool cnn_is_smaller = cnn.parameters * 5 < mlp.parameters;
    const bool cnn_shifts_better = cnn_drop < mlp_drop;
    const bool depth_helped = cnn.test_accuracy > single.test_accuracy;
    const bool everything_beat_linear = cnn.test_accuracy > linear.test_accuracy;

    if (!cnn_is_competitive || !cnn_is_smaller || !cnn_shifts_better || !depth_helped ||
        !everything_beat_linear) {
        std::cerr << "experiment failed its criteria: cnn=" << cnn.test_accuracy
                  << ", mlp=" << mlp.test_accuracy << ", single=" << single.test_accuracy
                  << ", linear=" << linear.test_accuracy << ", cnn drop=" << cnn_drop
                  << ", mlp drop=" << mlp_drop << '\n';
        return 1;
    }
    return 0;
}
