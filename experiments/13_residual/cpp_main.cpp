#include "ml_scratch/cifar10.hpp"
#include "ml_scratch/convolution.hpp"
#include "ml_scratch/mnist.hpp"
#include "ml_scratch/neural_network.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260916;
constexpr std::size_t class_count = 10;

// Part one: a plain against a residual stack of identical shape, on a small MNIST subset.
constexpr std::size_t mlp_width = 64;
constexpr std::size_t mlp_train = 5'000;
constexpr std::size_t mlp_validation = 1'000;
constexpr std::size_t mlp_epochs = 10;

// Part two: a small convolutional network on CIFAR-10.
constexpr std::size_t cifar_train = 5'000;
constexpr std::size_t cifar_validation = 1'000;
constexpr std::size_t cifar_test = 2'000;
constexpr std::size_t cifar_epochs = 12;
constexpr std::size_t filters = 8;

std::string data_directory(const char* variable, const std::string& fallback) {
    if (const char* override_path = std::getenv(variable)) {
        return override_path;
    }
    return fallback;
}

ml_scratch::NetworkTrainingConfig training_config(const std::size_t epochs) {
    ml_scratch::NetworkTrainingConfig config;
    config.optimizer.kind = ml_scratch::OptimizerKind::adam;
    config.learning_rate = 0.001;
    config.batch_size = 32;
    config.max_epochs = epochs;
    config.seed = seed;
    return config;
}

// The squared norm of the first layer's gradient, which is what a deep plain stack loses.
double first_layer_gradient_norm(const ml_scratch::FeedForwardNetwork& network,
                                 const ml_scratch::LabeledDataset& dataset) {
    const auto gradient = network.gradient(dataset);
    const auto& first = network.layers().front();
    const std::size_t count = first.input_size * first.output_size + first.output_size;
    double total = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        total += gradient[index] * gradient[index];
    }
    return std::sqrt(total);
}

enum class Variant { plain, residual, residual_normalized };

std::string_view variant_name(const Variant variant) {
    switch (variant) {
    case Variant::plain:
        return "plain stacks";
    case Variant::residual:
        return "residual stacks, no normalization";
    case Variant::residual_normalized:
        return "residual stacks with layer normalization";
    }
    return "";
}

std::vector<ml_scratch::DenseLayer> build_stack(const std::size_t features, const std::size_t depth,
                                                const Variant variant) {
    const bool residual = variant != Variant::plain;
    const auto normalization = variant == Variant::residual_normalized
                                   ? ml_scratch::Normalization::layer
                                   : ml_scratch::Normalization::none;
    std::vector<ml_scratch::DenseLayer> layers{
        {features, mlp_width, ml_scratch::Activation::rectified_linear}};
    for (std::size_t index = 0; index < depth; ++index) {
        layers.push_back({mlp_width, mlp_width, ml_scratch::Activation::rectified_linear,
                          normalization, 0.0, residual});
    }
    layers.push_back({mlp_width, class_count, ml_scratch::Activation::identity});
    return layers;
}

struct StackReport {
    std::size_t depth;
    std::size_t parameters;
    double initial_gradient_norm;
    double train_loss;
    double validation_accuracy;
    double seconds;
};

struct ConvReport {
    std::string label;
    std::size_t parameters;
    double train_loss;
    double train_accuracy;
    double validation_accuracy;
    double test_accuracy;
    double seconds;
};

void print_stack_header() {
    std::cout << "  depth  parameters   initial |grad|   train loss   validation   seconds\n";
}

void print_stack(const StackReport& report) {
    std::cout << "  " << std::setw(5) << report.depth << std::setw(12) << report.parameters
              << std::setw(17) << std::scientific << std::setprecision(3)
              << report.initial_gradient_norm << std::defaultfloat << std::fixed
              << std::setprecision(4) << std::setw(14) << report.train_loss << std::setw(13)
              << report.validation_accuracy << std::setw(10) << std::setprecision(1)
              << report.seconds << std::setprecision(4) << '\n';
}

void print_conv_header() {
    std::cout << "  model                  parameters   train loss    train   valid.     test   "
                 "seconds\n";
}

void print_conv(const ConvReport& report) {
    std::string label = report.label;
    label.resize(21, ' ');
    std::cout << "  " << label << std::setw(12) << report.parameters << std::setw(13)
              << report.train_loss << std::setw(9) << report.train_accuracy << std::setw(9)
              << report.validation_accuracy << std::setw(9) << report.test_accuracy << std::setw(10)
              << std::setprecision(1) << report.seconds << std::setprecision(4) << '\n';
}

} // namespace

int main() {
    // ---------- Part one: an identity skip isolated in an MLP ----------
    const std::string mnist = data_directory("ML_SCRATCH_MNIST_DIR", "data/mnist");
    ml_scratch::LabeledDataset mlp_training;
    ml_scratch::LabeledDataset mlp_held_out;
    std::size_t features = 0;
    try {
        const auto data =
            ml_scratch::load_mnist(mnist + "/train-images-idx3-ubyte",
                                   mnist + "/train-labels-idx1-ubyte", mlp_train + mlp_validation);
        mlp_training.assign(data.samples.begin(),
                            data.samples.begin() + static_cast<std::ptrdiff_t>(mlp_train));
        mlp_held_out.assign(data.samples.begin() + static_cast<std::ptrdiff_t>(mlp_train),
                            data.samples.end());
        features = data.rows * data.columns;
    } catch (const std::exception& error) {
        std::cerr << "could not load MNIST from " << mnist << ": " << error.what() << "\n\n"
                  << "Download it first: ./scripts/download_mnist.sh\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(4) << "C++ residual connections (seed=" << seed
              << ")\n\n"
              << "part one: an identity skip isolated in an MLP\n"
              << "  stacks of " << mlp_width << "-unit ReLU layers on " << mlp_training.size()
              << " MNIST images, Adam at 0.001, " << mlp_epochs << " epochs\n"
              << "  the three groups differ only in whether each middle layer computes g(Wx + b),\n"
              << "  g(Wx + b + x), or the same with the pre-activation normalized first\n\n";

    const std::vector<std::size_t> depths{2, 5, 10, 20};
    std::vector<StackReport> plain_stacks;
    std::vector<StackReport> residual_stacks;
    std::vector<StackReport> normalized_stacks;

    for (const Variant variant :
         {Variant::plain, Variant::residual, Variant::residual_normalized}) {
        std::cout << "  " << variant_name(variant) << '\n';
        print_stack_header();
        for (const std::size_t depth : depths) {
            ml_scratch::FeedForwardNetwork network{build_stack(features, depth, variant),
                                                   ml_scratch::Loss::softmax_cross_entropy, seed};
            // Measured before any training, so it reflects the architecture rather than the fit.
            const double initial = first_layer_gradient_norm(network, mlp_held_out);

            const auto start = std::chrono::steady_clock::now();
            const auto result = network.fit(mlp_training, training_config(mlp_epochs));
            const auto finish = std::chrono::steady_clock::now();

            const StackReport report{depth,
                                     network.parameter_count(),
                                     initial,
                                     result.loss_per_epoch.back(),
                                     network.accuracy(mlp_held_out),
                                     std::chrono::duration<double>(finish - start).count()};
            print_stack(report);
            switch (variant) {
            case Variant::plain:
                plain_stacks.push_back(report);
                break;
            case Variant::residual:
                residual_stacks.push_back(report);
                break;
            case Variant::residual_normalized:
                normalized_stacks.push_back(report);
                break;
            }
        }
        std::cout << '\n';
    }

    // ---------- Part two: convolutional residual blocks on CIFAR-10 ----------
    const std::string cifar = data_directory("ML_SCRATCH_CIFAR10_DIR", "data/cifar10");
    ml_scratch::LabeledDataset cifar_training;
    ml_scratch::LabeledDataset cifar_held_out;
    ml_scratch::LabeledDataset cifar_evaluation;
    try {
        const auto training =
            ml_scratch::load_cifar10({cifar + "/data_batch_1.bin", cifar + "/data_batch_2.bin"},
                                     cifar_train + cifar_validation);
        const auto evaluation = ml_scratch::load_cifar10({cifar + "/test_batch.bin"}, cifar_test);
        cifar_training.assign(training.samples.begin(),
                              training.samples.begin() + static_cast<std::ptrdiff_t>(cifar_train));
        cifar_held_out.assign(training.samples.begin() + static_cast<std::ptrdiff_t>(cifar_train),
                              training.samples.end());
        cifar_evaluation = evaluation.samples;
    } catch (const std::exception& error) {
        std::cerr << "could not load CIFAR-10 from " << cifar << ": " << error.what() << "\n\n"
                  << "Download it first (the data is not committed):\n"
                  << "    ./scripts/download_cifar10.sh\n"
                  << "or point ML_SCRATCH_CIFAR10_DIR at a directory holding the .bin batches.\n";
        return 1;
    }

    std::cout << "part two: convolutional residual blocks on CIFAR-10\n"
              << "  " << cifar_training.size() << " training, " << cifar_held_out.size()
              << " validation and " << cifar_evaluation.size()
              << " test images at 3x32x32, Adam at 0.001, " << cifar_epochs << " epochs\n"
              << "  the plain and residual networks have identical shapes and parameter counts; "
                 "only the skip differs\n"
              << "  the convolutional network has no normalization, so part one predicts the bare "
                 "skip should help at\n"
              << "  small depth and stop helping as the blocks accumulate\n\n";

    // A stem that halves the resolution, then `blocks` shape-preserving convolutions per stage,
    // with a pooling step between the stages.
    const auto build_convolutional = [](const std::size_t blocks, const bool residual) {
        std::vector<ml_scratch::ConvLayerSpec> layers{
            ml_scratch::convolution(filters, 3, ml_scratch::Activation::rectified_linear, 2, 1)};
        const auto add_stage = [&] {
            for (std::size_t index = 0; index < blocks; ++index) {
                layers.push_back(
                    residual ? ml_scratch::residual_convolution(
                                   filters, 3, ml_scratch::Activation::rectified_linear)
                             : ml_scratch::convolution(
                                   filters, 3, ml_scratch::Activation::rectified_linear, 1, 1));
            }
        };
        add_stage();
        layers.push_back(ml_scratch::max_pooling(2));
        add_stage();
        layers.push_back(ml_scratch::max_pooling(2));
        layers.push_back(ml_scratch::flatten());
        layers.push_back(ml_scratch::dense(class_count, ml_scratch::Activation::identity));
        return layers;
    };

    const auto run_convolutional = [&](const std::string& label, const std::size_t blocks,
                                       const bool residual) {
        ml_scratch::ConvolutionalNetwork network{{3, 32, 32},
                                                 build_convolutional(blocks, residual),
                                                 ml_scratch::Loss::softmax_cross_entropy,
                                                 seed};
        const auto start = std::chrono::steady_clock::now();
        const auto result = network.fit(cifar_training, training_config(cifar_epochs));
        const auto finish = std::chrono::steady_clock::now();
        return ConvReport{label,
                          network.parameter_count(),
                          result.loss_per_epoch.back(),
                          network.accuracy(cifar_training),
                          network.accuracy(cifar_held_out),
                          network.accuracy(cifar_evaluation),
                          std::chrono::duration<double>(finish - start).count()};
    };

    print_conv_header();
    const ConvReport shallow_plain = run_convolutional("plain, 2 per stage", 2, false);
    print_conv(shallow_plain);
    const ConvReport shallow_residual = run_convolutional("residual, 2 per stage", 2, true);
    print_conv(shallow_residual);
    const ConvReport deep_plain = run_convolutional("plain, 5 per stage", 5, false);
    print_conv(deep_plain);
    const ConvReport deep_residual = run_convolutional("residual, 5 per stage", 5, true);
    print_conv(deep_residual);

    const double chance = 1.0 / static_cast<double>(class_count);
    const StackReport& plain_deepest = plain_stacks.back();
    const StackReport& residual_deepest = residual_stacks.back();
    const StackReport& normalized_deepest = normalized_stacks.back();

    std::cout << "\nsummary\n"
              << "  at depth " << plain_deepest.depth
              << " the plain stack's first-layer gradient was " << std::scientific
              << std::setprecision(3) << plain_deepest.initial_gradient_norm << " and the residual "
              << "stack's " << residual_deepest.initial_gradient_norm << std::defaultfloat
              << std::fixed << std::setprecision(4) << '\n'
              << "  the deepest plain stack reached validation accuracy "
              << plain_deepest.validation_accuracy << ", the unnormalized residual one "
              << residual_deepest.validation_accuracy << ", and the normalized residual one "
              << normalized_deepest.validation_accuracy << '\n'
              << "  on CIFAR-10 the skip was worth "
              << shallow_residual.test_accuracy - shallow_plain.test_accuracy
              << " test accuracy at 2 blocks per stage and "
              << deep_residual.test_accuracy - deep_plain.test_accuracy
              << " at 5, the same reversal part one shows without normalization (chance " << chance
              << ")\n";

    const bool deep_plain_stack_degraded =
        plain_deepest.validation_accuracy < plain_stacks.front().validation_accuracy;
    const bool skip_preserved_gradients =
        residual_deepest.initial_gradient_norm > plain_deepest.initial_gradient_norm;
    // An identity skip alone does not rescue depth: it replaces a vanishing gradient with a
    // growing one, because each block adds its input back and the activations compound. The skip
    // only pays off once something holds the scale steady, which is why a residual block in
    // practice always carries a normalization.
    const bool bare_skip_overshot =
        residual_deepest.initial_gradient_norm > 100.0 * plain_deepest.initial_gradient_norm;
    const bool normalization_rescued_depth =
        normalized_deepest.validation_accuracy > plain_deepest.validation_accuracy &&
        normalized_deepest.validation_accuracy > residual_deepest.validation_accuracy;
    const bool identical_parameter_counts = deep_plain.parameters == deep_residual.parameters &&
                                            shallow_plain.parameters == shallow_residual.parameters;
    // Part one established that a bare skip helps until the compounding identity path takes over.
    // The convolutional halves of this experiment carry no normalization either, so the same
    // reversal is what should be observed: an advantage at two blocks per stage that is gone by
    // five. Asserting the reversal, rather than asserting that residual always wins, is what makes
    // this a prediction the run can falsify.
    const bool skip_helped_when_shallow =
        shallow_residual.test_accuracy > shallow_plain.test_accuracy;
    const bool advantage_reversed_with_depth =
        deep_residual.test_accuracy - deep_plain.test_accuracy <
        shallow_residual.test_accuracy - shallow_plain.test_accuracy;
    const bool everything_beat_chance =
        deep_residual.test_accuracy > chance && shallow_residual.test_accuracy > chance;

    if (!deep_plain_stack_degraded || !skip_preserved_gradients || !bare_skip_overshot ||
        !normalization_rescued_depth || !identical_parameter_counts || !skip_helped_when_shallow ||
        !advantage_reversed_with_depth || !everything_beat_chance) {
        std::cerr << "experiment failed its criteria: plain deep validation="
                  << plain_deepest.validation_accuracy
                  << ", residual deep validation=" << residual_deepest.validation_accuracy
                  << ", plain CIFAR test=" << deep_plain.test_accuracy
                  << ", residual CIFAR test=" << deep_residual.test_accuracy << '\n';
        return 1;
    }
    return 0;
}
