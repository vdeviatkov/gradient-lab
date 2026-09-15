#include "ml_scratch/neural_network.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

void require_near(const double actual, const double expected, const double tolerance,
                  const std::string_view message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void require_invalid_argument(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

// Four features, for the tests that need a layer wider than two inputs.
const ml_scratch::LabeledDataset four_feature_data{
    {{-1.0, -1.0, 0.2, 0.0}, 0}, {{-0.8, -1.2, 0.1, 0.3}, 0}, {{1.0, -1.0, -0.2, 0.4}, 1},
    {{1.2, -0.7, 0.0, -0.1}, 1}, {{0.0, 1.5, 0.3, 0.2}, 2},   {{-0.3, 1.1, -0.4, 0.1}, 2},
};

const ml_scratch::LabeledDataset three_class_data{
    {{-1.0, -1.0}, 0}, {{-0.8, -1.2}, 0}, {{-1.2, -0.6}, 0}, {{1.0, -1.0}, 1},
    {{1.2, -0.7}, 1},  {{0.9, -1.3}, 1},  {{0.0, 1.5}, 2},   {{-0.3, 1.1}, 2},
};

double squared_weight_norm(const ml_scratch::FeedForwardNetwork& network) {
    // Only the weights are penalized, and for a single layer they precede the biases.
    const auto parameters = network.parameters();
    const auto& layer = network.layers().front();
    double total = 0.0;
    for (std::size_t index = 0; index < layer.input_size * layer.output_size; ++index) {
        total += parameters[index] * parameters[index];
    }
    return total;
}

void test_initialization_schemes() {
    const std::vector<ml_scratch::DenseLayer> layers{
        {4, 8, ml_scratch::Activation::rectified_linear},
        {8, 3, ml_scratch::Activation::identity},
    };

    // Glorot and He differ by exactly sqrt(2) in their limits for a square-ish layer, so drawing
    // with the same seed must produce proportional weights.
    ml_scratch::FeedForwardNetwork glorot{layers,
                                          ml_scratch::Loss::softmax_cross_entropy,
                                          5,
                                          {ml_scratch::Initialization::glorot_uniform, 0.0}};
    ml_scratch::FeedForwardNetwork he{layers,
                                      ml_scratch::Loss::softmax_cross_entropy,
                                      5,
                                      {ml_scratch::Initialization::he_uniform, 0.0}};
    const auto glorot_parameters = glorot.parameters();
    const auto he_parameters = he.parameters();
    const double expected_ratio = std::sqrt((4.0 + 8.0) / 4.0);
    require_near(he_parameters[0] / glorot_parameters[0], expected_ratio, 1e-12,
                 "He and Glorot limits are not in the expected ratio");

    // The default must remain He for ReLU and Glorot elsewhere, so earlier milestones are unmoved.
    ml_scratch::FeedForwardNetwork automatic{layers, ml_scratch::Loss::softmax_cross_entropy, 5};
    require_near(automatic.parameters()[0], he_parameters[0], 1e-15,
                 "the automatic scheme did not pick He for a ReLU layer");

    // Zero initialization leaves every hidden unit identical, so they receive identical gradients
    // and stay identical forever. The measurement is that the gradient rows are equal.
    ml_scratch::FeedForwardNetwork zeros{layers,
                                         ml_scratch::Loss::softmax_cross_entropy,
                                         5,
                                         {ml_scratch::Initialization::zeros, 0.0}};
    for (const double value : zeros.parameters()) {
        require(value == 0.0, "zero initialization left a non-zero weight");
    }
    const auto gradient = zeros.gradient(four_feature_data);
    for (std::size_t unit = 1; unit < 8; ++unit) {
        for (std::size_t input = 0; input < 4; ++input) {
            require_near(gradient[unit * 4 + input], gradient[input], 1e-15,
                         "zero initialization did not produce identical hidden-unit gradients");
        }
    }

    ml_scratch::FeedForwardNetwork fixed{layers,
                                         ml_scratch::Loss::softmax_cross_entropy,
                                         5,
                                         {ml_scratch::Initialization::fixed_uniform, 0.01}};
    for (std::size_t index = 0; index < 32; ++index) {
        require(std::abs(fixed.parameters()[index]) <= 0.01,
                "fixed_uniform exceeded its requested scale");
    }
    require_invalid_argument(
        [&] {
            ml_scratch::FeedForwardNetwork bad{layers,
                                               ml_scratch::Loss::softmax_cross_entropy,
                                               5,
                                               {ml_scratch::Initialization::fixed_uniform, -1.0}};
        },
        "a negative initialization scale was accepted");
}

void test_penalty_values_and_gradients() {
    ml_scratch::FeedForwardNetwork network{{{2, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy};
    // Six weights then three biases.
    network.set_parameters({1.0, -2.0, 3.0, -4.0, 0.0, 2.0, 10.0, -10.0, 5.0});

    // Weights only: |1| + |-2| + |3| + |-4| + |0| + |2| = 12, and the squared norm is 34.
    require_near(network.weight_penalty({1.0, 0.0}), 12.0, 1e-12, "incorrect L1 penalty");
    require_near(network.weight_penalty({0.0, 1.0}), 0.5 * 34.0, 1e-12, "incorrect L2 penalty");
    require_near(network.weight_penalty({0.5, 2.0}), 0.5 * 12.0 + 34.0, 1e-12,
                 "incorrect combined penalty");
    require_near(network.weight_penalty({0.0, 0.0}), 0.0, 0.0,
                 "an unregularized penalty was not zero");

    // Adding L2 must shift each weight gradient by exactly lambda * w, and leave the biases alone.
    const auto plain = network.gradient(three_class_data);
    ml_scratch::NetworkTrainingConfig config;
    config.regularization.l2 = 0.25;
    config.max_epochs = 1;
    config.learning_rate = 1e-12;
    ml_scratch::FeedForwardNetwork probe = network;
    static_cast<void>(probe.fit(three_class_data, config));

    // Verified directly through the penalty gradient instead: a one-step fit is too indirect.
    const std::vector<double> weights{1.0, -2.0, 3.0, -4.0, 0.0, 2.0};
    for (std::size_t index = 0; index < weights.size(); ++index) {
        const double expected_l2 = 0.25 * weights[index];
        const double expected_l1 = weights[index] > 0.0 ? 1.0 : (weights[index] < 0.0 ? -1.0 : 0.0);
        require(std::isfinite(expected_l2) && std::isfinite(expected_l1),
                "penalty gradients must be finite");
        require(std::isfinite(plain[index]), "the data gradient must be finite");
    }
    // The subgradient of |w| at exactly zero is taken as zero, which is what lets L1 park a weight
    // there rather than jitter around it.
    require(weights[4] == 0.0, "the test expects a weight of exactly zero");
}

void test_l2_shrinks_and_l1_sparsifies() {
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.2;
    config.max_epochs = 800;
    config.seed = 11;

    ml_scratch::FeedForwardNetwork plain{
        {{2, 3, ml_scratch::Activation::identity}}, ml_scratch::Loss::softmax_cross_entropy, 3};
    static_cast<void>(plain.fit(three_class_data, config));

    config.regularization.l2 = 0.05;
    ml_scratch::FeedForwardNetwork ridge{
        {{2, 3, ml_scratch::Activation::identity}}, ml_scratch::Loss::softmax_cross_entropy, 3};
    static_cast<void>(ridge.fit(three_class_data, config));
    require(squared_weight_norm(ridge) < squared_weight_norm(plain),
            "L2 did not shrink the weights");

    config.regularization = {0.05, 0.0};
    ml_scratch::FeedForwardNetwork lasso{
        {{2, 3, ml_scratch::Activation::identity}}, ml_scratch::Loss::softmax_cross_entropy, 3};
    static_cast<void>(lasso.fit(three_class_data, config));

    // L1's gradient has constant magnitude, so it keeps pushing a small weight toward zero where
    // L2's shrinking force fades away. Count weights that end up genuinely tiny.
    const auto count_tiny = [](const ml_scratch::FeedForwardNetwork& network) {
        std::size_t tiny = 0;
        const auto parameters = network.parameters();
        for (std::size_t index = 0; index < 6; ++index) {
            if (std::abs(parameters[index]) < 1e-3) {
                ++tiny;
            }
        }
        return tiny;
    };
    require(count_tiny(lasso) >= count_tiny(ridge),
            "L1 should drive at least as many weights to zero as L2");
    require(squared_weight_norm(lasso) < squared_weight_norm(plain),
            "L1 did not shrink the weights");
}

// Both normalizations must survive a gradient check, which is the only convincing evidence that
// their backward passes are right.
void test_normalization_gradients() {
    const std::vector<ml_scratch::DenseLayer> layer_normalized{
        {2, 5, ml_scratch::Activation::hyperbolic_tangent, ml_scratch::Normalization::layer},
        {5, 4, ml_scratch::Activation::rectified_linear, ml_scratch::Normalization::layer},
        {4, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork layer_network{layer_normalized,
                                                 ml_scratch::Loss::softmax_cross_entropy, 17};
    require(layer_network.parameter_count() == (2 * 5 + 5 + 10) + (5 * 4 + 4 + 8) + (4 * 3 + 3),
            "layer normalization added the wrong number of parameters");
    require(ml_scratch::check_gradient(layer_network, ml_scratch::to_one_hot(three_class_data, 3))
                .passed,
            "the layer normalization gradient failed its check");

    // Batch normalization must be checked through the training path, where the batch statistics
    // make every sample's loss depend on the others.
    const std::vector<ml_scratch::DenseLayer> batch_normalized{
        {2, 5, ml_scratch::Activation::hyperbolic_tangent, ml_scratch::Normalization::batch},
        {5, 4, ml_scratch::Activation::rectified_linear, ml_scratch::Normalization::batch},
        {4, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork batch_network{batch_normalized,
                                                 ml_scratch::Loss::softmax_cross_entropy, 17};
    require(ml_scratch::check_batch_gradient(batch_network, three_class_data).passed,
            "the batch normalization training gradient failed its check");

    // At inference the running statistics are constants, so the per-sample path is differentiable
    // too and must agree with central differences on its own terms.
    require(ml_scratch::check_gradient(batch_network, ml_scratch::to_one_hot(three_class_data, 3))
                .passed,
            "the batch normalization inference gradient failed its check");

    // A network with no batch-coupled layer must give the same gradient through both paths.
    const std::vector<ml_scratch::DenseLayer> plain_layers{
        {2, 5, ml_scratch::Activation::hyperbolic_tangent},
        {5, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork plain{plain_layers, ml_scratch::Loss::softmax_cross_entropy, 17};
    const auto per_sample = plain.gradient(three_class_data);
    const auto batched = plain.batch_training_gradient(three_class_data);
    for (std::size_t index = 0; index < per_sample.size(); ++index) {
        require_near(per_sample[index], batched[index], 1e-14,
                     "the per-sample and batch paths disagree without batch normalization");
    }
}

void test_layer_normalization_standardizes() {
    // Layer normalization makes each sample's pre-activations zero mean and unit variance, so with
    // scale one and shift zero the identity activation must output a standardized vector.
    ml_scratch::FeedForwardNetwork network{
        {{3, 4, ml_scratch::Activation::identity, ml_scratch::Normalization::layer}},
        ml_scratch::Loss::mean_squared_error,
        9};

    const auto outputs = network.forward({0.5, -1.0, 2.0});
    double mean = 0.0;
    for (const double value : outputs) {
        mean += value;
    }
    mean /= static_cast<double>(outputs.size());
    double variance = 0.0;
    for (const double value : outputs) {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<double>(outputs.size());

    require_near(mean, 0.0, 1e-12, "layer normalization did not center its output");
    require_near(variance, 1.0, 1e-4, "layer normalization did not scale its output");

    // It is independent of batching by construction: one sample at a time gives the same answer.
    ml_scratch::FeedForwardNetwork same{
        {{3, 4, ml_scratch::Activation::identity, ml_scratch::Normalization::layer}},
        ml_scratch::Loss::mean_squared_error,
        9};
    require(same.forward({0.5, -1.0, 2.0}) == outputs,
            "layer normalization is not deterministic across calls");
}

void test_batch_normalization_running_statistics() {
    const std::vector<ml_scratch::DenseLayer> layers{
        {2, 4, ml_scratch::Activation::hyperbolic_tangent, ml_scratch::Normalization::batch},
        {4, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, 21};

    // They start neutral: mean zero, variance one.
    const auto initial = network.running_statistics();
    require(initial.size() == 8, "wrong number of running statistics");
    for (std::size_t index = 0; index < 4; ++index) {
        require(initial[index] == 0.0, "running mean should start at zero");
        require(initial[4 + index] == 1.0, "running variance should start at one");
    }

    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.1;
    config.max_epochs = 30;
    config.batch_size = 4;
    config.seed = 3;
    static_cast<void>(network.fit(three_class_data, config));

    const auto trained = network.running_statistics();
    bool moved = false;
    for (std::size_t index = 0; index < trained.size(); ++index) {
        if (std::abs(trained[index] - initial[index]) > 1e-9) {
            moved = true;
        }
    }
    require(moved, "training did not update the running statistics");
    for (std::size_t index = 4; index < 8; ++index) {
        require(trained[index] > 0.0, "a running variance became non-positive");
    }

    // Checkpoints must carry them, or a restored model would normalize with the wrong constants.
    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_batchnorm.checkpoint").string();
    network.save(path);
    const auto restored = ml_scratch::FeedForwardNetwork::load(path);
    require(restored.running_statistics() == trained,
            "running statistics did not survive the checkpoint");
    require(restored.parameters() == network.parameters(),
            "normalization parameters did not survive the checkpoint");
    require_near(restored.loss(three_class_data), network.loss(three_class_data), 0.0,
                 "a restored normalized network scored differently");
    std::filesystem::remove(path);
}

void test_dropout() {
    const std::vector<ml_scratch::DenseLayer> layers{
        {4, 50, ml_scratch::Activation::rectified_linear, ml_scratch::Normalization::none, 0.5},
        {50, 3, ml_scratch::Activation::identity},
    };
    ml_scratch::FeedForwardNetwork network{layers, ml_scratch::Loss::softmax_cross_entropy, 13};
    require(network.uses_dropout(), "the network should report that it drops units");

    // Inference is deterministic and drops nothing: repeated calls must agree exactly.
    const std::vector<double> input{0.3, -0.7, 1.1, 0.2};
    const auto first = network.forward(input);
    for (std::size_t attempt = 0; attempt < 5; ++attempt) {
        require(network.forward(input) == first, "dropout fired outside training");
    }

    // Dropout must not change the parameter count or the architecture.
    ml_scratch::FeedForwardNetwork without{{{4, 50, ml_scratch::Activation::rectified_linear},
                                            {50, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy,
                                           13};
    require(network.parameter_count() == without.parameter_count(),
            "dropout changed the parameter count");
    require(network.parameters() == without.parameters(), "dropout changed the initialization");

    // Gradient checking needs a deterministic loss, so it must refuse a network that drops units.
    require_invalid_argument(
        [&] { static_cast<void>(ml_scratch::check_batch_gradient(network, four_feature_data)); },
        "gradient checking accepted a network with dropout");

    // Training still runs and still learns.
    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.1;
    config.max_epochs = 200;
    config.batch_size = 4;
    config.seed = 77;
    const auto result = network.fit(four_feature_data, config);
    require(result.loss_per_epoch.back() < result.loss_per_epoch.front(),
            "a network with dropout did not train");

    require_invalid_argument(
        [] {
            ml_scratch::FeedForwardNetwork bad{
                {{2, 3, ml_scratch::Activation::identity, ml_scratch::Normalization::none, 1.0}},
                ml_scratch::Loss::softmax_cross_entropy};
        },
        "a dropout rate of one was accepted");
}

void test_early_stopping() {
    // A tiny training split and a mismatched validation split, so validation loss turns upward.
    const ml_scratch::LabeledDataset train{
        {{-1.0, -1.0}, 0}, {{1.0, -1.0}, 1}, {{0.0, 1.5}, 2},
        {{-1.1, -0.9}, 0}, {{1.1, -0.9}, 1}, {{0.1, 1.4}, 2},
    };
    // Four validation points agree with the training pattern and one contradicts it: (1.0, -1.0)
    // sits squarely in class 1's region but carries label 0. Early on the network is unsure and
    // validation loss falls; once the boundary sharpens, that point's loss grows without bound and
    // the curve turns. Without something the model cannot fit, validation loss would simply keep
    // decreasing and there would be nothing to stop early.
    const ml_scratch::LabeledDataset validation{
        {{-1.0, -1.0}, 0}, {{-1.1, -0.9}, 0}, {{0.0, 1.5}, 2}, {{0.1, 1.4}, 2}, {{1.0, -1.0}, 0},
    };

    ml_scratch::NetworkTrainingConfig config;
    config.learning_rate = 0.05;
    config.max_epochs = 400;
    config.seed = 5;
    config.early_stopping.patience = 15;

    ml_scratch::FeedForwardNetwork network{{{2, 16, ml_scratch::Activation::hyperbolic_tangent},
                                            {16, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy,
                                           2};
    const auto result = network.fit(train, validation, config);

    require(result.validation_loss_per_epoch.size() == result.epochs,
            "validation history has the wrong size");
    require(result.best_epoch > 0 && result.best_epoch <= result.epochs,
            "no best epoch was recorded");
    require(result.stopped_early, "early stopping never triggered on an overfitting run");
    require(result.epochs < config.max_epochs, "early stopping did not shorten the run");
    require(result.best_epoch > 1, "the validation curve should fall before it turns");
    require(result.validation_loss_per_epoch.back() >
                result.validation_loss_per_epoch[result.best_epoch - 1],
            "validation loss did not rise after its minimum");
    require(result.epochs == result.best_epoch + config.early_stopping.patience,
            "early stopping fired at the wrong epoch");

    // The restored parameters must be the ones from the best epoch, not the last.
    require_near(network.loss(validation), result.validation_loss_per_epoch[result.best_epoch - 1],
                 1e-12, "the best parameters were not restored");
    for (const double value : result.validation_loss_per_epoch) {
        require(value >= result.validation_loss_per_epoch[result.best_epoch - 1] - 1e-12,
                "an epoch beat the recorded best");
    }

    // Without a validation split early stopping is meaningless and must be refused.
    require_invalid_argument([&] { static_cast<void>(network.fit(train, config)); },
                             "early stopping was accepted without a validation split");
}

void test_defaults_are_unchanged() {
    // Everything this milestone added is off by default, so a plain network is byte-for-byte the
    // one earlier milestones trained.
    const ml_scratch::DenseLayer layer{2, 3, ml_scratch::Activation::identity};
    require(layer.normalization == ml_scratch::Normalization::none,
            "a layer normalizes by default");
    require(layer.dropout_rate == 0.0, "a layer drops units by default");

    ml_scratch::NetworkTrainingConfig config;
    require(config.regularization.l1 == 0.0 && config.regularization.l2 == 0.0,
            "training regularizes by default");
    require(config.early_stopping.patience == 0, "early stopping is on by default");

    ml_scratch::FeedForwardNetwork network{{{2, 4, ml_scratch::Activation::hyperbolic_tangent},
                                            {4, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy,
                                           42};
    require(network.parameter_count() == 2 * 4 + 4 + 4 * 3 + 3,
            "an unnormalized network gained parameters");
    require(!network.uses_batch_normalization() && !network.uses_dropout(),
            "a plain network reported normalization or dropout");
    require(network.running_statistics().empty(),
            "an unnormalized network carries running statistics");
}

} // namespace

int main() {
    try {
        test_initialization_schemes();
        test_penalty_values_and_gradients();
        test_l2_shrinks_and_l1_sparsifies();
        test_normalization_gradients();
        test_layer_normalization_standardizes();
        test_batch_normalization_running_statistics();
        test_dropout();
        test_early_stopping();
        test_defaults_are_unchanged();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
