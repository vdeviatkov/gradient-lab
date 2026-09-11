#include "ml_scratch/mnist.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/softmax_regression.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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

template <typename Function>
void require_runtime_error(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, message);
}

// Three classes that one linear boundary set can separate.
const ml_scratch::LabeledDataset separable_data{
    {{2.0, 0.0}, 0},  {{2.5, 0.4}, 0},   {{2.2, -0.3}, 0},  {{-1.0, 2.0}, 1},  {{-1.4, 2.3}, 1},
    {{-0.8, 1.7}, 1}, {{-1.0, -2.0}, 2}, {{-1.3, -2.4}, 2}, {{-0.7, -1.8}, 2},
};

void test_untrained_model_is_uniform() {
    ml_scratch::SoftmaxRegression model{2, 3};
    require(model.parameter_count() == 3 * 3, "incorrect parameter count");

    const auto probabilities = model.predict_probabilities({1.0, -2.0});
    for (const double probability : probabilities) {
        require_near(probability, 1.0 / 3.0, 1e-12, "zero parameters should give a uniform output");
    }
    // Cross-entropy of a uniform distribution over K classes is log K.
    require_near(model.cross_entropy(separable_data), std::log(3.0), 1e-12,
                 "zero-parameter loss should equal log(K)");
    require(model.predict({1.0, -2.0}) == 0, "a tie should resolve to the lowest class index");
}

void test_gradient_by_hand() {
    // One sample, zero parameters: p is uniform, so dL/dz = 1/K - [k == y].
    ml_scratch::SoftmaxRegression model{2, 2};
    const auto gradient = model.gradient({{{3.0, -1.0}, 0}});

    require_near(gradient[0], -1.5, 1e-12, "incorrect weight gradient for the true class");
    require_near(gradient[1], 0.5, 1e-12, "incorrect weight gradient for the true class");
    require_near(gradient[2], 1.5, 1e-12, "incorrect weight gradient for the other class");
    require_near(gradient[3], -0.5, 1e-12, "incorrect weight gradient for the other class");
    require_near(gradient[4], -0.5, 1e-12, "incorrect bias gradient");
    require_near(gradient[5], 0.5, 1e-12, "incorrect bias gradient");

    // The L2 term touches the weights only, never the biases.
    ml_scratch::SoftmaxRegression penalized{2, 2};
    penalized.set_parameters({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const auto plain = penalized.gradient({{{3.0, -1.0}, 0}}, 0.0);
    const auto regularized = penalized.gradient({{{3.0, -1.0}, 0}}, 0.5);
    for (std::size_t index = 0; index < 4; ++index) {
        require_near(regularized[index] - plain[index], 0.5 * (static_cast<double>(index) + 1.0),
                     1e-12, "incorrect L2 weight gradient");
    }
    require_near(regularized[4], plain[4], 0.0, "L2 must not touch the bias");
    require_near(regularized[5], plain[5], 0.0, "L2 must not touch the bias");
}

// The milestone 7 network computes the same function with one linear layer under softmax
// cross-entropy. Agreement between two independent implementations is the point of this test.
void test_agrees_with_the_general_network() {
    ml_scratch::SoftmaxRegression model{2, 3};
    const std::vector<double> parameters{0.4, -0.2, 0.1, 0.7, -0.5, 0.25, 0.05, -0.3, 0.6};
    model.set_parameters(parameters);

    ml_scratch::FeedForwardNetwork network{{{2, 3, ml_scratch::Activation::identity}},
                                           ml_scratch::Loss::softmax_cross_entropy};
    require(network.parameter_count() == model.parameter_count(),
            "the two implementations disagree on the parameter count");
    network.set_parameters(parameters);

    const auto encoded = ml_scratch::to_one_hot(separable_data, 3);
    require_near(model.cross_entropy(separable_data), network.loss(encoded), 1e-12,
                 "the two implementations disagree on the loss");

    const auto model_gradient = model.gradient(separable_data);
    const auto network_gradient = network.gradient(encoded);
    for (std::size_t index = 0; index < model_gradient.size(); ++index) {
        require_near(model_gradient[index], network_gradient[index], 1e-12,
                     "the two implementations disagree on the gradient");
    }

    for (const auto& sample : separable_data) {
        const auto model_probabilities = model.predict_probabilities(sample.features);
        const auto network_probabilities = network.predict(sample.features);
        for (std::size_t label = 0; label < 3; ++label) {
            require_near(model_probabilities[label], network_probabilities[label], 1e-12,
                         "the two implementations disagree on the probabilities");
        }
    }
}

void test_gradient_matches_central_differences() {
    ml_scratch::SoftmaxRegression model{2, 3};
    model.set_parameters({0.4, -0.2, 0.1, 0.7, -0.5, 0.25, 0.05, -0.3, 0.6});
    const double l2 = 0.3;

    const auto analytic = model.gradient(separable_data, l2);
    const auto original = model.parameters();
    std::vector<double> perturbed = original;
    std::vector<double> numerical(analytic.size(), 0.0);
    constexpr double epsilon = 1e-6;

    for (std::size_t index = 0; index < analytic.size(); ++index) {
        perturbed[index] = original[index] + epsilon;
        model.set_parameters(perturbed);
        const double raised = model.cross_entropy(separable_data, l2);
        perturbed[index] = original[index] - epsilon;
        model.set_parameters(perturbed);
        const double lowered = model.cross_entropy(separable_data, l2);
        perturbed[index] = original[index];
        numerical[index] = (raised - lowered) / (2.0 * epsilon);
    }
    model.set_parameters(original);

    const auto check = ml_scratch::check_gradient(analytic, numerical);
    require(check.passed, "the softmax gradient disagreed with central differences");
}

void test_training_and_stability() {
    ml_scratch::SoftmaxTrainingConfig config;
    config.learning_rate = 0.5;
    config.max_epochs = 2'000;
    config.batch_size = 3;
    config.seed = 20260911;
    config.target_loss = 0.05;

    ml_scratch::SoftmaxRegression model{2, 3};
    const auto result = model.fit(separable_data, config);

    require(result.converged, "separable data did not reach the loss target");
    require(result.epochs == result.loss_per_epoch.size(), "loss history has the wrong size");
    require(result.loss_per_epoch.back() <= config.target_loss,
            "reported convergence above the target loss");
    require_near(model.accuracy(separable_data), 1.0, 1e-12,
                 "separable data was not classified perfectly");

    // Saturated scores must not overflow into a non-finite loss or probability.
    ml_scratch::SoftmaxRegression extreme{1, 2};
    extreme.set_parameters({1000.0, -1000.0, 0.0, 0.0});
    const auto probabilities = extreme.predict_probabilities({1000.0});
    require_near(probabilities[0], 1.0, 1e-12, "stable softmax failed at a large score");
    require_near(probabilities[1], 0.0, 1e-12, "stable softmax failed at a large score");
    const double loss = extreme.cross_entropy({{{1000.0}, 1}});
    require(std::isfinite(loss) && loss > 0.0, "log-sum-exp overflowed at a large score");
}

void test_l2_shrinks_the_weights() {
    ml_scratch::SoftmaxTrainingConfig config;
    config.learning_rate = 0.5;
    config.max_epochs = 300;
    config.batch_size = 3;
    config.seed = 7;

    const auto squared_norm = [](const ml_scratch::SoftmaxRegression& model) {
        double total = 0.0;
        for (const auto& row : model.weights()) {
            for (const double weight : row) {
                total += weight * weight;
            }
        }
        return total;
    };

    ml_scratch::SoftmaxRegression plain{2, 3};
    plain.fit(separable_data, config);
    config.l2_regularization = 0.5;
    ml_scratch::SoftmaxRegression penalized{2, 3};
    penalized.fit(separable_data, config);

    require(squared_norm(penalized) < squared_norm(plain),
            "the L2 penalty did not shrink the weights");
    require(penalized.biases().size() == 3, "regularization changed the model shape");
}

void test_metrics() {
    // confusion[actual][predicted]: class 0 perfect, class 1 half confused with class 2.
    const ml_scratch::ConfusionCounts confusion{{4, 0, 0}, {0, 2, 2}, {0, 0, 4}};
    const auto metrics = ml_scratch::multiclass_metrics(confusion);

    require_near(metrics.accuracy, 10.0 / 12.0, 1e-12, "incorrect accuracy");
    require_near(metrics.per_class_recall[1], 0.5, 1e-12, "incorrect recall");
    require_near(metrics.per_class_precision[2], 4.0 / 6.0, 1e-12, "incorrect precision");
    require_near(metrics.per_class_precision[0], 1.0, 1e-12, "incorrect precision");
    require_near(metrics.macro_recall, (1.0 + 0.5 + 1.0) / 3.0, 1e-12, "incorrect macro recall");

    // A class that is never predicted has zero precision rather than an undefined ratio.
    const auto missing = ml_scratch::multiclass_metrics({{2, 0}, {2, 0}});
    require_near(missing.per_class_precision[1], 0.0, 1e-12, "an unpredicted class should be zero");
    require_near(missing.per_class_f1[1], 0.0, 1e-12, "an unpredicted class should be zero");

    require_invalid_argument([] { static_cast<void>(ml_scratch::multiclass_metrics({})); },
                             "an empty confusion matrix was accepted");
    require_invalid_argument([] { static_cast<void>(ml_scratch::multiclass_metrics({{1, 2}})); },
                             "a non-square confusion matrix was accepted");
}

void test_seed_reproducibility() {
    ml_scratch::SoftmaxTrainingConfig config;
    config.learning_rate = 0.2;
    config.max_epochs = 40;
    config.batch_size = 2;
    config.seed = 4242;

    ml_scratch::SoftmaxRegression first{2, 3};
    ml_scratch::SoftmaxRegression second{2, 3};
    const auto first_result = first.fit(separable_data, config);
    const auto second_result = second.fit(separable_data, config);

    require(first_result == second_result, "identical seeds produced different histories");
    require(first.parameters() == second.parameters(),
            "identical seeds produced different weights");
}

// A small IDX file is written here and read back, so the parser is tested without needing MNIST.
std::string write_temporary_idx(const std::string& name, const std::vector<unsigned char>& bytes) {
    const std::string path =
        (std::filesystem::temp_directory_path() / ("ml_scratch_idx_" + name)).string();
    std::ofstream stream{path, std::ios::binary};
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stream.close();
    return path;
}

void test_idx_reader() {
    // Two 2x2 images: magic 0x00000803, then the three dimensions, then four pixels each.
    const std::vector<unsigned char> image_bytes{
        0x00, 0x00, 0x08, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02,
        0x00, 0x00, 0x00, 0x02, 0,    255,  51,   204,  0,    0,    255,  255,
    };
    const std::vector<unsigned char> label_bytes{
        0x00, 0x00, 0x08, 0x01, 0x00, 0x00, 0x00, 0x02, 3, 7,
    };
    const std::string image_path = write_temporary_idx("_images", image_bytes);
    const std::string label_path = write_temporary_idx("_labels", label_bytes);

    const auto images = ml_scratch::read_idx_file(image_path);
    require(images.dimensions == std::vector<std::size_t>{2, 2, 2}, "incorrect image dimensions");
    require(images.value_count() == 8, "incorrect pixel count");
    require(images.values[1] == 255, "incorrect pixel value");

    const auto split = ml_scratch::load_mnist(image_path, label_path);
    require(split.rows == 2 && split.columns == 2, "incorrect image shape");
    require(split.samples.size() == 2, "incorrect sample count");
    require(split.samples[0].label == 3 && split.samples[1].label == 7, "incorrect labels");
    // Pixels are rescaled from 0..255 into [0, 1].
    require_near(split.samples[0].features[0], 0.0, 1e-12, "incorrect normalization");
    require_near(split.samples[0].features[1], 1.0, 1e-12, "incorrect normalization");
    require_near(split.samples[0].features[2], 51.0 / 255.0, 1e-12, "incorrect normalization");

    const auto capped = ml_scratch::load_mnist(image_path, label_path, 1);
    require(capped.samples.size() == 1, "max_samples was ignored");

    require_runtime_error([] { static_cast<void>(ml_scratch::read_idx_file("/no/such/file")); },
                          "a missing file was accepted");
    // A wrong element type and a truncated payload must both be rejected.
    const std::string bad_type =
        write_temporary_idx("_badtype", {0x00, 0x00, 0x0D, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00});
    require_runtime_error([&] { static_cast<void>(ml_scratch::read_idx_file(bad_type)); },
                          "an unsupported element type was accepted");
    const std::string truncated =
        write_temporary_idx("_short", {0x00, 0x00, 0x08, 0x01, 0x00, 0x00, 0x00, 0x04, 1, 2});
    require_runtime_error([&] { static_cast<void>(ml_scratch::read_idx_file(truncated)); },
                          "a truncated payload was accepted");
    // Mismatched counts between the two files must be caught.
    const std::string one_label =
        write_temporary_idx("_one", {0x00, 0x00, 0x08, 0x01, 0x00, 0x00, 0x00, 0x01, 5});
    require_runtime_error([&] { static_cast<void>(ml_scratch::load_mnist(image_path, one_label)); },
                          "mismatched image and label counts were accepted");

    for (const std::string& path : {image_path, label_path, bad_type, truncated, one_label}) {
        std::filesystem::remove(path);
    }
}

void test_input_validation() {
    require_invalid_argument([] { ml_scratch::SoftmaxRegression(0, 3); },
                             "a zero feature count was accepted");
    require_invalid_argument([] { ml_scratch::SoftmaxRegression(2, 1); },
                             "a single-class model was accepted");

    ml_scratch::SoftmaxRegression model{2, 3};
    require_invalid_argument([&] { static_cast<void>(model.scores({1.0})); },
                             "a wrong feature count was accepted");
    require_invalid_argument(
        [&] { static_cast<void>(model.scores({std::numeric_limits<double>::infinity(), 1.0})); },
        "a non-finite feature was accepted");
    require_invalid_argument([&] { static_cast<void>(model.accuracy({{{1.0, 2.0}, 9}})); },
                             "a label outside the class count was accepted");
    require_invalid_argument([&] { static_cast<void>(model.cross_entropy(separable_data, -1.0)); },
                             "a negative L2 coefficient was accepted");
    require_invalid_argument([&] { model.set_parameters({1.0}); },
                             "a wrong parameter vector size was accepted");

    ml_scratch::SoftmaxTrainingConfig config;
    config.learning_rate = 0.0;
    require_invalid_argument([&] { model.fit(separable_data, config); },
                             "a zero learning rate was accepted");
    config.learning_rate = 0.1;
    config.max_epochs = 0;
    require_invalid_argument([&] { model.fit(separable_data, config); },
                             "a zero epoch budget was accepted");
}

} // namespace

int main() {
    try {
        test_untrained_model_is_uniform();
        test_gradient_by_hand();
        test_agrees_with_the_general_network();
        test_gradient_matches_central_differences();
        test_training_and_stability();
        test_l2_shrinks_the_weights();
        test_metrics();
        test_seed_reproducibility();
        test_idx_reader();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
