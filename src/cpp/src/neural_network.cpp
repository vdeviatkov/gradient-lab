#include "ml_scratch/neural_network.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

double sigmoid(const double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

// log(1 + e^v) without overflow, used to write binary cross-entropy in terms of logits.
double softplus(const double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double apply_activation(const double value, const Activation activation) {
    switch (activation) {
    case Activation::sigmoid:
        return sigmoid(value);
    case Activation::hyperbolic_tangent:
        return std::tanh(value);
    case Activation::rectified_linear:
        return std::max(value, 0.0);
    case Activation::identity:
        break;
    }
    return value;
}

// Derivative with respect to the pre-activation, expressed through the activation where that is
// cheaper and better conditioned: sigmoid' = a(1 - a) and tanh' = 1 - a^2.
double activation_derivative(const double pre_activation, const double activation_value,
                             const Activation activation) {
    switch (activation) {
    case Activation::sigmoid:
        return activation_value * (1.0 - activation_value);
    case Activation::hyperbolic_tangent:
        return 1.0 - activation_value * activation_value;
    case Activation::rectified_linear:
        // The kink at zero has no derivative; the subgradient zero is the usual convention.
        return pre_activation > 0.0 ? 1.0 : 0.0;
    case Activation::identity:
        break;
    }
    return 1.0;
}

std::vector<double> softmax(const std::vector<double>& logits) {
    const double largest = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size());
    double total = 0.0;
    for (std::size_t index = 0; index < logits.size(); ++index) {
        probabilities[index] = std::exp(logits[index] - largest);
        total += probabilities[index];
    }
    for (double& probability : probabilities) {
        probability /= total;
    }
    return probabilities;
}

} // namespace

FeedForwardNetwork::FeedForwardNetwork(std::vector<DenseLayer> layers, const Loss loss,
                                       const std::uint32_t seed)
    : layers_(std::move(layers)), loss_(loss) {
    if (layers_.empty()) {
        throw std::invalid_argument("a network needs at least one layer");
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        if (layers_[index].input_size == 0 || layers_[index].output_size == 0) {
            throw std::invalid_argument("layer sizes must be positive");
        }
        if (index > 0 && layers_[index].input_size != layers_[index - 1].output_size) {
            throw std::invalid_argument("layer sizes do not connect");
        }
    }
    const bool logit_loss = loss_ != Loss::mean_squared_error;
    if (logit_loss && layers_.back().activation != Activation::identity) {
        throw std::invalid_argument(
            "cross-entropy losses apply their own output transform, so the final layer must be "
            "linear");
    }
    if (loss_ == Loss::softmax_cross_entropy && layers_.back().output_size < 2) {
        throw std::invalid_argument("softmax_cross_entropy needs at least two outputs");
    }

    std::mt19937 random_engine{seed};
    // Uniform draws are taken from the engine directly rather than through a standard
    // distribution, whose implementation is not specified bit-for-bit across libraries.
    const auto uniform = [&random_engine](const double limit) {
        constexpr double range = 4294967296.0; // std::mt19937::max() + 1
        const double unit = static_cast<double>(random_engine()) / range;
        return -limit + 2.0 * limit * unit;
    };

    for (const DenseLayer& layer : layers_) {
        const auto fan_in = static_cast<double>(layer.input_size);
        const auto fan_out = static_cast<double>(layer.output_size);
        // He scaling for the one-sided ReLU, Glorot for the symmetric activations.
        const double limit = layer.activation == Activation::rectified_linear
                                 ? std::sqrt(6.0 / fan_in)
                                 : std::sqrt(6.0 / (fan_in + fan_out));

        LayerParameters parameters;
        parameters.weights.assign(layer.output_size, std::vector<double>(layer.input_size, 0.0));
        parameters.biases.assign(layer.output_size, 0.0);
        for (auto& row : parameters.weights) {
            for (double& weight : row) {
                weight = uniform(limit);
            }
        }
        parameter_count_ += layer.output_size * layer.input_size + layer.output_size;
        parameters_.push_back(std::move(parameters));
    }
}

void FeedForwardNetwork::validate_for_network(const SupervisedDataset& dataset) const {
    const SupervisedShape shape = validate_supervised_dataset(dataset);
    if (shape.feature_count != input_size()) {
        throw std::invalid_argument("feature count does not match the network input size");
    }
    if (shape.target_count != output_size()) {
        throw std::invalid_argument("target count does not match the network output size");
    }
}

FeedForwardNetwork::ForwardCache
FeedForwardNetwork::forward_cache(const std::vector<double>& input) const {
    if (input.size() != input_size()) {
        throw std::invalid_argument("input size does not match the network");
    }
    for (const double value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("inputs must be finite");
        }
    }

    ForwardCache cache;
    cache.activations.reserve(layers_.size() + 1);
    cache.pre_activations.reserve(layers_.size());
    cache.activations.push_back(input);

    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const DenseLayer& layer = layers_[index];
        const std::vector<double>& previous = cache.activations.back();
        std::vector<double> pre_activation(layer.output_size, 0.0);
        std::vector<double> activation(layer.output_size, 0.0);

        for (std::size_t output = 0; output < layer.output_size; ++output) {
            double total = parameters_[index].biases[output];
            const std::vector<double>& row = parameters_[index].weights[output];
            for (std::size_t input_index = 0; input_index < layer.input_size; ++input_index) {
                total += row[input_index] * previous[input_index];
            }
            pre_activation[output] = total;
            activation[output] = apply_activation(total, layer.activation);
        }
        cache.pre_activations.push_back(std::move(pre_activation));
        cache.activations.push_back(std::move(activation));
    }
    return cache;
}

std::vector<double> FeedForwardNetwork::forward(const std::vector<double>& input) const {
    return forward_cache(input).activations.back();
}

std::vector<double> FeedForwardNetwork::predict(const std::vector<double>& input) const {
    std::vector<double> outputs = forward(input);
    switch (loss_) {
    case Loss::softmax_cross_entropy:
        return softmax(outputs);
    case Loss::binary_cross_entropy:
        for (double& value : outputs) {
            value = sigmoid(value);
        }
        return outputs;
    case Loss::mean_squared_error:
        break;
    }
    return outputs;
}

std::size_t FeedForwardNetwork::predict_class(const std::vector<double>& input) const {
    const std::vector<double> outputs = forward(input);
    std::size_t best = 0;
    for (std::size_t index = 1; index < outputs.size(); ++index) {
        if (outputs[index] > outputs[best]) {
            best = index;
        }
    }
    return best;
}

double FeedForwardNetwork::sample_loss(const std::vector<double>& outputs,
                                       const std::vector<double>& targets) const {
    const auto count = static_cast<double>(outputs.size());
    switch (loss_) {
    case Loss::mean_squared_error: {
        double total = 0.0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            const double residual = outputs[index] - targets[index];
            total += residual * residual;
        }
        return total / count;
    }
    case Loss::binary_cross_entropy: {
        // softplus(z) - y z is -log p written so that neither branch overflows.
        double total = 0.0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            total += softplus(outputs[index]) - targets[index] * outputs[index];
        }
        return total / count;
    }
    case Loss::softmax_cross_entropy: {
        const double largest = *std::max_element(outputs.begin(), outputs.end());
        double sum_of_exponentials = 0.0;
        for (const double logit : outputs) {
            sum_of_exponentials += std::exp(logit - largest);
        }
        const double log_normalizer = largest + std::log(sum_of_exponentials);
        double total = 0.0;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            total += targets[index] * (log_normalizer - outputs[index]);
        }
        return total;
    }
    }
    return 0.0;
}

std::vector<double> FeedForwardNetwork::output_delta(const std::vector<double>& pre_activations,
                                                     const std::vector<double>& outputs,
                                                     const std::vector<double>& targets) const {
    const auto count = static_cast<double>(outputs.size());
    std::vector<double> delta(outputs.size(), 0.0);

    switch (loss_) {
    case Loss::mean_squared_error: {
        const Activation activation = layers_.back().activation;
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            const double loss_gradient = 2.0 * (outputs[index] - targets[index]) / count;
            // outputs are activations here, so the chain rule still needs the activation slope.
            delta[index] = loss_gradient * activation_derivative(pre_activations[index],
                                                                 outputs[index], activation);
        }
        return delta;
    }
    case Loss::binary_cross_entropy:
        // Composing sigmoid with the loss collapses to p - y, with no separate slope factor.
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            delta[index] = (sigmoid(outputs[index]) - targets[index]) / count;
        }
        return delta;
    case Loss::softmax_cross_entropy: {
        const std::vector<double> probabilities = softmax(outputs);
        double target_total = 0.0;
        for (const double target : targets) {
            target_total += target;
        }
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            delta[index] = target_total * probabilities[index] - targets[index];
        }
        return delta;
    }
    }
    return delta;
}

double FeedForwardNetwork::loss(const SupervisedDataset& dataset) const {
    validate_for_network(dataset);
    double total = 0.0;
    for (const auto& sample : dataset) {
        total += sample_loss(forward(sample.features), sample.targets);
    }
    return total / static_cast<double>(dataset.size());
}

double FeedForwardNetwork::accuracy(const SupervisedDataset& dataset) const {
    validate_for_network(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        const auto expected = static_cast<std::size_t>(
            std::distance(sample.targets.begin(),
                          std::max_element(sample.targets.begin(), sample.targets.end())));
        if (predict_class(sample.features) == expected) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

std::vector<double> FeedForwardNetwork::gradient(const SupervisedDataset& dataset) const {
    validate_for_network(dataset);

    std::vector<double> flat(parameter_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(dataset.size());

    // Offset of each layer's block inside the flat parameter vector.
    std::vector<std::size_t> layer_offsets(layers_.size(), 0);
    for (std::size_t index = 1; index < layers_.size(); ++index) {
        layer_offsets[index] = layer_offsets[index - 1] +
                               layers_[index - 1].output_size * layers_[index - 1].input_size +
                               layers_[index - 1].output_size;
    }

    for (const auto& sample : dataset) {
        const ForwardCache cache = forward_cache(sample.features);
        // delta holds dL/dz for the layer currently being visited.
        std::vector<double> delta =
            output_delta(cache.pre_activations.back(), cache.activations.back(), sample.targets);

        for (std::size_t index = layers_.size(); index-- > 0;) {
            const DenseLayer& layer = layers_[index];
            const std::vector<double>& inputs = cache.activations[index];

            const std::size_t offset = layer_offsets[index];
            const std::size_t bias_offset = offset + layer.output_size * layer.input_size;

            for (std::size_t output = 0; output < layer.output_size; ++output) {
                const double unit_delta = delta[output];
                const std::size_t row_offset = offset + output * layer.input_size;
                for (std::size_t input = 0; input < layer.input_size; ++input) {
                    flat[row_offset + input] += scale * unit_delta * inputs[input];
                }
                flat[bias_offset + output] += scale * unit_delta;
            }

            if (index == 0) {
                break;
            }
            // Propagate to the previous layer: delta_prev = (W^T delta) * g'(z_prev).
            const DenseLayer& previous_layer = layers_[index - 1];
            std::vector<double> previous_delta(previous_layer.output_size, 0.0);
            for (std::size_t input = 0; input < layer.input_size; ++input) {
                double total = 0.0;
                for (std::size_t output = 0; output < layer.output_size; ++output) {
                    total += parameters_[index].weights[output][input] * delta[output];
                }
                previous_delta[input] =
                    total * activation_derivative(cache.pre_activations[index - 1][input],
                                                  cache.activations[index][input],
                                                  previous_layer.activation);
            }
            delta = std::move(previous_delta);
        }
    }
    return flat;
}

std::vector<double> FeedForwardNetwork::numerical_gradient(const SupervisedDataset& dataset,
                                                           const double epsilon) const {
    validate_for_network(dataset);
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("epsilon must be finite and positive");
    }

    // A copy keeps the const contract while the parameters are perturbed in place.
    FeedForwardNetwork probe = *this;
    const std::vector<double> original = parameters();
    std::vector<double> estimate(parameter_count_, 0.0);
    std::vector<double> perturbed = original;

    for (std::size_t index = 0; index < parameter_count_; ++index) {
        perturbed[index] = original[index] + epsilon;
        probe.set_parameters(perturbed);
        const double raised = probe.loss(dataset);

        perturbed[index] = original[index] - epsilon;
        probe.set_parameters(perturbed);
        const double lowered = probe.loss(dataset);

        perturbed[index] = original[index];
        // Central difference: its error is O(epsilon^2) rather than the O(epsilon) of a
        // one-sided difference.
        estimate[index] = (raised - lowered) / (2.0 * epsilon);
    }
    return estimate;
}

std::vector<double> FeedForwardNetwork::parameters() const {
    std::vector<double> flat;
    flat.reserve(parameter_count_);
    for (const LayerParameters& layer : parameters_) {
        for (const std::vector<double>& row : layer.weights) {
            flat.insert(flat.end(), row.begin(), row.end());
        }
        flat.insert(flat.end(), layer.biases.begin(), layer.biases.end());
    }
    return flat;
}

void FeedForwardNetwork::set_parameters(const std::vector<double>& values) {
    if (values.size() != parameter_count_) {
        throw std::invalid_argument("parameter vector has the wrong size");
    }
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("parameters must be finite");
        }
    }

    std::size_t cursor = 0;
    for (LayerParameters& layer : parameters_) {
        for (std::vector<double>& row : layer.weights) {
            for (double& weight : row) {
                weight = values[cursor++];
            }
        }
        for (double& bias : layer.biases) {
            bias = values[cursor++];
        }
    }
}

NetworkTrainingResult FeedForwardNetwork::fit(const SupervisedDataset& dataset,
                                              const NetworkTrainingConfig& config) {
    validate_for_network(dataset);
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }
    if (!std::isfinite(config.target_loss) || config.target_loss < 0.0) {
        throw std::invalid_argument("target_loss must be finite and non-negative");
    }

    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    const std::size_t batch_size =
        config.batch_size == 0 ? dataset.size() : std::min(config.batch_size, dataset.size());

    std::vector<double> history;
    history.reserve(config.max_epochs);
    for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        if (config.shuffle && batch_size < dataset.size()) {
            std::shuffle(order.begin(), order.end(), random_engine);
        }

        for (std::size_t begin = 0; begin < dataset.size(); begin += batch_size) {
            const std::size_t end = std::min(begin + batch_size, dataset.size());
            SupervisedDataset batch;
            batch.reserve(end - begin);
            for (std::size_t position = begin; position < end; ++position) {
                batch.push_back(dataset[order[position]]);
            }

            const std::vector<double> batch_gradient = gradient(batch);
            std::vector<double> updated = parameters();
            for (std::size_t index = 0; index < updated.size(); ++index) {
                updated[index] -= config.learning_rate * batch_gradient[index];
            }
            set_parameters(updated);
        }

        const double epoch_loss = loss(dataset);
        if (!std::isfinite(epoch_loss)) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        history.push_back(epoch_loss);
        if (config.target_loss > 0.0 && epoch_loss <= config.target_loss) {
            return {epoch, true, std::move(history)};
        }
    }
    return {config.max_epochs, false, std::move(history)};
}

GradientCheckResult check_gradient(const std::vector<double>& analytic,
                                   const std::vector<double>& numerical, const double tolerance) {
    if (analytic.empty()) {
        throw std::invalid_argument("gradients must not be empty");
    }
    if (analytic.size() != numerical.size()) {
        throw std::invalid_argument("gradients must have equal sizes");
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument("tolerance must be finite and positive");
    }

    GradientCheckResult result{0.0, 0.0, 0.0, 0, analytic.front(), numerical.front(), true};
    double difference_norm = 0.0;
    double analytic_norm = 0.0;
    double numerical_norm = 0.0;

    for (std::size_t index = 0; index < analytic.size(); ++index) {
        const double difference = std::abs(analytic[index] - numerical[index]);
        difference_norm += difference * difference;
        analytic_norm += analytic[index] * analytic[index];
        numerical_norm += numerical[index] * numerical[index];

        if (difference > result.max_absolute_error) {
            result.max_absolute_error = difference;
            result.worst_parameter = index;
            result.analytic_value = analytic[index];
            result.numerical_value = numerical[index];
        }
        // The floor keeps two entries that are both essentially zero from dividing a tiny
        // difference by an equally tiny denominator.
        const double denominator =
            std::max(std::abs(analytic[index]) + std::abs(numerical[index]), 1e-8);
        result.max_elementwise_relative_error =
            std::max(result.max_elementwise_relative_error, difference / denominator);
    }

    const double scale = std::sqrt(analytic_norm) + std::sqrt(numerical_norm);
    // An all-zero gradient pair is a perfect match, not a division by zero.
    result.relative_error = scale == 0.0 ? 0.0 : std::sqrt(difference_norm) / scale;
    result.passed = result.relative_error <= tolerance;
    return result;
}

GradientCheckResult check_gradient(const FeedForwardNetwork& network,
                                   const SupervisedDataset& dataset, const double epsilon,
                                   const double tolerance) {
    return check_gradient(network.gradient(dataset), network.numerical_gradient(dataset, epsilon),
                          tolerance);
}

} // namespace ml_scratch
