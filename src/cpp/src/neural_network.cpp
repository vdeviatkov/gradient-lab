#include "ml_scratch/neural_network.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
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
                                       const std::uint32_t seed,
                                       const InitializationConfig initialization)
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
        const double rate = layers_[index].dropout_rate;
        if (!std::isfinite(rate) || rate < 0.0 || rate >= 1.0) {
            throw std::invalid_argument("dropout_rate must be finite and in the interval [0, 1)");
        }
        if (layers_[index].normalization == Normalization::layer &&
            layers_[index].output_size < 2) {
            throw std::invalid_argument(
                "layer normalization needs at least two units to have a spread");
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
    if (!std::isfinite(initialization.scale) || initialization.scale < 0.0) {
        throw std::invalid_argument("initialization scale must be finite and non-negative");
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
        // Glorot balances forward and backward variance for activations symmetric about zero.
        // ReLU zeroes half its inputs and so halves the variance it passes on; He is the
        // factor-of-two correction for that.
        const double glorot = std::sqrt(6.0 / (fan_in + fan_out));
        const double he = std::sqrt(6.0 / fan_in);
        double limit = 0.0;
        switch (initialization.kind) {
        case Initialization::automatic:
            limit = layer.activation == Activation::rectified_linear ? he : glorot;
            break;
        case Initialization::glorot_uniform:
            limit = glorot;
            break;
        case Initialization::he_uniform:
            limit = he;
            break;
        case Initialization::fixed_uniform:
            limit = initialization.scale;
            break;
        case Initialization::zeros:
            limit = 0.0;
            break;
        }

        LayerParameters parameters;
        parameters.weights.assign(layer.output_size, std::vector<double>(layer.input_size, 0.0));
        parameters.biases.assign(layer.output_size, 0.0);
        for (auto& row : parameters.weights) {
            for (double& weight : row) {
                // A zero limit still consumes the engine, so switching schemes does not silently
                // shift every later draw.
                const double value = uniform(limit);
                weight = limit == 0.0 ? 0.0 : value;
            }
        }

        layer_offsets_.push_back(parameter_count_);
        parameter_count_ += layer.output_size * layer.input_size + layer.output_size;
        if (layer.normalization != Normalization::none) {
            // Scale starts at one and shift at zero, so a freshly built normalized layer passes
            // its standardized values through unchanged.
            parameters.scale.assign(layer.output_size, 1.0);
            parameters.shift.assign(layer.output_size, 0.0);
            parameter_count_ += 2 * layer.output_size;
        }
        if (layer.normalization == Normalization::batch) {
            parameters.running_mean.assign(layer.output_size, 0.0);
            parameters.running_variance.assign(layer.output_size, 1.0);
        }
        parameters_.push_back(std::move(parameters));
    }
}

bool FeedForwardNetwork::uses_batch_normalization() const noexcept {
    for (const DenseLayer& layer : layers_) {
        if (layer.normalization == Normalization::batch) {
            return true;
        }
    }
    return false;
}

bool FeedForwardNetwork::uses_dropout() const noexcept {
    for (const DenseLayer& layer : layers_) {
        if (layer.dropout_rate > 0.0) {
            return true;
        }
    }
    return false;
}

std::size_t FeedForwardNetwork::normalization_offset(const std::size_t layer) const {
    return layer_offsets_[layer] + layers_[layer].output_size * layers_[layer].input_size +
           layers_[layer].output_size;
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

namespace {

constexpr double normalization_epsilon = 1e-5;

// Standardizes `values` across its own entries and reports the statistics used. For layer
// normalization the entries are one sample's units; the batch path reuses the same algebra with the
// batch as the axis instead.
void standardize(const std::vector<double>& values, std::vector<double>& normalized, double& mean,
                 double& inverse_deviation) {
    const auto count = static_cast<double>(values.size());
    mean = 0.0;
    for (const double value : values) {
        mean += value;
    }
    mean /= count;

    double variance = 0.0;
    for (const double value : values) {
        const double centered = value - mean;
        variance += centered * centered;
    }
    variance /= count;

    inverse_deviation = 1.0 / std::sqrt(variance + normalization_epsilon);
    normalized.resize(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        normalized[index] = (values[index] - mean) * inverse_deviation;
    }
}

} // namespace

void FeedForwardNetwork::forward_into(const std::vector<double>& input, ForwardCache& cache,
                                      const bool training, std::mt19937* dropout_engine) const {
    if (input.size() != input_size()) {
        throw std::invalid_argument("input size does not match the network");
    }
    for (const double value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("inputs must be finite");
        }
    }

    // Sizes are fixed by the architecture, so the buffers are allocated on the first call and
    // reused afterwards; a training epoch then allocates nothing per sample.
    cache.activations.resize(layers_.size() + 1);
    cache.pre_activations.resize(layers_.size());
    cache.normalized.resize(layers_.size());
    cache.activation_inputs.resize(layers_.size());
    cache.activation_values.resize(layers_.size());
    cache.dropout_factors.resize(layers_.size());
    cache.inverse_deviation.resize(layers_.size());
    cache.activations[0] = input;

    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const DenseLayer& layer = layers_[index];
        const std::vector<double>& previous = cache.activations[index];
        cache.pre_activations[index].resize(layer.output_size);
        cache.activation_inputs[index].resize(layer.output_size);
        cache.activation_values[index].resize(layer.output_size);
        cache.activations[index + 1].resize(layer.output_size);

        for (std::size_t output = 0; output < layer.output_size; ++output) {
            double total = parameters_[index].biases[output];
            const std::vector<double>& row = parameters_[index].weights[output];
            for (std::size_t input_index = 0; input_index < layer.input_size; ++input_index) {
                total += row[input_index] * previous[input_index];
            }
            cache.pre_activations[index][output] = total;
        }

        std::vector<double>& scores = cache.pre_activations[index];
        if (layer.normalization == Normalization::layer) {
            double mean = 0.0;
            standardize(scores, cache.normalized[index], mean, cache.inverse_deviation[index]);
        } else if (layer.normalization == Normalization::batch) {
            // One sample has no batch, so the running statistics are the only sensible source.
            // Training goes through the batch path instead; this branch is inference.
            cache.normalized[index].resize(layer.output_size);
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                cache.normalized[index][output] =
                    (scores[output] - parameters_[index].running_mean[output]) /
                    std::sqrt(parameters_[index].running_variance[output] + normalization_epsilon);
            }
        }

        for (std::size_t output = 0; output < layer.output_size; ++output) {
            const double activation_input =
                layer.normalization == Normalization::none
                    ? scores[output]
                    : parameters_[index].scale[output] * cache.normalized[index][output] +
                          parameters_[index].shift[output];
            cache.activation_inputs[index][output] = activation_input;
            cache.activation_values[index][output] =
                apply_activation(activation_input, layer.activation);
            cache.activations[index + 1][output] = cache.activation_values[index][output];
        }

        // Inverted dropout: survivors are scaled up during training so inference needs no
        // rescaling and computes the expected value of the training-time network.
        if (training && layer.dropout_rate > 0.0 && dropout_engine != nullptr) {
            const double keep = 1.0 - layer.dropout_rate;
            constexpr double range = 4294967296.0; // std::mt19937::max() + 1
            cache.dropout_factors[index].resize(layer.output_size);
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                const double draw = static_cast<double>((*dropout_engine)()) / range;
                const double factor = draw < keep ? 1.0 / keep : 0.0;
                cache.dropout_factors[index][output] = factor;
                cache.activations[index + 1][output] *= factor;
            }
        } else {
            cache.dropout_factors[index].clear();
        }
    }
}

FeedForwardNetwork::ForwardCache
FeedForwardNetwork::forward_cache(const std::vector<double>& input) const {
    ForwardCache cache;
    forward_into(input, cache);
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

namespace {

// Backward pass through a standardization, shared by layer and batch normalization: they differ
// only in which axis the mean and variance were taken over.
//
// With n_i = (s_i - mu) / sigma over a group of `count` values, the derivative carries three terms:
// the direct one, the shift of the mean, and the shift of the variance. Collecting them gives
//   dL/ds_i = (1/sigma) * (dL/dn_i - mean(dL/dn) - n_i * mean(dL/dn . n)).
void backward_standardize(const std::vector<double>& gradient_normalized,
                          const std::vector<double>& normalized, const double inverse_deviation,
                          std::vector<double>& gradient_scores) {
    const std::size_t count = gradient_normalized.size();
    const auto scale = static_cast<double>(count);
    double sum = 0.0;
    double sum_times_normalized = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        sum += gradient_normalized[index];
        sum_times_normalized += gradient_normalized[index] * normalized[index];
    }

    gradient_scores.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        gradient_scores[index] =
            inverse_deviation * (gradient_normalized[index] - sum / scale -
                                 normalized[index] * sum_times_normalized / scale);
    }
}

} // namespace

void FeedForwardNetwork::accumulate_gradient(const std::vector<double>& features,
                                             const std::vector<double>* targets,
                                             const std::size_t label, const double scale,
                                             std::vector<double>& flat, Workspace& workspace,
                                             const bool training,
                                             std::mt19937* dropout_engine) const {
    forward_into(features, workspace.cache, training, dropout_engine);
    const ForwardCache& cache = workspace.cache;

    if (targets != nullptr) {
        workspace.delta =
            output_delta(cache.activation_inputs.back(), cache.activations.back(), *targets);
    } else {
        // The one-hot target is implied by `label`, so it is never materialized: under
        // softmax_cross_entropy the output delta is simply p - y.
        workspace.delta = softmax(cache.activations.back());
        workspace.delta[label] -= 1.0;
    }

    // delta holds dL/dy for the layer currently being visited, where y is what enters its
    // activation. Without normalization y is the affine output itself.
    for (std::size_t index = layers_.size(); index-- > 0;) {
        const DenseLayer& layer = layers_[index];
        const std::vector<double>& inputs = cache.activations[index];

        const std::size_t offset = layer_offsets_[index];
        const std::size_t bias_offset = offset + layer.output_size * layer.input_size;

        // A normalized layer splits dL/dy into its scale and shift gradients and then pushes the
        // rest back through the standardization to reach the affine output.
        if (layer.normalization != Normalization::none) {
            const std::size_t scale_offset = normalization_offset(index);
            const std::size_t shift_offset = scale_offset + layer.output_size;
            workspace.normalized.resize(layer.output_size);
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                flat[scale_offset + output] +=
                    scale * workspace.delta[output] * cache.normalized[index][output];
                flat[shift_offset + output] += scale * workspace.delta[output];
                workspace.normalized[output] =
                    workspace.delta[output] * parameters_[index].scale[output];
            }
            if (layer.normalization == Normalization::layer) {
                backward_standardize(workspace.normalized, cache.normalized[index],
                                     cache.inverse_deviation[index], workspace.delta);
            } else {
                // Batch normalization at inference divides by a constant, so its backward pass is
                // just that constant. Training never reaches here; it uses the batch path.
                for (std::size_t output = 0; output < layer.output_size; ++output) {
                    workspace.delta[output] =
                        workspace.normalized[output] /
                        std::sqrt(parameters_[index].running_variance[output] +
                                  normalization_epsilon);
                }
            }
        }

        for (std::size_t output = 0; output < layer.output_size; ++output) {
            const double unit_delta = workspace.delta[output];
            const std::size_t row_offset = offset + output * layer.input_size;
            for (std::size_t input = 0; input < layer.input_size; ++input) {
                flat[row_offset + input] += scale * unit_delta * inputs[input];
            }
            flat[bias_offset + output] += scale * unit_delta;
        }

        if (index == 0) {
            break;
        }
        // Propagate to the previous layer: delta_prev = (W^T delta), then through that layer's
        // dropout mask and activation slope.
        const DenseLayer& previous_layer = layers_[index - 1];
        workspace.previous_delta.assign(previous_layer.output_size, 0.0);
        const std::vector<double>& factors = cache.dropout_factors[index - 1];
        for (std::size_t input = 0; input < layer.input_size; ++input) {
            double total = 0.0;
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                total += parameters_[index].weights[output][input] * workspace.delta[output];
            }
            if (!factors.empty()) {
                total *= factors[input];
            }
            workspace.previous_delta[input] =
                total * activation_derivative(cache.activation_inputs[index - 1][input],
                                              cache.activation_values[index - 1][input],
                                              previous_layer.activation);
        }
        workspace.delta.swap(workspace.previous_delta);
    }
}

// Batch normalization makes the loss of one sample depend on every other sample in its batch, so
// its forward and backward passes cannot be done a sample at a time. This path forwards the whole
// batch, standardizes each unit across the batch dimension, and backpropagates layer by layer.
double FeedForwardNetwork::accumulate_batch_gradient(
    const FeatureMatrix& inputs, const std::vector<const std::vector<double>*>& targets,
    const std::vector<std::size_t>& labels, const double scale, std::vector<double>& flat,
    BatchWorkspace& workspace, const bool training, std::mt19937* dropout_engine,
    const bool update_running) {
    const std::size_t batch = inputs.size();
    const auto batch_scale = static_cast<double>(batch);
    const std::size_t depth = layers_.size();

    workspace.pre_activations.assign(depth, {});
    workspace.normalized.assign(depth, {});
    workspace.activations.assign(depth + 1, {});
    workspace.deltas.assign(depth, {});
    workspace.batch_mean.assign(depth, {});
    workspace.batch_inverse_deviation.assign(depth, {});
    workspace.activations[0] = inputs;

    std::vector<FeatureMatrix> activation_inputs(depth);
    std::vector<FeatureMatrix> activation_values(depth);
    std::vector<FeatureMatrix> dropout_factors(depth);

    for (std::size_t index = 0; index < depth; ++index) {
        const DenseLayer& layer = layers_[index];
        const FeatureMatrix& previous = workspace.activations[index];
        workspace.pre_activations[index].assign(batch, std::vector<double>(layer.output_size, 0.0));

        for (std::size_t sample = 0; sample < batch; ++sample) {
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                double total = parameters_[index].biases[output];
                const std::vector<double>& row = parameters_[index].weights[output];
                for (std::size_t input = 0; input < layer.input_size; ++input) {
                    total += row[input] * previous[sample][input];
                }
                workspace.pre_activations[index][sample][output] = total;
            }
        }

        workspace.normalized[index].assign(batch, std::vector<double>(layer.output_size, 0.0));
        if (layer.normalization == Normalization::batch) {
            workspace.batch_mean[index].assign(layer.output_size, 0.0);
            workspace.batch_inverse_deviation[index].assign(layer.output_size, 0.0);
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                double mean = 0.0;
                for (std::size_t sample = 0; sample < batch; ++sample) {
                    mean += workspace.pre_activations[index][sample][output];
                }
                mean /= batch_scale;

                double variance = 0.0;
                for (std::size_t sample = 0; sample < batch; ++sample) {
                    const double centered = workspace.pre_activations[index][sample][output] - mean;
                    variance += centered * centered;
                }
                variance /= batch_scale;

                const double inverse = 1.0 / std::sqrt(variance + normalization_epsilon);
                workspace.batch_mean[index][output] = mean;
                workspace.batch_inverse_deviation[index][output] = inverse;
                for (std::size_t sample = 0; sample < batch; ++sample) {
                    workspace.normalized[index][sample][output] =
                        (workspace.pre_activations[index][sample][output] - mean) * inverse;
                }

                // The running statistics inference will use. The variance is stored with the
                // unbiased (n - 1) correction, since the batch is a sample of the population.
                if (training && update_running) {
                    const double unbiased =
                        batch > 1 ? variance * batch_scale / (batch_scale - 1.0) : variance;
                    parameters_[index].running_mean[output] =
                        normalization_momentum_ * parameters_[index].running_mean[output] +
                        (1.0 - normalization_momentum_) * mean;
                    parameters_[index].running_variance[output] =
                        normalization_momentum_ * parameters_[index].running_variance[output] +
                        (1.0 - normalization_momentum_) * unbiased;
                }
            }
        } else if (layer.normalization == Normalization::layer) {
            workspace.batch_inverse_deviation[index].assign(batch, 0.0);
            for (std::size_t sample = 0; sample < batch; ++sample) {
                double mean = 0.0;
                double inverse = 0.0;
                standardize(workspace.pre_activations[index][sample],
                            workspace.normalized[index][sample], mean, inverse);
                workspace.batch_inverse_deviation[index][sample] = inverse;
            }
        }

        activation_inputs[index].assign(batch, std::vector<double>(layer.output_size, 0.0));
        activation_values[index].assign(batch, std::vector<double>(layer.output_size, 0.0));
        workspace.activations[index + 1].assign(batch, std::vector<double>(layer.output_size, 0.0));
        for (std::size_t sample = 0; sample < batch; ++sample) {
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                const double value = layer.normalization == Normalization::none
                                         ? workspace.pre_activations[index][sample][output]
                                         : parameters_[index].scale[output] *
                                                   workspace.normalized[index][sample][output] +
                                               parameters_[index].shift[output];
                activation_inputs[index][sample][output] = value;
                activation_values[index][sample][output] =
                    apply_activation(value, layer.activation);
                workspace.activations[index + 1][sample][output] =
                    activation_values[index][sample][output];
            }
        }

        if (training && layer.dropout_rate > 0.0 && dropout_engine != nullptr) {
            const double keep = 1.0 - layer.dropout_rate;
            constexpr double range = 4294967296.0;
            dropout_factors[index].assign(batch, std::vector<double>(layer.output_size, 1.0));
            for (std::size_t sample = 0; sample < batch; ++sample) {
                for (std::size_t output = 0; output < layer.output_size; ++output) {
                    const double draw = static_cast<double>((*dropout_engine)()) / range;
                    const double factor = draw < keep ? 1.0 / keep : 0.0;
                    dropout_factors[index][sample][output] = factor;
                    workspace.activations[index + 1][sample][output] *= factor;
                }
            }
        }
    }

    // Output deltas, one per sample, and the mean loss over the batch.
    workspace.deltas[depth - 1].assign(batch, {});
    double total_loss = 0.0;
    for (std::size_t sample = 0; sample < batch; ++sample) {
        if (!targets.empty()) {
            workspace.deltas[depth - 1][sample] =
                output_delta(activation_inputs[depth - 1][sample],
                             workspace.activations[depth][sample], *targets[sample]);
            total_loss += sample_loss(workspace.activations[depth][sample], *targets[sample]);
        } else {
            const std::vector<double>& logits = workspace.activations[depth][sample];
            workspace.deltas[depth - 1][sample] = softmax(logits);
            workspace.deltas[depth - 1][sample][labels[sample]] -= 1.0;
            const double largest = *std::max_element(logits.begin(), logits.end());
            double sum_of_exponentials = 0.0;
            for (const double logit : logits) {
                sum_of_exponentials += std::exp(logit - largest);
            }
            total_loss += largest + std::log(sum_of_exponentials) - logits[labels[sample]];
        }
    }

    std::vector<double> gradient_normalized(1);
    std::vector<double> gradient_scores(1);
    for (std::size_t index = depth; index-- > 0;) {
        const DenseLayer& layer = layers_[index];
        FeatureMatrix& deltas = workspace.deltas[index];
        const std::size_t offset = layer_offsets_[index];
        const std::size_t bias_offset = offset + layer.output_size * layer.input_size;

        if (layer.normalization != Normalization::none) {
            const std::size_t scale_offset = normalization_offset(index);
            const std::size_t shift_offset = scale_offset + layer.output_size;
            for (std::size_t sample = 0; sample < batch; ++sample) {
                for (std::size_t output = 0; output < layer.output_size; ++output) {
                    flat[scale_offset + output] += scale * deltas[sample][output] *
                                                   workspace.normalized[index][sample][output];
                    flat[shift_offset + output] += scale * deltas[sample][output];
                }
            }

            if (layer.normalization == Normalization::batch) {
                // The reduction runs down the batch for each unit independently.
                gradient_normalized.resize(batch);
                std::vector<double> column(batch);
                for (std::size_t output = 0; output < layer.output_size; ++output) {
                    for (std::size_t sample = 0; sample < batch; ++sample) {
                        gradient_normalized[sample] =
                            deltas[sample][output] * parameters_[index].scale[output];
                        column[sample] = workspace.normalized[index][sample][output];
                    }
                    backward_standardize(gradient_normalized, column,
                                         workspace.batch_inverse_deviation[index][output],
                                         gradient_scores);
                    for (std::size_t sample = 0; sample < batch; ++sample) {
                        deltas[sample][output] = gradient_scores[sample];
                    }
                }
            } else {
                // Layer normalization reduces across each sample's own units.
                for (std::size_t sample = 0; sample < batch; ++sample) {
                    gradient_normalized.resize(layer.output_size);
                    for (std::size_t output = 0; output < layer.output_size; ++output) {
                        gradient_normalized[output] =
                            deltas[sample][output] * parameters_[index].scale[output];
                    }
                    backward_standardize(gradient_normalized, workspace.normalized[index][sample],
                                         workspace.batch_inverse_deviation[index][sample],
                                         gradient_scores);
                    deltas[sample] = gradient_scores;
                }
            }
        }

        for (std::size_t sample = 0; sample < batch; ++sample) {
            const std::vector<double>& sample_inputs = workspace.activations[index][sample];
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                const double unit_delta = deltas[sample][output];
                const std::size_t row_offset = offset + output * layer.input_size;
                for (std::size_t input = 0; input < layer.input_size; ++input) {
                    flat[row_offset + input] += scale * unit_delta * sample_inputs[input];
                }
                flat[bias_offset + output] += scale * unit_delta;
            }
        }

        if (index == 0) {
            break;
        }
        const DenseLayer& previous_layer = layers_[index - 1];
        workspace.deltas[index - 1].assign(batch,
                                           std::vector<double>(previous_layer.output_size, 0.0));
        for (std::size_t sample = 0; sample < batch; ++sample) {
            for (std::size_t input = 0; input < layer.input_size; ++input) {
                double total = 0.0;
                for (std::size_t output = 0; output < layer.output_size; ++output) {
                    total += parameters_[index].weights[output][input] * deltas[sample][output];
                }
                if (!dropout_factors[index - 1].empty()) {
                    total *= dropout_factors[index - 1][sample][input];
                }
                workspace.deltas[index - 1][sample][input] =
                    total * activation_derivative(activation_inputs[index - 1][sample][input],
                                                  activation_values[index - 1][sample][input],
                                                  previous_layer.activation);
            }
        }
    }
    return total_loss / batch_scale;
}

double FeedForwardNetwork::batch_training_loss(const LabeledDataset& batch) {
    static_cast<void>(validate_labeled(batch));
    FeatureMatrix inputs;
    std::vector<std::size_t> labels;
    inputs.reserve(batch.size());
    labels.reserve(batch.size());
    for (const auto& sample : batch) {
        inputs.push_back(sample.features);
        labels.push_back(sample.label);
    }

    BatchWorkspace workspace;
    std::vector<double> discarded(parameter_count_, 0.0);
    return accumulate_batch_gradient(inputs, {}, labels, 0.0, discarded, workspace, true, nullptr,
                                     false);
}

std::vector<double> FeedForwardNetwork::batch_training_gradient(const LabeledDataset& batch) {
    static_cast<void>(validate_labeled(batch));
    FeatureMatrix inputs;
    std::vector<std::size_t> labels;
    inputs.reserve(batch.size());
    labels.reserve(batch.size());
    for (const auto& sample : batch) {
        inputs.push_back(sample.features);
        labels.push_back(sample.label);
    }

    BatchWorkspace workspace;
    std::vector<double> flat(parameter_count_, 0.0);
    static_cast<void>(accumulate_batch_gradient(inputs, {}, labels,
                                                1.0 / static_cast<double>(batch.size()), flat,
                                                workspace, true, nullptr, false));
    return flat;
}

void FeedForwardNetwork::subtract_update(const std::vector<double>& update) {
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const DenseLayer& layer = layers_[index];
        std::size_t cursor = layer_offsets_[index];
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            std::vector<double>& row = parameters_[index].weights[output];
            for (std::size_t input = 0; input < layer.input_size; ++input) {
                row[input] -= update[cursor++];
            }
        }
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            parameters_[index].biases[output] -= update[cursor++];
        }
    }
}

std::vector<double> FeedForwardNetwork::gradient(const SupervisedDataset& dataset) const {
    validate_for_network(dataset);

    std::vector<double> flat(parameter_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(dataset.size());
    Workspace workspace;
    for (const auto& sample : dataset) {
        accumulate_gradient(sample.features, &sample.targets, 0, scale, flat, workspace);
    }
    return flat;
}

void FeedForwardNetwork::require_classification() const {
    if (loss_ != Loss::softmax_cross_entropy) {
        throw std::invalid_argument(
            "the class-index overloads require the softmax_cross_entropy loss");
    }
}

std::size_t FeedForwardNetwork::validate_labeled(const LabeledDataset& dataset) const {
    require_classification();
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.feature_count != input_size()) {
        throw std::invalid_argument("feature count does not match the network input size");
    }
    if (shape.class_count > output_size()) {
        throw std::invalid_argument("dataset contains a label outside the network output size");
    }
    return shape.feature_count;
}

std::vector<double> FeedForwardNetwork::gradient(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    std::vector<double> flat(parameter_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(dataset.size());
    Workspace workspace;
    for (const auto& sample : dataset) {
        accumulate_gradient(sample.features, nullptr, sample.label, scale, flat, workspace);
    }
    return flat;
}

double FeedForwardNetwork::loss(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    double total = 0.0;
    ForwardCache cache;
    for (const auto& sample : dataset) {
        forward_into(sample.features, cache);
        const std::vector<double>& logits = cache.activations.back();
        const double largest = *std::max_element(logits.begin(), logits.end());
        double sum_of_exponentials = 0.0;
        for (const double logit : logits) {
            sum_of_exponentials += std::exp(logit - largest);
        }
        total += largest + std::log(sum_of_exponentials) - logits[sample.label];
    }
    return total / static_cast<double>(dataset.size());
}

double FeedForwardNetwork::accuracy(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict_class(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

std::vector<std::vector<std::size_t>>
FeedForwardNetwork::confusion_matrix(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    std::vector<std::vector<std::size_t>> confusion(output_size(),
                                                    std::vector<std::size_t>(output_size(), 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict_class(sample.features)];
    }
    return confusion;
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
        // Empty unless the layer normalizes, so an unnormalized network's layout is unchanged.
        flat.insert(flat.end(), layer.scale.begin(), layer.scale.end());
        flat.insert(flat.end(), layer.shift.begin(), layer.shift.end());
    }
    return flat;
}

std::vector<double> FeedForwardNetwork::running_statistics() const {
    std::vector<double> flat;
    for (const LayerParameters& layer : parameters_) {
        flat.insert(flat.end(), layer.running_mean.begin(), layer.running_mean.end());
        flat.insert(flat.end(), layer.running_variance.begin(), layer.running_variance.end());
    }
    return flat;
}

void FeedForwardNetwork::set_running_statistics(const std::vector<double>& values) {
    std::size_t cursor = 0;
    for (const LayerParameters& layer : parameters_) {
        cursor += layer.running_mean.size() + layer.running_variance.size();
    }
    if (values.size() != cursor) {
        throw std::invalid_argument("running statistics vector has the wrong size");
    }
    cursor = 0;
    for (LayerParameters& layer : parameters_) {
        for (double& value : layer.running_mean) {
            value = values[cursor++];
        }
        for (double& value : layer.running_variance) {
            if (values[cursor] < 0.0) {
                throw std::invalid_argument("a running variance must not be negative");
            }
            value = values[cursor++];
        }
    }
}

double FeedForwardNetwork::weight_penalty(const RegularizationConfig& regularization) const {
    if (!std::isfinite(regularization.l1) || regularization.l1 < 0.0 ||
        !std::isfinite(regularization.l2) || regularization.l2 < 0.0) {
        throw std::invalid_argument("regularization coefficients must be finite and non-negative");
    }
    if (regularization.l1 == 0.0 && regularization.l2 == 0.0) {
        return 0.0;
    }

    double absolute = 0.0;
    double squared = 0.0;
    for (const LayerParameters& layer : parameters_) {
        for (const std::vector<double>& row : layer.weights) {
            for (const double weight : row) {
                absolute += std::abs(weight);
                squared += weight * weight;
            }
        }
    }
    return regularization.l1 * absolute + 0.5 * regularization.l2 * squared;
}

void FeedForwardNetwork::add_penalty_gradient(const RegularizationConfig& regularization,
                                              std::vector<double>& flat) const {
    if (regularization.l1 == 0.0 && regularization.l2 == 0.0) {
        return;
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const DenseLayer& layer = layers_[index];
        std::size_t cursor = layer_offsets_[index];
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            for (std::size_t input = 0; input < layer.input_size; ++input) {
                const double weight = parameters_[index].weights[output][input];
                // |w| has no derivative at zero; the subgradient zero is the usual choice, and it
                // is what lets L1 leave a weight sitting exactly at zero instead of jittering.
                const double sign = weight > 0.0 ? 1.0 : (weight < 0.0 ? -1.0 : 0.0);
                flat[cursor++] += regularization.l1 * sign + regularization.l2 * weight;
            }
        }
        // Biases and normalization parameters are deliberately skipped.
    }
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
        for (double& value : layer.scale) {
            value = values[cursor++];
        }
        for (double& value : layer.shift) {
            value = values[cursor++];
        }
    }
}

namespace {

void validate_training_config(const NetworkTrainingConfig& config) {
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }
    if (!std::isfinite(config.target_loss) || config.target_loss < 0.0) {
        throw std::invalid_argument("target_loss must be finite and non-negative");
    }
    if (!std::isfinite(config.regularization.l1) || config.regularization.l1 < 0.0 ||
        !std::isfinite(config.regularization.l2) || config.regularization.l2 < 0.0) {
        throw std::invalid_argument("regularization coefficients must be finite and non-negative");
    }
    if (!std::isfinite(config.early_stopping.min_improvement) ||
        config.early_stopping.min_improvement < 0.0) {
        throw std::invalid_argument("min_improvement must be finite and non-negative");
    }
    if (!std::isfinite(config.normalization_momentum) || config.normalization_momentum < 0.0 ||
        config.normalization_momentum >= 1.0) {
        throw std::invalid_argument(
            "normalization_momentum must be finite and in the interval [0, 1)");
    }
}

} // namespace

// The fit overloads share this loop. `Dataset` is either a SupervisedDataset of explicit targets or
// a LabeledDataset of class indices; `accumulate` hides the difference, and `evaluate` reports the
// training objective while `validate` reports the held-out loss when one was supplied.
template <typename Dataset, typename Accumulate, typename Evaluate, typename Validate>
static NetworkTrainingResult
run_training(const Dataset& dataset, const NetworkTrainingConfig& config,
             const std::size_t parameter_count, Accumulate accumulate, Evaluate evaluate,
             Validate validate, const bool has_validation, FeedForwardNetwork& network) {
    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    const std::size_t batch_size =
        config.batch_size == 0 ? dataset.size() : std::min(config.batch_size, dataset.size());

    std::vector<double> batch_gradient(parameter_count, 0.0);
    NetworkTrainingResult result{0, false, {}, {}, 0, false};
    result.loss_per_epoch.reserve(config.max_epochs);

    std::vector<double> best_parameters;
    std::vector<double> best_statistics;
    double best_validation = std::numeric_limits<double>::infinity();
    std::size_t epochs_without_improvement = 0;

    for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        if (config.shuffle && batch_size < dataset.size()) {
            std::shuffle(order.begin(), order.end(), random_engine);
        }

        for (std::size_t begin = 0; begin < dataset.size(); begin += batch_size) {
            const std::size_t end = std::min(begin + batch_size, dataset.size());
            // Samples are indexed in place and the gradient buffer is reused, so a batch is never
            // copied and an epoch allocates nothing.
            std::fill(batch_gradient.begin(), batch_gradient.end(), 0.0);
            accumulate(order, begin, end, batch_gradient);
        }

        const double epoch_loss = evaluate();
        if (!std::isfinite(epoch_loss)) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        result.loss_per_epoch.push_back(epoch_loss);
        result.epochs = epoch;

        if (has_validation) {
            const double validation_loss = validate();
            result.validation_loss_per_epoch.push_back(validation_loss);
            if (validation_loss < best_validation - config.early_stopping.min_improvement) {
                best_validation = validation_loss;
                result.best_epoch = epoch;
                epochs_without_improvement = 0;
                // Keeping the best parameters is what makes early stopping usable: by the time the
                // curve has clearly turned, the peak is already several epochs behind.
                best_parameters = network.parameters();
                best_statistics = network.running_statistics();
            } else {
                ++epochs_without_improvement;
            }

            if (config.early_stopping.patience > 0 &&
                epochs_without_improvement >= config.early_stopping.patience) {
                result.stopped_early = true;
                network.set_parameters(best_parameters);
                network.set_running_statistics(best_statistics);
                return result;
            }
        }

        if (config.target_loss > 0.0 && epoch_loss <= config.target_loss) {
            result.converged = true;
            return result;
        }
    }

    // A completed run still restores the best epoch when early stopping was requested.
    if (has_validation && config.early_stopping.patience > 0 && !best_parameters.empty()) {
        network.set_parameters(best_parameters);
        network.set_running_statistics(best_statistics);
    }
    return result;
}

NetworkTrainingResult FeedForwardNetwork::fit(const SupervisedDataset& dataset,
                                              const NetworkTrainingConfig& config) {
    return fit(dataset, SupervisedDataset{}, config);
}

NetworkTrainingResult FeedForwardNetwork::fit(const LabeledDataset& dataset,
                                              const NetworkTrainingConfig& config) {
    return fit(dataset, LabeledDataset{}, config);
}

NetworkTrainingResult FeedForwardNetwork::fit(const SupervisedDataset& dataset,
                                              const SupervisedDataset& validation,
                                              const NetworkTrainingConfig& config) {
    validate_for_network(dataset);
    validate_training_config(config);
    const bool has_validation = !validation.empty();
    if (has_validation) {
        validate_for_network(validation);
    }
    if (config.early_stopping.patience > 0 && !has_validation) {
        throw std::invalid_argument("early stopping requires a validation split");
    }
    normalization_momentum_ = config.normalization_momentum;

    Workspace workspace;
    BatchWorkspace batch_workspace;
    Optimizer optimizer{config.optimizer, parameter_count_};
    std::mt19937 dropout_engine{config.seed + 1u};
    std::vector<double> update;
    const bool batched = uses_batch_normalization();

    return run_training(
        dataset, config, parameter_count_,
        [&](const std::vector<std::size_t>& order, const std::size_t begin, const std::size_t end,
            std::vector<double>& flat) {
            const double scale = 1.0 / static_cast<double>(end - begin);
            if (batched) {
                FeatureMatrix inputs;
                std::vector<const std::vector<double>*> targets;
                inputs.reserve(end - begin);
                targets.reserve(end - begin);
                for (std::size_t position = begin; position < end; ++position) {
                    inputs.push_back(dataset[order[position]].features);
                    targets.push_back(&dataset[order[position]].targets);
                }
                accumulate_batch_gradient(inputs, targets, {}, scale, flat, batch_workspace, true,
                                          &dropout_engine);
            } else {
                for (std::size_t position = begin; position < end; ++position) {
                    const SupervisedSample& sample = dataset[order[position]];
                    accumulate_gradient(sample.features, &sample.targets, 0, scale, flat, workspace,
                                        true, &dropout_engine);
                }
            }
            add_penalty_gradient(config.regularization, flat);
            optimizer.compute_update(flat, config.learning_rate, update);
            subtract_update(update);
        },
        [&] { return loss(dataset) + weight_penalty(config.regularization); },
        [&] { return has_validation ? loss(validation) : 0.0; }, has_validation, *this);
}

NetworkTrainingResult FeedForwardNetwork::fit(const LabeledDataset& dataset,
                                              const LabeledDataset& validation,
                                              const NetworkTrainingConfig& config) {
    static_cast<void>(validate_labeled(dataset));
    validate_training_config(config);
    const bool has_validation = !validation.empty();
    if (has_validation) {
        static_cast<void>(validate_labeled(validation));
    }
    if (config.early_stopping.patience > 0 && !has_validation) {
        throw std::invalid_argument("early stopping requires a validation split");
    }
    normalization_momentum_ = config.normalization_momentum;

    Workspace workspace;
    BatchWorkspace batch_workspace;
    Optimizer optimizer{config.optimizer, parameter_count_};
    std::mt19937 dropout_engine{config.seed + 1u};
    std::vector<double> update;
    const bool batched = uses_batch_normalization();

    return run_training(
        dataset, config, parameter_count_,
        [&](const std::vector<std::size_t>& order, const std::size_t begin, const std::size_t end,
            std::vector<double>& flat) {
            const double scale = 1.0 / static_cast<double>(end - begin);
            if (batched) {
                FeatureMatrix inputs;
                std::vector<std::size_t> labels;
                inputs.reserve(end - begin);
                labels.reserve(end - begin);
                for (std::size_t position = begin; position < end; ++position) {
                    inputs.push_back(dataset[order[position]].features);
                    labels.push_back(dataset[order[position]].label);
                }
                accumulate_batch_gradient(inputs, {}, labels, scale, flat, batch_workspace, true,
                                          &dropout_engine);
            } else {
                for (std::size_t position = begin; position < end; ++position) {
                    const LabeledSample& sample = dataset[order[position]];
                    accumulate_gradient(sample.features, nullptr, sample.label, scale, flat,
                                        workspace, true, &dropout_engine);
                }
            }
            add_penalty_gradient(config.regularization, flat);
            optimizer.compute_update(flat, config.learning_rate, update);
            subtract_update(update);
        },
        [&] { return loss(dataset) + weight_penalty(config.regularization); },
        [&] { return has_validation ? loss(validation) : 0.0; }, has_validation, *this);
}

void FeedForwardNetwork::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }

    stream << "ml_scratch_feedforward 2\n"
           << "loss " << static_cast<int>(loss_) << '\n'
           << "layers " << layers_.size() << '\n'
           << std::setprecision(17);
    for (const DenseLayer& layer : layers_) {
        stream << layer.input_size << ' ' << layer.output_size << ' '
               << static_cast<int>(layer.activation) << ' ' << static_cast<int>(layer.normalization)
               << ' ' << layer.dropout_rate << '\n';
    }

    const std::vector<double> flat = parameters();
    stream << "parameters " << flat.size() << '\n';
    // 17 significant digits round-trip an IEEE-754 double exactly.
    for (const double value : flat) {
        stream << value << '\n';
    }

    // Batch normalization's running statistics are not parameters, but inference is wrong without
    // them, so a checkpoint that omitted them would restore a different function.
    const std::vector<double> statistics = running_statistics();
    stream << "statistics " << statistics.size() << '\n';
    for (const double value : statistics) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

FeedForwardNetwork FeedForwardNetwork::load(const std::string& path) {
    std::ifstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint: " + path);
    }

    const auto expect = [&stream, &path](const std::string& keyword) {
        std::string token;
        if (!(stream >> token) || token != keyword) {
            throw std::runtime_error("malformed checkpoint " + path + ": expected " + keyword);
        }
    };
    const auto read_size = [&stream, &path] {
        long long value = 0;
        if (!(stream >> value) || value < 0) {
            throw std::runtime_error("malformed checkpoint " + path + ": expected a size");
        }
        return static_cast<std::size_t>(value);
    };

    expect("ml_scratch_feedforward");
    const std::size_t version = read_size();
    if (version != 1 && version != 2) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("loss");
    const std::size_t loss_value = read_size();
    if (loss_value > static_cast<std::size_t>(Loss::softmax_cross_entropy)) {
        throw std::runtime_error("unknown loss in checkpoint " + path);
    }
    expect("layers");
    const std::size_t layer_count = read_size();
    if (layer_count == 0) {
        throw std::runtime_error("checkpoint declares no layers: " + path);
    }

    std::vector<DenseLayer> layers;
    layers.reserve(layer_count);
    for (std::size_t index = 0; index < layer_count; ++index) {
        const std::size_t input_size = read_size();
        const std::size_t output_size = read_size();
        const std::size_t activation = read_size();
        if (activation > static_cast<std::size_t>(Activation::rectified_linear)) {
            throw std::runtime_error("unknown activation in checkpoint " + path);
        }

        // Version 1 predates normalization and dropout, so those fields are absent and default.
        Normalization normalization = Normalization::none;
        double dropout_rate = 0.0;
        if (version >= 2) {
            const std::size_t encoded = read_size();
            if (encoded > static_cast<std::size_t>(Normalization::layer)) {
                throw std::runtime_error("unknown normalization in checkpoint " + path);
            }
            normalization = static_cast<Normalization>(encoded);
            if (!(stream >> dropout_rate) || !std::isfinite(dropout_rate) || dropout_rate < 0.0 ||
                dropout_rate >= 1.0) {
                throw std::runtime_error("invalid dropout rate in checkpoint " + path);
            }
        }
        layers.push_back({input_size, output_size, static_cast<Activation>(activation),
                          normalization, dropout_rate});
    }

    // The constructor re-validates the architecture, so a corrupted shape is rejected here.
    FeedForwardNetwork network{std::move(layers), static_cast<Loss>(loss_value)};

    expect("parameters");
    const std::size_t count = read_size();
    if (count != network.parameter_count()) {
        throw std::runtime_error("checkpoint parameter count does not match its architecture: " +
                                 path);
    }
    std::vector<double> values(count, 0.0);
    for (double& value : values) {
        if (!(stream >> value)) {
            throw std::runtime_error("truncated checkpoint: " + path);
        }
    }
    network.set_parameters(values);

    if (version >= 2) {
        expect("statistics");
        const std::size_t statistics_count = read_size();
        if (statistics_count != network.running_statistics().size()) {
            throw std::runtime_error("checkpoint statistics count does not match its "
                                     "architecture: " +
                                     path);
        }
        std::vector<double> statistics(statistics_count, 0.0);
        for (double& value : statistics) {
            if (!(stream >> value)) {
                throw std::runtime_error("truncated checkpoint: " + path);
            }
        }
        network.set_running_statistics(statistics);
    }
    return network;
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

GradientCheckResult check_batch_gradient(FeedForwardNetwork& network, const LabeledDataset& batch,
                                         const double epsilon, const double tolerance) {
    if (!std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("epsilon must be finite and positive");
    }
    if (network.uses_dropout()) {
        throw std::invalid_argument(
            "gradient checking needs a deterministic loss, so dropout must be disabled");
    }

    const std::vector<double> analytic = network.batch_training_gradient(batch);
    const std::vector<double> original = network.parameters();
    std::vector<double> perturbed = original;
    std::vector<double> numerical(original.size(), 0.0);

    // The batch is perturbed as one unit, because batch normalization makes the loss a function of
    // the whole batch rather than a sum over independent samples.
    for (std::size_t index = 0; index < original.size(); ++index) {
        perturbed[index] = original[index] + epsilon;
        network.set_parameters(perturbed);
        const double raised = network.batch_training_loss(batch);

        perturbed[index] = original[index] - epsilon;
        network.set_parameters(perturbed);
        const double lowered = network.batch_training_loss(batch);

        perturbed[index] = original[index];
        numerical[index] = (raised - lowered) / (2.0 * epsilon);
    }
    network.set_parameters(original);
    return check_gradient(analytic, numerical, tolerance);
}

} // namespace ml_scratch
