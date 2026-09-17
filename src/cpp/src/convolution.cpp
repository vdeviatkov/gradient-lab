#include "ml_scratch/convolution.hpp"

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

double activation_derivative(const double pre_activation, const double activation_value,
                             const Activation activation) {
    switch (activation) {
    case Activation::sigmoid:
        return activation_value * (1.0 - activation_value);
    case Activation::hyperbolic_tangent:
        return 1.0 - activation_value * activation_value;
    case Activation::rectified_linear:
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

// The number of window positions that fit, which is what fixes each layer's output size.
std::size_t output_extent(const std::size_t input, const std::size_t window,
                          const std::size_t stride, const std::size_t padding) {
    const std::size_t padded = input + 2 * padding;
    if (window == 0 || stride == 0 || padded < window) {
        throw std::invalid_argument("layer window does not fit its input");
    }
    return (padded - window) / stride + 1;
}

} // namespace

ConvLayerSpec convolution(const std::size_t filters, const std::size_t kernel_size,
                          const Activation activation, const std::size_t stride,
                          const std::size_t padding) {
    ConvLayerSpec spec;
    spec.kind = ConvLayerKind::convolution;
    spec.activation = activation;
    spec.filters = filters;
    spec.kernel_size = kernel_size;
    spec.stride = stride;
    spec.padding = padding;
    return spec;
}

ConvLayerSpec residual_convolution(const std::size_t filters, const std::size_t kernel_size,
                                   const Activation activation) {
    if (kernel_size % 2 == 0) {
        throw std::invalid_argument("a residual convolution needs an odd kernel to keep its shape");
    }
    ConvLayerSpec spec = convolution(filters, kernel_size, activation, 1, (kernel_size - 1) / 2);
    spec.residual = true;
    return spec;
}

ConvLayerSpec max_pooling(const std::size_t pool_size, const std::size_t stride) {
    ConvLayerSpec spec;
    spec.kind = ConvLayerKind::max_pooling;
    spec.pool_size = pool_size;
    spec.pool_stride = stride == 0 ? pool_size : stride;
    return spec;
}

ConvLayerSpec average_pooling(const std::size_t pool_size, const std::size_t stride) {
    ConvLayerSpec spec;
    spec.kind = ConvLayerKind::average_pooling;
    spec.pool_size = pool_size;
    spec.pool_stride = stride == 0 ? pool_size : stride;
    return spec;
}

ConvLayerSpec flatten() {
    ConvLayerSpec spec;
    spec.kind = ConvLayerKind::flatten;
    return spec;
}

ConvLayerSpec dense(const std::size_t units, const Activation activation) {
    ConvLayerSpec spec;
    spec.kind = ConvLayerKind::dense;
    spec.units = units;
    spec.activation = activation;
    return spec;
}

ConvolutionalNetwork::ConvolutionalNetwork(const TensorShape input_shape,
                                           std::vector<ConvLayerSpec> layers, const Loss loss,
                                           const std::uint32_t seed)
    : layers_(std::move(layers)), loss_(loss) {
    if (layers_.empty()) {
        throw std::invalid_argument("a network needs at least one layer");
    }
    if (input_shape.size() == 0) {
        throw std::invalid_argument("the input shape must be non-empty");
    }

    std::mt19937 random_engine{seed};
    const auto uniform = [&random_engine](const double limit) {
        constexpr double range = 4294967296.0; // std::mt19937::max() + 1
        const double unit = static_cast<double>(random_engine()) / range;
        return -limit + 2.0 * limit * unit;
    };

    shapes_.push_back(input_shape);
    for (const ConvLayerSpec& layer : layers_) {
        const TensorShape input = shapes_.back();
        LayerParameters parameters;
        TensorShape output = input;

        switch (layer.kind) {
        case ConvLayerKind::convolution: {
            if (layer.filters == 0 || layer.kernel_size == 0 || layer.stride == 0) {
                throw std::invalid_argument("a convolution needs filters, a kernel, and a stride");
            }
            output.channels = layer.filters;
            output.height =
                output_extent(input.height, layer.kernel_size, layer.stride, layer.padding);
            output.width =
                output_extent(input.width, layer.kernel_size, layer.stride, layer.padding);

            // One kernel per filter, spanning every input channel. Fan-in counts every weight that
            // contributes to one output value, which is what the He and Glorot scalings need.
            const std::size_t weights_per_filter =
                input.channels * layer.kernel_size * layer.kernel_size;
            const auto fan_in = static_cast<double>(weights_per_filter);
            const auto fan_out =
                static_cast<double>(layer.filters * layer.kernel_size * layer.kernel_size);
            const double limit = layer.activation == Activation::rectified_linear
                                     ? std::sqrt(6.0 / fan_in)
                                     : std::sqrt(6.0 / (fan_in + fan_out));
            parameters.weights.resize(layer.filters * weights_per_filter);
            for (double& weight : parameters.weights) {
                weight = uniform(limit);
            }
            parameters.biases.assign(layer.filters, 0.0);
            if (layer.residual && !(output == input)) {
                throw std::invalid_argument(
                    "a residual convolution must preserve its input shape, since the skip adds the "
                    "input to the pre-activation");
            }
            break;
        }
        case ConvLayerKind::max_pooling:
        case ConvLayerKind::average_pooling: {
            if (layer.pool_size == 0 || layer.pool_stride == 0) {
                throw std::invalid_argument("a pooling layer needs a window and a stride");
            }
            output.height = output_extent(input.height, layer.pool_size, layer.pool_stride, 0);
            output.width = output_extent(input.width, layer.pool_size, layer.pool_stride, 0);
            break;
        }
        case ConvLayerKind::flatten:
            output = {1, 1, input.size()};
            break;
        case ConvLayerKind::dense: {
            if (layer.units == 0) {
                throw std::invalid_argument("a dense layer needs units");
            }
            if (input.channels != 1 || input.height != 1) {
                throw std::invalid_argument("a dense layer needs a flattened input");
            }
            const auto fan_in = static_cast<double>(input.width);
            const auto fan_out = static_cast<double>(layer.units);
            const double limit = layer.activation == Activation::rectified_linear
                                     ? std::sqrt(6.0 / fan_in)
                                     : std::sqrt(6.0 / (fan_in + fan_out));
            parameters.weights.resize(layer.units * input.width);
            for (double& weight : parameters.weights) {
                weight = uniform(limit);
            }
            parameters.biases.assign(layer.units, 0.0);
            output = {1, 1, layer.units};
            break;
        }
        }

        layer_offsets_.push_back(parameter_count_);
        parameter_count_ += parameters.weights.size() + parameters.biases.size();
        parameters_.push_back(std::move(parameters));
        shapes_.push_back(output);
    }

    const bool logit_loss = loss_ != Loss::mean_squared_error;
    if (logit_loss && layers_.back().activation != Activation::identity) {
        throw std::invalid_argument(
            "cross-entropy losses apply their own output transform, so the final layer must be "
            "linear");
    }
    if (loss_ == Loss::softmax_cross_entropy && output_shape().size() < 2) {
        throw std::invalid_argument("softmax_cross_entropy needs at least two outputs");
    }
}

void ConvolutionalNetwork::forward_into(const std::vector<double>& input, Cache& cache) const {
    if (input.size() != input_shape().size()) {
        throw std::invalid_argument("input size does not match the network");
    }
    for (const double value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("inputs must be finite");
        }
    }

    cache.activations.resize(layers_.size() + 1);
    cache.pre_activations.resize(layers_.size());
    cache.argmax.resize(layers_.size());
    cache.activations[0] = input;

    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const ConvLayerSpec& layer = layers_[index];
        const TensorShape in = shapes_[index];
        const TensorShape out = shapes_[index + 1];
        const std::vector<double>& source = cache.activations[index];
        cache.pre_activations[index].assign(out.size(), 0.0);
        cache.activations[index + 1].assign(out.size(), 0.0);

        switch (layer.kind) {
        case ConvLayerKind::convolution: {
            const std::size_t kernel = layer.kernel_size;
            const std::size_t weights_per_filter = in.channels * kernel * kernel;
            for (std::size_t filter = 0; filter < layer.filters; ++filter) {
                const std::size_t filter_offset = filter * weights_per_filter;
                for (std::size_t row = 0; row < out.height; ++row) {
                    for (std::size_t column = 0; column < out.width; ++column) {
                        double total = parameters_[index].biases[filter];
                        for (std::size_t channel = 0; channel < in.channels; ++channel) {
                            for (std::size_t ky = 0; ky < kernel; ++ky) {
                                // Padding is implicit: positions outside the input contribute zero,
                                // so they are simply skipped rather than materialized.
                                const std::size_t source_row =
                                    row * layer.stride + ky - layer.padding;
                                if (row * layer.stride + ky < layer.padding ||
                                    source_row >= in.height) {
                                    continue;
                                }
                                for (std::size_t kx = 0; kx < kernel; ++kx) {
                                    const std::size_t source_column =
                                        column * layer.stride + kx - layer.padding;
                                    if (column * layer.stride + kx < layer.padding ||
                                        source_column >= in.width) {
                                        continue;
                                    }
                                    total += parameters_[index]
                                                 .weights[filter_offset +
                                                          (channel * kernel + ky) * kernel + kx] *
                                             source[in.index(channel, source_row, source_column)];
                                }
                            }
                        }
                        const std::size_t target = out.index(filter, row, column);
                        // The identity term joins the affine output before the activation.
                        if (layer.residual) {
                            total += source[target];
                        }
                        cache.pre_activations[index][target] = total;
                        cache.activations[index + 1][target] =
                            apply_activation(total, layer.activation);
                    }
                }
            }
            break;
        }
        case ConvLayerKind::max_pooling: {
            cache.argmax[index].assign(out.size(), 0);
            for (std::size_t channel = 0; channel < in.channels; ++channel) {
                for (std::size_t row = 0; row < out.height; ++row) {
                    for (std::size_t column = 0; column < out.width; ++column) {
                        double best = -std::numeric_limits<double>::infinity();
                        std::size_t best_index = 0;
                        for (std::size_t wy = 0; wy < layer.pool_size; ++wy) {
                            for (std::size_t wx = 0; wx < layer.pool_size; ++wx) {
                                const std::size_t source_index =
                                    in.index(channel, row * layer.pool_stride + wy,
                                             column * layer.pool_stride + wx);
                                if (source[source_index] > best) {
                                    best = source[source_index];
                                    best_index = source_index;
                                }
                            }
                        }
                        const std::size_t target = out.index(channel, row, column);
                        cache.activations[index + 1][target] = best;
                        // Only the winning position receives gradient, so its index is cached.
                        cache.argmax[index][target] = best_index;
                    }
                }
            }
            break;
        }
        case ConvLayerKind::average_pooling: {
            const auto window = static_cast<double>(layer.pool_size * layer.pool_size);
            for (std::size_t channel = 0; channel < in.channels; ++channel) {
                for (std::size_t row = 0; row < out.height; ++row) {
                    for (std::size_t column = 0; column < out.width; ++column) {
                        double total = 0.0;
                        for (std::size_t wy = 0; wy < layer.pool_size; ++wy) {
                            for (std::size_t wx = 0; wx < layer.pool_size; ++wx) {
                                total += source[in.index(channel, row * layer.pool_stride + wy,
                                                         column * layer.pool_stride + wx)];
                            }
                        }
                        cache.activations[index + 1][out.index(channel, row, column)] =
                            total / window;
                    }
                }
            }
            break;
        }
        case ConvLayerKind::flatten:
            // A pure reinterpretation: the layout is already channel-major and contiguous.
            cache.activations[index + 1] = source;
            break;
        case ConvLayerKind::dense: {
            const std::size_t inputs = in.size();
            for (std::size_t unit = 0; unit < layer.units; ++unit) {
                double total = parameters_[index].biases[unit];
                const std::size_t row_offset = unit * inputs;
                for (std::size_t input_index = 0; input_index < inputs; ++input_index) {
                    total +=
                        parameters_[index].weights[row_offset + input_index] * source[input_index];
                }
                cache.pre_activations[index][unit] = total;
                cache.activations[index + 1][unit] = apply_activation(total, layer.activation);
            }
            break;
        }
        }
    }
}

std::vector<double> ConvolutionalNetwork::forward(const std::vector<double>& input) const {
    Cache cache;
    forward_into(input, cache);
    return cache.activations.back();
}

std::vector<double> ConvolutionalNetwork::predict(const std::vector<double>& input) const {
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

std::size_t ConvolutionalNetwork::predict_class(const std::vector<double>& input) const {
    const std::vector<double> outputs = forward(input);
    std::size_t best = 0;
    for (std::size_t index = 1; index < outputs.size(); ++index) {
        if (outputs[index] > outputs[best]) {
            best = index;
        }
    }
    return best;
}

void ConvolutionalNetwork::validate_dataset(const LabeledDataset& dataset) const {
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.feature_count != input_shape().size()) {
        throw std::invalid_argument("feature count does not match the network input shape");
    }
    if (loss_ != Loss::softmax_cross_entropy) {
        throw std::invalid_argument("class indices require the softmax_cross_entropy loss");
    }
    if (shape.class_count > output_shape().size()) {
        throw std::invalid_argument("dataset contains a label outside the network output size");
    }
}

double ConvolutionalNetwork::loss(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    Cache cache;
    double total = 0.0;
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

double ConvolutionalNetwork::accuracy(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict_class(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

std::vector<std::vector<std::size_t>>
ConvolutionalNetwork::confusion_matrix(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    const std::size_t classes = output_shape().size();
    std::vector<std::vector<std::size_t>> confusion(classes, std::vector<std::size_t>(classes, 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict_class(sample.features)];
    }
    return confusion;
}

void ConvolutionalNetwork::accumulate_gradient(const std::vector<double>& features,
                                               const std::size_t label, const double scale,
                                               std::vector<double>& flat, Cache& cache,
                                               std::vector<double>& delta,
                                               std::vector<double>& previous) const {
    forward_into(features, cache);

    // Composing softmax with cross-entropy collapses the output delta to p - y.
    delta = softmax(cache.activations.back());
    delta[label] -= 1.0;

    for (std::size_t index = layers_.size(); index-- > 0;) {
        const ConvLayerSpec& layer = layers_[index];
        const TensorShape in = shapes_[index];
        const TensorShape out = shapes_[index + 1];
        const std::vector<double>& source = cache.activations[index];
        const std::size_t offset = layer_offsets_[index];
        const bool propagate = index > 0;
        previous.assign(propagate ? in.size() : 0, 0.0);

        switch (layer.kind) {
        case ConvLayerKind::convolution: {
            const std::size_t kernel = layer.kernel_size;
            const std::size_t weights_per_filter = in.channels * kernel * kernel;
            const std::size_t bias_offset = offset + layer.filters * weights_per_filter;

            for (std::size_t filter = 0; filter < layer.filters; ++filter) {
                const std::size_t filter_offset = filter * weights_per_filter;
                for (std::size_t row = 0; row < out.height; ++row) {
                    for (std::size_t column = 0; column < out.width; ++column) {
                        const std::size_t target = out.index(filter, row, column);
                        // Through the activation first: delta enters as dL/da, leaves as dL/dz.
                        const double unit_delta =
                            delta[target] *
                            activation_derivative(cache.pre_activations[index][target],
                                                  cache.activations[index + 1][target],
                                                  layer.activation);
                        flat[bias_offset + filter] += scale * unit_delta;
                        // The skip carries the gradient straight back to the matching input
                        // position, bypassing the kernel entirely.
                        if (layer.residual && propagate) {
                            previous[target] += unit_delta;
                        }

                        for (std::size_t channel = 0; channel < in.channels; ++channel) {
                            for (std::size_t ky = 0; ky < kernel; ++ky) {
                                const std::size_t source_row =
                                    row * layer.stride + ky - layer.padding;
                                if (row * layer.stride + ky < layer.padding ||
                                    source_row >= in.height) {
                                    continue;
                                }
                                for (std::size_t kx = 0; kx < kernel; ++kx) {
                                    const std::size_t source_column =
                                        column * layer.stride + kx - layer.padding;
                                    if (column * layer.stride + kx < layer.padding ||
                                        source_column >= in.width) {
                                        continue;
                                    }
                                    const std::size_t weight_index =
                                        filter_offset + (channel * kernel + ky) * kernel + kx;
                                    const std::size_t source_index =
                                        in.index(channel, source_row, source_column);
                                    // A kernel weight is shared across every position it visits,
                                    // so its gradient sums over all of them.
                                    flat[offset + weight_index] +=
                                        scale * unit_delta * source[source_index];
                                    if (propagate) {
                                        previous[source_index] +=
                                            unit_delta * parameters_[index].weights[weight_index];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            break;
        }
        case ConvLayerKind::max_pooling:
            if (propagate) {
                // Only the winning input affected the output, so only it receives gradient.
                for (std::size_t target = 0; target < out.size(); ++target) {
                    previous[cache.argmax[index][target]] += delta[target];
                }
            }
            break;
        case ConvLayerKind::average_pooling:
            if (propagate) {
                const auto window = static_cast<double>(layer.pool_size * layer.pool_size);
                for (std::size_t channel = 0; channel < in.channels; ++channel) {
                    for (std::size_t row = 0; row < out.height; ++row) {
                        for (std::size_t column = 0; column < out.width; ++column) {
                            const double share = delta[out.index(channel, row, column)] / window;
                            for (std::size_t wy = 0; wy < layer.pool_size; ++wy) {
                                for (std::size_t wx = 0; wx < layer.pool_size; ++wx) {
                                    previous[in.index(channel, row * layer.pool_stride + wy,
                                                      column * layer.pool_stride + wx)] += share;
                                }
                            }
                        }
                    }
                }
            }
            break;
        case ConvLayerKind::flatten:
            if (propagate) {
                previous = delta;
            }
            break;
        case ConvLayerKind::dense: {
            const std::size_t inputs = in.size();
            const std::size_t bias_offset = offset + layer.units * inputs;
            for (std::size_t unit = 0; unit < layer.units; ++unit) {
                const double unit_delta =
                    delta[unit] * activation_derivative(cache.pre_activations[index][unit],
                                                        cache.activations[index + 1][unit],
                                                        layer.activation);
                const std::size_t row_offset = offset + unit * inputs;
                for (std::size_t input_index = 0; input_index < inputs; ++input_index) {
                    flat[row_offset + input_index] += scale * unit_delta * source[input_index];
                    if (propagate) {
                        previous[input_index] +=
                            unit_delta * parameters_[index].weights[unit * inputs + input_index];
                    }
                }
                flat[bias_offset + unit] += scale * unit_delta;
            }
            break;
        }
        }

        if (!propagate) {
            break;
        }
        delta.swap(previous);
    }
}

std::vector<double> ConvolutionalNetwork::gradient(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    std::vector<double> flat(parameter_count_, 0.0);
    const double scale = 1.0 / static_cast<double>(dataset.size());
    Cache cache;
    std::vector<double> delta;
    std::vector<double> previous;
    for (const auto& sample : dataset) {
        accumulate_gradient(sample.features, sample.label, scale, flat, cache, delta, previous);
    }
    return flat;
}

std::vector<double> ConvolutionalNetwork::parameters() const {
    std::vector<double> flat;
    flat.reserve(parameter_count_);
    for (const LayerParameters& layer : parameters_) {
        flat.insert(flat.end(), layer.weights.begin(), layer.weights.end());
        flat.insert(flat.end(), layer.biases.begin(), layer.biases.end());
    }
    return flat;
}

void ConvolutionalNetwork::set_parameters(const std::vector<double>& values) {
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
        for (double& weight : layer.weights) {
            weight = values[cursor++];
        }
        for (double& bias : layer.biases) {
            bias = values[cursor++];
        }
    }
}

void ConvolutionalNetwork::subtract_update(const std::vector<double>& update) {
    std::size_t cursor = 0;
    for (LayerParameters& layer : parameters_) {
        for (double& weight : layer.weights) {
            weight -= update[cursor++];
        }
        for (double& bias : layer.biases) {
            bias -= update[cursor++];
        }
    }
}

NetworkTrainingResult ConvolutionalNetwork::fit(const LabeledDataset& dataset,
                                                const NetworkTrainingConfig& config) {
    return fit(dataset, LabeledDataset{}, config);
}

NetworkTrainingResult ConvolutionalNetwork::fit(const LabeledDataset& dataset,
                                                const LabeledDataset& validation,
                                                const NetworkTrainingConfig& config) {
    validate_dataset(dataset);
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }
    const bool has_validation = !validation.empty();
    if (has_validation) {
        validate_dataset(validation);
    }
    if (config.early_stopping.patience > 0 && !has_validation) {
        throw std::invalid_argument("early stopping requires a validation split");
    }

    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    const std::size_t batch_size =
        config.batch_size == 0 ? dataset.size() : std::min(config.batch_size, dataset.size());

    Optimizer optimizer{config.optimizer, parameter_count_};
    std::vector<double> batch_gradient(parameter_count_, 0.0);
    std::vector<double> update;
    Cache cache;
    std::vector<double> delta;
    std::vector<double> previous;

    NetworkTrainingResult result{0, false, {}, {}, 0, false};
    result.loss_per_epoch.reserve(config.max_epochs);
    std::vector<double> best_parameters;
    double best_validation = std::numeric_limits<double>::infinity();
    std::size_t epochs_without_improvement = 0;

    for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        if (config.shuffle && batch_size < dataset.size()) {
            std::shuffle(order.begin(), order.end(), random_engine);
        }

        for (std::size_t begin = 0; begin < dataset.size(); begin += batch_size) {
            const std::size_t end = std::min(begin + batch_size, dataset.size());
            std::fill(batch_gradient.begin(), batch_gradient.end(), 0.0);
            const double scale = 1.0 / static_cast<double>(end - begin);
            for (std::size_t position = begin; position < end; ++position) {
                const LabeledSample& sample = dataset[order[position]];
                accumulate_gradient(sample.features, sample.label, scale, batch_gradient, cache,
                                    delta, previous);
            }
            optimizer.compute_update(batch_gradient, config.learning_rate, update);
            subtract_update(update);
        }

        const double epoch_loss = loss(dataset);
        if (!std::isfinite(epoch_loss)) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        result.loss_per_epoch.push_back(epoch_loss);
        result.epochs = epoch;

        if (has_validation) {
            const double validation_loss = loss(validation);
            result.validation_loss_per_epoch.push_back(validation_loss);
            if (validation_loss < best_validation - config.early_stopping.min_improvement) {
                best_validation = validation_loss;
                result.best_epoch = epoch;
                epochs_without_improvement = 0;
                best_parameters = parameters();
            } else {
                ++epochs_without_improvement;
            }
            if (config.early_stopping.patience > 0 &&
                epochs_without_improvement >= config.early_stopping.patience) {
                result.stopped_early = true;
                set_parameters(best_parameters);
                return result;
            }
        }

        if (config.target_loss > 0.0 && epoch_loss <= config.target_loss) {
            result.converged = true;
            return result;
        }
    }

    if (has_validation && config.early_stopping.patience > 0 && !best_parameters.empty()) {
        set_parameters(best_parameters);
    }
    return result;
}

void ConvolutionalNetwork::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }

    const TensorShape input = input_shape();
    stream << "ml_scratch_convolutional 1\n"
           << "loss " << static_cast<int>(loss_) << '\n'
           << "input " << input.channels << ' ' << input.height << ' ' << input.width << '\n'
           << "layers " << layers_.size() << '\n';
    for (const ConvLayerSpec& layer : layers_) {
        stream << static_cast<int>(layer.kind) << ' ' << static_cast<int>(layer.activation) << ' '
               << layer.filters << ' ' << layer.kernel_size << ' ' << layer.stride << ' '
               << layer.padding << ' ' << layer.pool_size << ' ' << layer.pool_stride << ' '
               << layer.units << ' ' << (layer.residual ? 1 : 0) << '\n';
    }

    const std::vector<double> flat = parameters();
    stream << "parameters " << flat.size() << '\n' << std::setprecision(17);
    for (const double value : flat) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

ConvolutionalNetwork ConvolutionalNetwork::load(const std::string& path) {
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

    expect("ml_scratch_convolutional");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("loss");
    const std::size_t loss_value = read_size();
    if (loss_value > static_cast<std::size_t>(Loss::softmax_cross_entropy)) {
        throw std::runtime_error("unknown loss in checkpoint " + path);
    }
    expect("input");
    TensorShape input;
    input.channels = read_size();
    input.height = read_size();
    input.width = read_size();
    expect("layers");
    const std::size_t layer_count = read_size();
    if (layer_count == 0) {
        throw std::runtime_error("checkpoint declares no layers: " + path);
    }

    std::vector<ConvLayerSpec> layers;
    layers.reserve(layer_count);
    for (std::size_t index = 0; index < layer_count; ++index) {
        ConvLayerSpec layer;
        const std::size_t kind = read_size();
        const std::size_t activation = read_size();
        if (kind > static_cast<std::size_t>(ConvLayerKind::dense)) {
            throw std::runtime_error("unknown layer kind in checkpoint " + path);
        }
        if (activation > static_cast<std::size_t>(Activation::rectified_linear)) {
            throw std::runtime_error("unknown activation in checkpoint " + path);
        }
        layer.kind = static_cast<ConvLayerKind>(kind);
        layer.activation = static_cast<Activation>(activation);
        layer.filters = read_size();
        layer.kernel_size = read_size();
        layer.stride = read_size();
        layer.padding = read_size();
        layer.pool_size = read_size();
        layer.pool_stride = read_size();
        layer.units = read_size();
        layer.residual = read_size() != 0;
        layers.push_back(layer);
    }

    // The constructor re-derives every shape, so a corrupted architecture is rejected here.
    ConvolutionalNetwork network{input, std::move(layers), static_cast<Loss>(loss_value)};

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
    return network;
}

} // namespace ml_scratch
