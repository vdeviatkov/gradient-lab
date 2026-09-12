#include "ml_scratch/neural_network.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
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
        layer_offsets_.push_back(parameter_count_);
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

void FeedForwardNetwork::forward_into(const std::vector<double>& input, ForwardCache& cache) const {
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
    cache.activations[0] = input;

    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const DenseLayer& layer = layers_[index];
        const std::vector<double>& previous = cache.activations[index];
        cache.pre_activations[index].resize(layer.output_size);
        cache.activations[index + 1].resize(layer.output_size);

        for (std::size_t output = 0; output < layer.output_size; ++output) {
            double total = parameters_[index].biases[output];
            const std::vector<double>& row = parameters_[index].weights[output];
            for (std::size_t input_index = 0; input_index < layer.input_size; ++input_index) {
                total += row[input_index] * previous[input_index];
            }
            cache.pre_activations[index][output] = total;
            cache.activations[index + 1][output] = apply_activation(total, layer.activation);
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

void FeedForwardNetwork::accumulate_gradient(const std::vector<double>& features,
                                             const std::vector<double>* targets,
                                             const std::size_t label, const double scale,
                                             std::vector<double>& flat,
                                             Workspace& workspace) const {
    forward_into(features, workspace.cache);
    const ForwardCache& cache = workspace.cache;

    if (targets != nullptr) {
        workspace.delta =
            output_delta(cache.pre_activations.back(), cache.activations.back(), *targets);
    } else {
        // The one-hot target is implied by `label`, so it is never materialized: under
        // softmax_cross_entropy the output delta is simply p - y.
        workspace.delta = softmax(cache.activations.back());
        workspace.delta[label] -= 1.0;
    }

    // delta holds dL/dz for the layer currently being visited.
    for (std::size_t index = layers_.size(); index-- > 0;) {
        const DenseLayer& layer = layers_[index];
        const std::vector<double>& inputs = cache.activations[index];
        const std::vector<double>& delta = workspace.delta;

        const std::size_t offset = layer_offsets_[index];
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
        workspace.previous_delta.assign(previous_layer.output_size, 0.0);
        for (std::size_t input = 0; input < layer.input_size; ++input) {
            double total = 0.0;
            for (std::size_t output = 0; output < layer.output_size; ++output) {
                total += parameters_[index].weights[output][input] * delta[output];
            }
            workspace.previous_delta[input] =
                total * activation_derivative(cache.pre_activations[index - 1][input],
                                              cache.activations[index][input],
                                              previous_layer.activation);
        }
        workspace.delta.swap(workspace.previous_delta);
    }
}

void FeedForwardNetwork::apply_gradient_step(const std::vector<double>& flat,
                                             const double learning_rate) {
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        const DenseLayer& layer = layers_[index];
        std::size_t cursor = layer_offsets_[index];
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            std::vector<double>& row = parameters_[index].weights[output];
            for (std::size_t input = 0; input < layer.input_size; ++input) {
                row[input] -= learning_rate * flat[cursor++];
            }
        }
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            parameters_[index].biases[output] -= learning_rate * flat[cursor++];
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
}

} // namespace

// The two fit overloads share this loop. `Dataset` is either a SupervisedDataset of explicit
// targets or a LabeledDataset of class indices; `accumulate` hides the difference.
template <typename Dataset, typename Accumulate, typename Evaluate>
static NetworkTrainingResult
run_training(const Dataset& dataset, const NetworkTrainingConfig& config,
             const std::size_t parameter_count, Accumulate accumulate, Evaluate evaluate) {
    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    const std::size_t batch_size =
        config.batch_size == 0 ? dataset.size() : std::min(config.batch_size, dataset.size());

    std::vector<double> batch_gradient(parameter_count, 0.0);
    std::vector<double> history;
    history.reserve(config.max_epochs);

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
        history.push_back(epoch_loss);
        if (config.target_loss > 0.0 && epoch_loss <= config.target_loss) {
            return {epoch, true, std::move(history)};
        }
    }
    return {config.max_epochs, false, std::move(history)};
}

NetworkTrainingResult FeedForwardNetwork::fit(const SupervisedDataset& dataset,
                                              const NetworkTrainingConfig& config) {
    validate_for_network(dataset);
    validate_training_config(config);

    Workspace workspace;
    return run_training(
        dataset, config, parameter_count_,
        [&](const std::vector<std::size_t>& order, const std::size_t begin, const std::size_t end,
            std::vector<double>& flat) {
            const double scale = 1.0 / static_cast<double>(end - begin);
            for (std::size_t position = begin; position < end; ++position) {
                const SupervisedSample& sample = dataset[order[position]];
                accumulate_gradient(sample.features, &sample.targets, 0, scale, flat, workspace);
            }
            apply_gradient_step(flat, config.learning_rate);
        },
        [&] { return loss(dataset); });
}

NetworkTrainingResult FeedForwardNetwork::fit(const LabeledDataset& dataset,
                                              const NetworkTrainingConfig& config) {
    static_cast<void>(validate_labeled(dataset));
    validate_training_config(config);

    Workspace workspace;
    return run_training(
        dataset, config, parameter_count_,
        [&](const std::vector<std::size_t>& order, const std::size_t begin, const std::size_t end,
            std::vector<double>& flat) {
            const double scale = 1.0 / static_cast<double>(end - begin);
            for (std::size_t position = begin; position < end; ++position) {
                const LabeledSample& sample = dataset[order[position]];
                accumulate_gradient(sample.features, nullptr, sample.label, scale, flat, workspace);
            }
            apply_gradient_step(flat, config.learning_rate);
        },
        [&] { return loss(dataset); });
}

void FeedForwardNetwork::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }

    stream << "ml_scratch_feedforward 1\n"
           << "loss " << static_cast<int>(loss_) << '\n'
           << "layers " << layers_.size() << '\n';
    for (const DenseLayer& layer : layers_) {
        stream << layer.input_size << ' ' << layer.output_size << ' '
               << static_cast<int>(layer.activation) << '\n';
    }

    const std::vector<double> flat = parameters();
    stream << "parameters " << flat.size() << '\n';
    // 17 significant digits round-trip an IEEE-754 double exactly.
    stream << std::setprecision(17);
    for (const double value : flat) {
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
    if (read_size() != 1) {
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
        layers.push_back({input_size, output_size, static_cast<Activation>(activation)});
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

} // namespace ml_scratch
