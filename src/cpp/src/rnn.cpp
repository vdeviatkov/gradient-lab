#include "ml_scratch/rnn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace ml_scratch {

double clip_gradient(std::vector<double>& gradient, const double max_norm) {
    // An infinite threshold is allowed: it never clips but still measures the norm.
    if (std::isnan(max_norm) || max_norm <= 0.0) {
        throw std::invalid_argument("the clipping threshold must be positive");
    }
    double squared = 0.0;
    for (const double value : gradient) {
        squared += value * value;
    }
    const double norm = std::sqrt(squared);
    if (norm > max_norm) {
        const double factor = max_norm / norm;
        for (double& value : gradient) {
            value *= factor;
        }
    }
    return norm;
}

CharRnn::CharRnn(const std::size_t vocabulary_size, const std::size_t hidden_size,
                 const std::uint32_t seed)
    : vocabulary_size_(vocabulary_size), hidden_size_(hidden_size) {
    if (vocabulary_size < 2) {
        throw std::invalid_argument("the vocabulary needs at least two characters");
    }
    if (hidden_size == 0) {
        throw std::invalid_argument("hidden_size must be positive");
    }
    parameters_.assign(output_bias_offset() + vocabulary_size_, 0.0);

    std::mt19937 random_engine{seed};
    const auto uniform = [&random_engine](const double limit) {
        constexpr double range = 4294967296.0; // std::mt19937::max() + 1
        const double unit = static_cast<double>(random_engine()) / range;
        return -limit + 2.0 * limit * unit;
    };
    // Glorot scaling for each matrix, with the biases at zero. For the recurrent matrix the fan-in
    // and fan-out are both the hidden size, which puts its entries' variance at 1/hidden_size and
    // its largest singular value near one: the boundary between a state that fades and one that
    // blows up as it is multiplied through the steps.
    const auto fill = [&](const std::size_t offset, const std::size_t count,
                          const std::size_t fan_in, const std::size_t fan_out) {
        const double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
        for (std::size_t index = 0; index < count; ++index) {
            parameters_[offset + index] = uniform(limit);
        }
    };
    fill(input_offset(), hidden_size_ * vocabulary_size_, vocabulary_size_, hidden_size_);
    fill(recurrent_offset(), hidden_size_ * hidden_size_, hidden_size_, hidden_size_);
    fill(output_offset(), vocabulary_size_ * hidden_size_, hidden_size_, vocabulary_size_);
}

void CharRnn::validate_ids(const TokenSequence& ids) const {
    if (ids.size() < 2) {
        throw std::invalid_argument("a sequence needs at least two ids to predict anything");
    }
    for (const std::size_t id : ids) {
        if (id >= vocabulary_size_) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
    }
}

void CharRnn::validate_state(const State& state) const {
    if (state.size() != hidden_size_) {
        throw std::invalid_argument("state size does not match the hidden size");
    }
    for (const double value : state) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("state must be finite");
        }
    }
}

void CharRnn::softmax_into(const std::vector<double>& logits, const double temperature,
                           std::vector<double>& probabilities) const {
    probabilities.resize(vocabulary_size_);
    const double largest = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (std::size_t id = 0; id < vocabulary_size_; ++id) {
        probabilities[id] = std::exp((logits[id] - largest) / temperature);
        total += probabilities[id];
    }
    for (double& probability : probabilities) {
        probability /= total;
    }
}

double CharRnn::forward(const TokenSequence& ids, const std::size_t offset,
                        const std::size_t steps, State& state, Workspace& workspace) const {
    const double* input_weights = parameters_.data() + input_offset();
    const double* recurrent_weights = parameters_.data() + recurrent_offset();
    const double* hidden_bias = parameters_.data() + hidden_bias_offset();
    const double* output_weights = parameters_.data() + output_offset();
    const double* output_bias = parameters_.data() + output_bias_offset();

    workspace.hidden.resize(steps * hidden_size_);
    workspace.probabilities.resize(steps * vocabulary_size_);
    workspace.logits.resize(vocabulary_size_);
    std::vector<double>& logits = workspace.logits;

    double total = 0.0;
    for (std::size_t step = 0; step < steps; ++step) {
        const std::size_t input = ids[offset + step];
        const std::size_t target = ids[offset + step + 1];
        double* hidden = workspace.hidden.data() + step * hidden_size_;
        const double* previous = step == 0 ? state.data() : hidden - hidden_size_;

        // The one-hot input selects one column of W_xh, so the input term is a column read.
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            double activation = hidden_bias[unit] + input_weights[unit * vocabulary_size_ + input];
            const double* row = recurrent_weights + unit * hidden_size_;
            for (std::size_t source = 0; source < hidden_size_; ++source) {
                activation += row[source] * previous[source];
            }
            hidden[unit] = std::tanh(activation);
        }
        for (std::size_t id = 0; id < vocabulary_size_; ++id) {
            double logit = output_bias[id];
            const double* row = output_weights + id * hidden_size_;
            for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
                logit += row[unit] * hidden[unit];
            }
            logits[id] = logit;
        }
        // Softmax cross-entropy in the log-sum-exp form, so a large logit cannot overflow.
        const double largest = *std::max_element(logits.begin(), logits.end());
        double sum_of_exponentials = 0.0;
        for (const double logit : logits) {
            sum_of_exponentials += std::exp(logit - largest);
        }
        const double log_normalizer = largest + std::log(sum_of_exponentials);
        total += log_normalizer - logits[target];
        double* probabilities = workspace.probabilities.data() + step * vocabulary_size_;
        for (std::size_t id = 0; id < vocabulary_size_; ++id) {
            probabilities[id] = std::exp(logits[id] - log_normalizer);
        }
    }
    state.assign(workspace.hidden.end() - static_cast<std::ptrdiff_t>(hidden_size_),
                 workspace.hidden.end());
    return total;
}

void CharRnn::backward(const TokenSequence& ids, const std::size_t offset,
                       const std::size_t steps, const State& initial, const double scale,
                       const bool last_only, std::vector<double>& flat, Workspace& workspace,
                       std::vector<double>* reach) const {
    const double* recurrent_weights = parameters_.data() + recurrent_offset();
    const double* output_weights = parameters_.data() + output_offset();
    double* input_gradient = flat.data() + input_offset();
    double* recurrent_gradient = flat.data() + recurrent_offset();
    double* hidden_bias_gradient = flat.data() + hidden_bias_offset();
    double* output_gradient = flat.data() + output_offset();
    double* output_bias_gradient = flat.data() + output_bias_offset();

    // hidden_delta accumulates dL/dh_t: the part arriving through this step's logits plus the
    // part carried back from step t + 1 through the recurrence. Walking the steps in reverse is
    // what "through time" means; each step's contribution to every parameter is summed, since
    // the same matrices act at every step.
    workspace.hidden_delta.assign(hidden_size_, 0.0);
    workspace.step_delta.assign(hidden_size_, 0.0);
    std::vector<double>& hidden_delta = workspace.hidden_delta;
    std::vector<double>& step_delta = workspace.step_delta;
    if (reach != nullptr) {
        reach->clear();
    }

    for (std::size_t step = steps; step-- > 0;) {
        const std::size_t input = ids[offset + step];
        const std::size_t target = ids[offset + step + 1];
        const double* hidden = workspace.hidden.data() + step * hidden_size_;
        const double* previous = step == 0 ? initial.data() : hidden - hidden_size_;
        const double* probabilities = workspace.probabilities.data() + step * vocabulary_size_;

        // dL/dz_t = p_t - onehot(target), unless this step's loss is excluded.
        if (!last_only || step + 1 == steps) {
            for (std::size_t id = 0; id < vocabulary_size_; ++id) {
                const double logit_delta = scale * (probabilities[id] - (id == target ? 1.0 : 0.0));
                double* row = output_gradient + id * hidden_size_;
                const double* weights = output_weights + id * hidden_size_;
                for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
                    row[unit] += logit_delta * hidden[unit];
                    hidden_delta[unit] += logit_delta * weights[unit];
                }
                output_bias_gradient[id] += logit_delta;
            }
        }
        if (reach != nullptr) {
            double squared = 0.0;
            for (const double value : hidden_delta) {
                squared += value * value;
            }
            reach->push_back(std::sqrt(squared));
        }

        // Through the tanh: dL/da_t = dL/dh_t * (1 - h_t^2).
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            step_delta[unit] = hidden_delta[unit] * (1.0 - hidden[unit] * hidden[unit]);
        }
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            const double delta = step_delta[unit];
            input_gradient[unit * vocabulary_size_ + input] += delta;
            hidden_bias_gradient[unit] += delta;
            double* row = recurrent_gradient + unit * hidden_size_;
            for (std::size_t source = 0; source < hidden_size_; ++source) {
                row[source] += delta * previous[source];
            }
        }
        // Carry to the previous step: dL/dh_{t-1} = W_hh^T dL/da_t.
        std::fill(hidden_delta.begin(), hidden_delta.end(), 0.0);
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            const double delta = step_delta[unit];
            const double* row = recurrent_weights + unit * hidden_size_;
            for (std::size_t source = 0; source < hidden_size_; ++source) {
                hidden_delta[source] += delta * row[source];
            }
        }
    }
}

double CharRnn::loss(const TokenSequence& ids, State& state) const {
    validate_ids(ids);
    validate_state(state);
    // Scored in windows so the per-step buffers stay small whatever the sequence's length; the
    // state carries across windows, so the result does not depend on the window size.
    constexpr std::size_t window = 256;
    Workspace workspace;
    double total = 0.0;
    const std::size_t predictions = ids.size() - 1;
    for (std::size_t offset = 0; offset < predictions; offset += window) {
        const std::size_t steps = std::min(window, predictions - offset);
        total += forward(ids, offset, steps, state, workspace);
    }
    return total / static_cast<double>(predictions);
}

double CharRnn::loss(const TokenSequence& ids) const {
    State state = initial_state();
    return loss(ids, state);
}

std::vector<double> CharRnn::predict(const TokenSequence& ids) const {
    if (ids.empty()) {
        throw std::invalid_argument("predict needs at least one id");
    }
    // Feeding the sequence through forward needs a target for every input; a dummy target of 0
    // appended costs one extra prediction whose loss is discarded.
    TokenSequence padded = ids;
    padded.push_back(0);
    validate_ids(padded);
    State state = initial_state();
    Workspace workspace;
    forward(padded, 0, ids.size(), state, workspace);
    return {workspace.probabilities.end() - static_cast<std::ptrdiff_t>(vocabulary_size_),
            workspace.probabilities.end()};
}

std::vector<double> CharRnn::gradient(const TokenSequence& ids, const State& state) const {
    validate_ids(ids);
    State initial = state.empty() ? initial_state() : state;
    validate_state(initial);
    const std::size_t steps = ids.size() - 1;
    Workspace workspace;
    State running = initial;
    forward(ids, 0, steps, running, workspace);
    std::vector<double> flat(parameters_.size(), 0.0);
    backward(ids, 0, steps, initial, 1.0 / static_cast<double>(steps), false, flat, workspace);
    return flat;
}

std::vector<double> CharRnn::numerical_gradient(const TokenSequence& ids, const State& state,
                                                const double epsilon) const {
    validate_ids(ids);
    const State initial = state.empty() ? initial_state() : state;
    validate_state(initial);
    CharRnn probe = *this;
    std::vector<double> numerical(parameters_.size(), 0.0);
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        const double original = parameters_[index];
        probe.parameters_[index] = original + epsilon;
        State above_state = initial;
        const double above = probe.loss(ids, above_state);
        probe.parameters_[index] = original - epsilon;
        State below_state = initial;
        const double below = probe.loss(ids, below_state);
        probe.parameters_[index] = original;
        numerical[index] = (above - below) / (2.0 * epsilon);
    }
    return numerical;
}

std::vector<double> CharRnn::gradient_reach(const TokenSequence& ids) const {
    validate_ids(ids);
    const std::size_t steps = ids.size() - 1;
    const State initial = initial_state();
    Workspace workspace;
    State running = initial;
    forward(ids, 0, steps, running, workspace);
    std::vector<double> flat(parameters_.size(), 0.0);
    std::vector<double> reach;
    backward(ids, 0, steps, initial, 1.0, true, flat, workspace, &reach);
    return reach;
}

namespace {

void validate_training_config(const RnnTrainingConfig& config) {
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.epochs == 0) {
        throw std::invalid_argument("epochs must be positive");
    }
    if (config.sequence_length == 0) {
        throw std::invalid_argument("sequence_length must be positive");
    }
    if (config.batch_size == 0) {
        throw std::invalid_argument("batch_size must be positive");
    }
    if (!std::isfinite(config.clip_norm) || config.clip_norm < 0.0) {
        throw std::invalid_argument("clip_norm must be finite and non-negative");
    }
}

} // namespace

RnnTrainingResult CharRnn::fit(const TokenSequence& train, const RnnTrainingConfig& config) {
    return fit(train, TokenSequence{}, config);
}

RnnTrainingResult CharRnn::fit(const TokenSequence& train, const TokenSequence& validation,
                               const RnnTrainingConfig& config) {
    validate_training_config(config);
    validate_ids(train);
    const bool has_validation = !validation.empty();
    if (has_validation) {
        validate_ids(validation);
    }

    // The text is cut into batch_size contiguous streams of equal length; each stream keeps its
    // own hidden state, which is carried from one window to the next without gradient. A window
    // of sequence_length predictions needs sequence_length + 1 ids.
    const std::size_t stream_length = train.size() / config.batch_size;
    const std::size_t steps = config.sequence_length;
    if (stream_length < steps + 1) {
        throw std::invalid_argument(
            "the training text is too short for this batch size and sequence length");
    }
    const std::size_t windows = (stream_length - 1) / steps;

    Optimizer optimizer{config.optimizer, parameters_.size()};
    std::vector<double> flat(parameters_.size(), 0.0);
    std::vector<double> update;
    Workspace workspace;
    std::vector<State> states(config.batch_size, initial_state());

    RnnTrainingResult result;
    const double scale = 1.0 / static_cast<double>(config.batch_size * steps);
    double training_seconds = 0.0;
    std::size_t characters = 0;

    for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        for (State& state : states) {
            std::fill(state.begin(), state.end(), 0.0);
        }
        double loss_total = 0.0;
        double norm_total = 0.0;
        double norm_max = 0.0;
        std::size_t clipped = 0;
        const auto start = std::chrono::steady_clock::now();

        for (std::size_t window = 0; window < windows; ++window) {
            std::fill(flat.begin(), flat.end(), 0.0);
            double window_loss = 0.0;
            for (std::size_t stream = 0; stream < config.batch_size; ++stream) {
                const std::size_t offset = stream * stream_length + window * steps;
                const State initial = states[stream];
                window_loss += forward(train, offset, steps, states[stream], workspace);
                backward(train, offset, steps, initial, scale, false, flat, workspace);
            }
            const double norm = clip_gradient(
                flat, config.clip_norm > 0.0 ? config.clip_norm
                                             : std::numeric_limits<double>::infinity());
            if (!std::isfinite(norm)) {
                throw std::runtime_error("training diverged to a non-finite gradient");
            }
            norm_total += norm;
            norm_max = std::max(norm_max, norm);
            if (config.clip_norm > 0.0 && norm > config.clip_norm) {
                ++clipped;
            }
            optimizer.compute_update(flat, config.learning_rate, update);
            subtract_update(update);
            loss_total += window_loss * scale;
            characters += config.batch_size * steps;
        }
        const auto finish = std::chrono::steady_clock::now();
        training_seconds += std::chrono::duration<double>(finish - start).count();

        const auto window_count = static_cast<double>(windows);
        result.epochs = epoch;
        result.updates += windows;
        result.loss_per_epoch.push_back(loss_total / window_count);
        result.clipped_fraction_per_epoch.push_back(static_cast<double>(clipped) / window_count);
        result.max_gradient_norm_per_epoch.push_back(norm_max);
        result.mean_gradient_norm_per_epoch.push_back(norm_total / window_count);
        if (!std::isfinite(result.loss_per_epoch.back())) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        if (has_validation) {
            result.validation_loss_per_epoch.push_back(loss(validation));
        }
    }
    result.characters_per_second =
        training_seconds > 0.0 ? static_cast<double>(characters) / training_seconds : 0.0;
    return result;
}

TokenSequence CharRnn::generate(const TokenSequence& prompt, const std::size_t length,
                                const double temperature, std::mt19937& engine) const {
    if (prompt.empty()) {
        throw std::invalid_argument("generation needs a prompt of at least one id");
    }
    if (!std::isfinite(temperature) || temperature < 0.0) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
    for (const std::size_t id : prompt) {
        if (id >= vocabulary_size_) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
    }

    // The prompt is read one id at a time through the same forward step, with a throwaway
    // target; only the state and the final distribution matter.
    State state = initial_state();
    Workspace workspace;
    TokenSequence pair(2, 0);
    std::vector<double> probabilities;
    const auto step = [&](const std::size_t input) {
        pair[0] = input;
        forward(pair, 0, 1, state, workspace);
    };
    for (const std::size_t id : prompt) {
        step(id);
    }

    TokenSequence output;
    output.reserve(length);
    constexpr double range = 4294967296.0; // std::mt19937::max() + 1
    for (std::size_t index = 0; index < length; ++index) {
        std::size_t next = 0;
        if (temperature == 0.0) {
            const auto begin = workspace.probabilities.begin();
            next = static_cast<std::size_t>(std::max_element(begin, begin + vocabulary_size_) -
                                            begin);
        } else {
            // Re-softmax the logits at the requested temperature, then invert the cumulative
            // distribution with one uniform draw.
            softmax_into(workspace.logits, temperature, probabilities);
            const double draw = static_cast<double>(engine()) / range;
            double cumulative = 0.0;
            next = vocabulary_size_ - 1;
            for (std::size_t id = 0; id < vocabulary_size_; ++id) {
                cumulative += probabilities[id];
                if (draw < cumulative) {
                    next = id;
                    break;
                }
            }
        }
        output.push_back(next);
        step(next);
    }
    return output;
}

std::vector<double> CharRnn::parameters() const { return parameters_; }

void CharRnn::set_parameters(const std::vector<double>& values) {
    if (values.size() != parameters_.size()) {
        throw std::invalid_argument("parameter count does not match the network");
    }
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("parameters must be finite");
        }
    }
    parameters_ = values;
}

void CharRnn::subtract_update(const std::vector<double>& update) {
    if (update.size() != parameters_.size()) {
        throw std::invalid_argument("update size does not match the parameter count");
    }
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        parameters_[index] -= update[index];
    }
}

void CharRnn::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    stream << "ml_scratch_char_rnn 1\n"
           << "vocabulary " << vocabulary_size_ << '\n'
           << "hidden " << hidden_size_ << '\n'
           << "parameters " << parameters_.size() << '\n'
           << std::setprecision(17);
    for (const double value : parameters_) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

CharRnn CharRnn::load(const std::string& path) {
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

    expect("ml_scratch_char_rnn");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("vocabulary");
    const std::size_t vocabulary_size = read_size();
    expect("hidden");
    const std::size_t hidden_size = read_size();
    if (vocabulary_size < 2 || hidden_size == 0) {
        throw std::runtime_error("checkpoint declares an invalid architecture: " + path);
    }
    CharRnn network{vocabulary_size, hidden_size};
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
