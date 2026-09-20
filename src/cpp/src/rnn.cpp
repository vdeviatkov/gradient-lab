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

const char* recurrent_cell_name(const RecurrentCell cell) {
    switch (cell) {
    case RecurrentCell::elman:
        return "elman";
    case RecurrentCell::lstm:
        return "lstm";
    case RecurrentCell::gru:
        return "gru";
    }
    return "";
}

namespace {

double sigmoid(const double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

} // namespace

std::size_t CharRnn::gate_rows() const noexcept {
    switch (cell_) {
    case RecurrentCell::elman:
        return hidden_size_;
    case RecurrentCell::lstm:
        return 4 * hidden_size_;
    case RecurrentCell::gru:
        return 3 * hidden_size_;
    }
    return hidden_size_;
}

CharRnn::CharRnn(const std::size_t vocabulary_size, const std::size_t hidden_size,
                 const std::uint32_t seed, const RecurrentCell cell)
    : vocabulary_size_(vocabulary_size), hidden_size_(hidden_size), cell_(cell) {
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
    // Each gate's block is scaled as its own hidden_size x fan_in matrix, so a gated cell's
    // blocks start at the same scale as the elman cell's single matrix.
    fill(input_offset(), gate_rows() * vocabulary_size_, vocabulary_size_, hidden_size_);
    fill(recurrent_offset(), gate_rows() * hidden_size_, hidden_size_, hidden_size_);
    fill(output_offset(), vocabulary_size_ * hidden_size_, hidden_size_, vocabulary_size_);
    if (cell_ == RecurrentCell::lstm) {
        // A forget gate near one at the start lets the cell state persist before anything has
        // been learned about when to clear it; at zero the state would halve every step.
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            parameters_[hidden_bias_offset() + hidden_size_ + unit] = 1.0;
        }
    }
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
    if (state.size() != state_size()) {
        throw std::invalid_argument("state size does not match the cell's state size");
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

void CharRnn::step_forward(const std::size_t input, const double* previous_hidden,
                           const double* previous_cell, const std::size_t step,
                           Workspace& workspace) const {
    const double* input_weights = parameters_.data() + input_offset();
    const double* recurrent_weights = parameters_.data() + recurrent_offset();
    const double* bias = parameters_.data() + hidden_bias_offset();
    double* hidden = workspace.hidden.data() + step * hidden_size_;

    // Every gate row's pre-activation is bias + one column of W_x (the one-hot input selects
    // it, so the input term is a column read) + a row of W_h times the previous hidden vector.
    const auto pre_activation = [&](const std::size_t row) {
        double total = bias[row] + input_weights[row * vocabulary_size_ + input];
        const double* weights = recurrent_weights + row * hidden_size_;
        for (std::size_t source = 0; source < hidden_size_; ++source) {
            total += weights[source] * previous_hidden[source];
        }
        return total;
    };

    switch (cell_) {
    case RecurrentCell::elman:
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            hidden[unit] = std::tanh(pre_activation(unit));
        }
        return;
    case RecurrentCell::lstm: {
        double* gates = workspace.gates.data() + step * 4 * hidden_size_;
        double* cell = workspace.cell.data() + step * hidden_size_;
        double* cell_tanh = workspace.extra.data() + step * hidden_size_;
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            const double input_gate = sigmoid(pre_activation(unit));
            const double forget_gate = sigmoid(pre_activation(hidden_size_ + unit));
            const double output_gate = sigmoid(pre_activation(2 * hidden_size_ + unit));
            const double candidate = std::tanh(pre_activation(3 * hidden_size_ + unit));
            gates[unit] = input_gate;
            gates[hidden_size_ + unit] = forget_gate;
            gates[2 * hidden_size_ + unit] = output_gate;
            gates[3 * hidden_size_ + unit] = candidate;
            // The additive update: the old cell scaled by the forget gate plus the gated
            // candidate. Nothing here squashes c_t, which is what lets it persist.
            cell[unit] = forget_gate * previous_cell[unit] + input_gate * candidate;
            cell_tanh[unit] = std::tanh(cell[unit]);
            hidden[unit] = output_gate * cell_tanh[unit];
        }
        return;
    }
    case RecurrentCell::gru: {
        double* gates = workspace.gates.data() + step * 3 * hidden_size_;
        double* recurrent_candidate = workspace.extra.data() + step * hidden_size_;
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            gates[unit] = sigmoid(pre_activation(unit));                   // reset
            gates[hidden_size_ + unit] = sigmoid(pre_activation(hidden_size_ + unit)); // update
        }
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            // The candidate's recurrent term is gated by reset before the tanh, so the
            // recurrent product W_hn h_{t-1} is kept on its own for the backward pass.
            const std::size_t row = 2 * hidden_size_ + unit;
            const double* weights = recurrent_weights + row * hidden_size_;
            double recurrent = 0.0;
            for (std::size_t source = 0; source < hidden_size_; ++source) {
                recurrent += weights[source] * previous_hidden[source];
            }
            recurrent_candidate[unit] = recurrent;
            const double candidate =
                std::tanh(bias[row] + input_weights[row * vocabulary_size_ + input] +
                          gates[unit] * recurrent);
            gates[row] = candidate;
            const double update = gates[hidden_size_ + unit];
            hidden[unit] = (1.0 - update) * candidate + update * previous_hidden[unit];
        }
        return;
    }
    }
}

double CharRnn::forward(const TokenSequence& ids, const std::size_t offset,
                        const std::size_t steps, State& state, Workspace& workspace) const {
    const double* output_weights = parameters_.data() + output_offset();
    const double* output_bias = parameters_.data() + output_bias_offset();
    const bool has_cell = cell_ == RecurrentCell::lstm;

    workspace.hidden.resize(steps * hidden_size_);
    workspace.cell.resize(has_cell ? steps * hidden_size_ : 0);
    workspace.gates.resize(cell_ == RecurrentCell::elman ? 0 : steps * gate_rows());
    workspace.extra.resize(cell_ == RecurrentCell::elman ? 0 : steps * hidden_size_);
    workspace.probabilities.resize(steps * vocabulary_size_);
    workspace.logits.resize(vocabulary_size_);
    std::vector<double>& logits = workspace.logits;

    double total = 0.0;
    for (std::size_t step = 0; step < steps; ++step) {
        const std::size_t input = ids[offset + step];
        const std::size_t target = ids[offset + step + 1];
        double* hidden = workspace.hidden.data() + step * hidden_size_;
        const double* previous = step == 0 ? state.data() : hidden - hidden_size_;
        const double* previous_cell =
            !has_cell ? nullptr
            : step == 0 ? state.data() + hidden_size_
                        : workspace.cell.data() + (step - 1) * hidden_size_;
        step_forward(input, previous, previous_cell, step, workspace);

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
    if (has_cell) {
        state.insert(state.end(), workspace.cell.end() - static_cast<std::ptrdiff_t>(hidden_size_),
                     workspace.cell.end());
    }
    return total;
}

void CharRnn::step_backward(const std::size_t input, const double* previous_hidden,
                            const double* previous_cell, const std::size_t step,
                            std::vector<double>& flat, Workspace& workspace) const {
    const double* recurrent_weights = parameters_.data() + recurrent_offset();
    double* input_gradient = flat.data() + input_offset();
    double* recurrent_gradient = flat.data() + recurrent_offset();
    double* bias_gradient = flat.data() + hidden_bias_offset();
    const double* hidden = workspace.hidden.data() + step * hidden_size_;
    std::vector<double>& hidden_delta = workspace.hidden_delta;
    std::vector<double>& cell_delta = workspace.cell_delta;
    std::vector<double>& step_delta = workspace.step_delta;

    // Once every gate row's dL/d(pre-activation) is in step_delta, the parameter gradients and
    // the carry to h_{t-1} are the same for every cell: the rows are affine in x_t and h_{t-1}.
    // `rows` limits which gate rows carry back through W_h; the GRU's candidate row does not,
    // since its recurrent term is handled separately.
    const auto accumulate_rows = [&](const std::size_t rows) {
        for (std::size_t row = 0; row < rows; ++row) {
            const double delta = step_delta[row];
            input_gradient[row * vocabulary_size_ + input] += delta;
            bias_gradient[row] += delta;
            double* gradient_row = recurrent_gradient + row * hidden_size_;
            for (std::size_t source = 0; source < hidden_size_; ++source) {
                gradient_row[source] += delta * previous_hidden[source];
            }
        }
    };
    const auto carry_rows = [&](const std::size_t rows) {
        for (std::size_t row = 0; row < rows; ++row) {
            const double delta = step_delta[row];
            const double* weights = recurrent_weights + row * hidden_size_;
            for (std::size_t source = 0; source < hidden_size_; ++source) {
                hidden_delta[source] += delta * weights[source];
            }
        }
    };

    switch (cell_) {
    case RecurrentCell::elman:
        // Through the tanh: dL/da_t = dL/dh_t * (1 - h_t^2).
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            step_delta[unit] = hidden_delta[unit] * (1.0 - hidden[unit] * hidden[unit]);
        }
        accumulate_rows(hidden_size_);
        // Carry to the previous step: dL/dh_{t-1} = W_hh^T dL/da_t.
        std::fill(hidden_delta.begin(), hidden_delta.end(), 0.0);
        carry_rows(hidden_size_);
        return;
    case RecurrentCell::lstm: {
        const double* gates = workspace.gates.data() + step * 4 * hidden_size_;
        const double* cell_tanh = workspace.extra.data() + step * hidden_size_;
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            const double input_gate = gates[unit];
            const double forget_gate = gates[hidden_size_ + unit];
            const double output_gate = gates[2 * hidden_size_ + unit];
            const double candidate = gates[3 * hidden_size_ + unit];
            // h_t = o * tanh(c_t): the output gate's gradient, and dL/dc_t gains the path
            // through h_t on top of what the next step carried back.
            const double output_delta = hidden_delta[unit] * cell_tanh[unit];
            const double total_cell_delta =
                cell_delta[unit] +
                hidden_delta[unit] * output_gate * (1.0 - cell_tanh[unit] * cell_tanh[unit]);
            // c_t = f * c_{t-1} + i * g.
            const double input_delta = total_cell_delta * candidate;
            const double candidate_delta = total_cell_delta * input_gate;
            const double forget_delta = total_cell_delta * previous_cell[unit];
            // The carry to c_{t-1} is a plain product with the forget gate: no weight matrix
            // and no squashing derivative, which is why the gradient survives many steps.
            cell_delta[unit] = total_cell_delta * forget_gate;
            step_delta[unit] = input_delta * input_gate * (1.0 - input_gate);
            step_delta[hidden_size_ + unit] = forget_delta * forget_gate * (1.0 - forget_gate);
            step_delta[2 * hidden_size_ + unit] = output_delta * output_gate * (1.0 - output_gate);
            step_delta[3 * hidden_size_ + unit] = candidate_delta * (1.0 - candidate * candidate);
        }
        accumulate_rows(4 * hidden_size_);
        std::fill(hidden_delta.begin(), hidden_delta.end(), 0.0);
        carry_rows(4 * hidden_size_);
        return;
    }
    case RecurrentCell::gru: {
        const double* gates = workspace.gates.data() + step * 3 * hidden_size_;
        const double* recurrent_candidate = workspace.extra.data() + step * hidden_size_;
        // dL/dh_{t-1} has three sources: the direct z * h_{t-1} term, the gates' W_h rows, and
        // the candidate's reset-gated recurrent term. The first goes into a fresh buffer while
        // hidden_delta (dL/dh_t) is still being read.
        std::vector<double>& previous_delta = workspace.cell_delta;
        previous_delta.assign(hidden_size_, 0.0);
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            const double reset = gates[unit];
            const double update = gates[hidden_size_ + unit];
            const double candidate = gates[2 * hidden_size_ + unit];
            const double delta = hidden_delta[unit];
            // h_t = (1 - z) * n + z * h_{t-1}.
            const double candidate_delta = delta * (1.0 - update);
            const double update_delta = delta * (previous_hidden[unit] - candidate);
            previous_delta[unit] = delta * update;
            const double candidate_pre_delta = candidate_delta * (1.0 - candidate * candidate);
            // n = tanh(... + r * q) with q = W_hn h_{t-1}: the reset gate sees q, and q's own
            // gradient is r * dL/da_n, carried to h_{t-1} through W_hn below.
            const double reset_delta = candidate_pre_delta * recurrent_candidate[unit];
            step_delta[unit] = reset_delta * reset * (1.0 - reset);
            step_delta[hidden_size_ + unit] = update_delta * update * (1.0 - update);
            step_delta[2 * hidden_size_ + unit] = candidate_pre_delta;
        }
        // The candidate row's input weights and bias take dL/da_n as they are; its recurrent
        // weights and its carry take r * dL/da_n, since W_hn h_{t-1} enters through the reset.
        for (std::size_t unit = 0; unit < hidden_size_; ++unit) {
            const std::size_t row = 2 * hidden_size_ + unit;
            const double delta = step_delta[row];
            input_gradient[row * vocabulary_size_ + input] += delta;
            bias_gradient[row] += delta;
            const double gated = delta * gates[unit];
            double* gradient_row = recurrent_gradient + row * hidden_size_;
            const double* weights = recurrent_weights + row * hidden_size_;
            for (std::size_t source = 0; source < hidden_size_; ++source) {
                gradient_row[source] += gated * previous_hidden[source];
                previous_delta[source] += gated * weights[source];
            }
        }
        accumulate_rows(2 * hidden_size_);
        hidden_delta.swap(previous_delta);
        carry_rows(2 * hidden_size_);
        return;
    }
    }
}

void CharRnn::backward(const TokenSequence& ids, const std::size_t offset,
                       const std::size_t steps, const State& initial, const double scale,
                       const bool last_only, std::vector<double>& flat, Workspace& workspace,
                       std::vector<double>* reach) const {
    const double* output_weights = parameters_.data() + output_offset();
    double* output_gradient = flat.data() + output_offset();
    double* output_bias_gradient = flat.data() + output_bias_offset();
    const bool has_cell = cell_ == RecurrentCell::lstm;

    // hidden_delta accumulates dL/dh_t: the part arriving through this step's logits plus the
    // part carried back from step t + 1 through the recurrence. Walking the steps in reverse is
    // what "through time" means; each step's contribution to every parameter is summed, since
    // the same matrices act at every step.
    workspace.hidden_delta.assign(hidden_size_, 0.0);
    workspace.cell_delta.assign(hidden_size_, 0.0);
    workspace.step_delta.assign(gate_rows(), 0.0);
    std::vector<double>& hidden_delta = workspace.hidden_delta;
    if (reach != nullptr) {
        reach->clear();
    }

    for (std::size_t step = steps; step-- > 0;) {
        const std::size_t input = ids[offset + step];
        const std::size_t target = ids[offset + step + 1];
        const double* hidden = workspace.hidden.data() + step * hidden_size_;
        const double* previous = step == 0 ? initial.data() : hidden - hidden_size_;
        const double* previous_cell =
            !has_cell ? nullptr
            : step == 0 ? initial.data() + hidden_size_
                        : workspace.cell.data() + (step - 1) * hidden_size_;
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
            // The gradient with respect to the state as it is passed between steps: h_t, plus
            // c_t as carried in from step t + 1 for the LSTM.
            double squared = 0.0;
            for (const double value : hidden_delta) {
                squared += value * value;
            }
            if (has_cell) {
                for (const double value : workspace.cell_delta) {
                    squared += value * value;
                }
            }
            reach->push_back(std::sqrt(squared));
        }

        step_backward(input, previous, previous_cell, step, flat, workspace);
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
    stream << "ml_scratch_char_rnn 2\n"
           << "cell " << static_cast<int>(cell_) << '\n'
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
    const std::size_t version = read_size();
    if (version < 1 || version > 2) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    // Version 1 predates the gated cells and is always an elman network.
    RecurrentCell cell = RecurrentCell::elman;
    if (version >= 2) {
        expect("cell");
        const std::size_t encoded = read_size();
        if (encoded > static_cast<std::size_t>(RecurrentCell::gru)) {
            throw std::runtime_error("unknown cell in checkpoint " + path);
        }
        cell = static_cast<RecurrentCell>(encoded);
    }
    expect("vocabulary");
    const std::size_t vocabulary_size = read_size();
    expect("hidden");
    const std::size_t hidden_size = read_size();
    if (vocabulary_size < 2 || hidden_size == 0) {
        throw std::runtime_error("checkpoint declares an invalid architecture: " + path);
    }
    CharRnn network{vocabulary_size, hidden_size, 0, cell};
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
