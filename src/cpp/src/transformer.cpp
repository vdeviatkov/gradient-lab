#include "ml_scratch/transformer.hpp"

#include "ml_scratch/rnn.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace ml_scratch {
namespace {

constexpr double normalization_epsilon = 1e-5;

// y = x W^T for one row: out[j] = sum_k x[k] W[j * columns + k].
void project(const double* x, const double* weights, const std::size_t rows,
             const std::size_t columns, double* out) {
    for (std::size_t j = 0; j < rows; ++j) {
        const double* row = weights + j * columns;
        double total = 0.0;
        for (std::size_t k = 0; k < columns; ++k) {
            total += row[k] * x[k];
        }
        out[j] = total;
    }
}

// Given dL/dy for y = x W^T, adds dL/dW and dL/dx: dW[j][k] += dy[j] x[k], dx[k] += dy[j] W[j][k].
void project_backward(const double* x, const double* weights, const double* dy,
                      const std::size_t rows, const std::size_t columns, double* dweights,
                      double* dx) {
    for (std::size_t j = 0; j < rows; ++j) {
        const double delta = dy[j];
        const double* row = weights + j * columns;
        double* gradient_row = dweights + j * columns;
        for (std::size_t k = 0; k < columns; ++k) {
            gradient_row[k] += delta * x[k];
            dx[k] += delta * row[k];
        }
    }
}

// Layer normalization of one row: writes the standardized values and their inverse deviation,
// then y = gain * n + bias.
void normalize(const double* x, const std::size_t width, const double* gain, const double* bias,
               double* normalized, double& inverse_deviation, double* y) {
    double mean = 0.0;
    for (std::size_t k = 0; k < width; ++k) {
        mean += x[k];
    }
    mean /= static_cast<double>(width);
    double variance = 0.0;
    for (std::size_t k = 0; k < width; ++k) {
        const double centered = x[k] - mean;
        variance += centered * centered;
    }
    variance /= static_cast<double>(width);
    inverse_deviation = 1.0 / std::sqrt(variance + normalization_epsilon);
    for (std::size_t k = 0; k < width; ++k) {
        normalized[k] = (x[k] - mean) * inverse_deviation;
        y[k] = gain[k] * normalized[k] + bias[k];
    }
}

// Backward of the normalization above: accumulates the gain and bias gradients and ADDS dL/dx.
//   dL/dn = dL/dy * gain;  dL/dx = inv * (dn - mean(dn) - n * mean(dn . n)).
void normalize_backward(const double* dy, const double* normalized, const double inverse_deviation,
                        const double* gain, const std::size_t width, double* dgain, double* dbias,
                        double* dx, double* scratch) {
    double sum = 0.0;
    double sum_times_normalized = 0.0;
    for (std::size_t k = 0; k < width; ++k) {
        dgain[k] += dy[k] * normalized[k];
        dbias[k] += dy[k];
        scratch[k] = dy[k] * gain[k];
        sum += scratch[k];
        sum_times_normalized += scratch[k] * normalized[k];
    }
    const auto count = static_cast<double>(width);
    for (std::size_t k = 0; k < width; ++k) {
        dx[k] += inverse_deviation *
                 (scratch[k] - sum / count - normalized[k] * sum_times_normalized / count);
    }
}

} // namespace

// The flat-vector offsets of one block's parameters.
struct CharTransformer::Offsets {
    std::size_t norm1_gain;
    std::size_t norm1_bias;
    std::size_t query;
    std::size_t key;
    std::size_t value;
    std::size_t output;
    std::size_t norm2_gain;
    std::size_t norm2_bias;
    std::size_t hidden;
    std::size_t hidden_bias;
    std::size_t down;
    std::size_t down_bias;
    std::size_t distance_bias; // relative position bias, present only when configured
};

struct CharTransformer::Workspace {
    // Per block, [position][dimension] unless noted.
    struct Block {
        std::vector<double> residual;          // the stream entering the block
        std::vector<double> norm1;             // standardized values
        std::vector<double> norm1_inverse;     // [position]
        std::vector<double> attention_input;   // gain * norm1 + bias
        std::vector<double> query;
        std::vector<double> key;
        std::vector<double> value;
        std::vector<double> attention;         // [head][query][key]
        std::vector<double> context;           // concatenated heads
        std::vector<double> after_attention;   // residual + context W_o^T
        std::vector<double> norm2;
        std::vector<double> norm2_inverse;
        std::vector<double> feed_forward_input;
        std::vector<double> hidden_pre;        // [position][feed_forward]
        std::vector<double> hidden;            // relu of the above
    };
    std::vector<Block> blocks;
    std::vector<double> final_residual;
    std::vector<double> final_norm;
    std::vector<double> final_inverse;
    std::vector<double> final_output;
    std::vector<double> logits;        // [position][id]
    std::vector<double> probabilities; // [position][id]
    // Backward scratch.
    std::vector<double> d_residual;
    std::vector<double> d_after_attention;
    std::vector<double> d_input;
    std::vector<double> d_hidden;
    std::vector<double> d_context;
    std::vector<double> d_query;
    std::vector<double> d_key;
    std::vector<double> d_value;
    std::vector<double> d_scores;
    std::vector<double> scratch;
};

CharTransformer::CharTransformer(TransformerConfig config, const std::uint32_t seed)
    : config_(config) {
    if (config_.vocabulary_size < 2) {
        throw std::invalid_argument("the vocabulary needs at least two characters");
    }
    if (config_.context_length == 0 || config_.model_size == 0 || config_.heads == 0 ||
        config_.layers == 0) {
        throw std::invalid_argument("context, model size, heads, and layers must be positive");
    }
    if (config_.model_size % config_.heads != 0) {
        throw std::invalid_argument("model_size must be a multiple of the head count");
    }
    if (config_.feed_forward_size == 0) {
        config_.feed_forward_size = 4 * config_.model_size;
    }
    parameters_.assign(output_bias_offset() + config_.vocabulary_size, 0.0);

    std::mt19937 random_engine{seed};
    const auto uniform = [&random_engine](const double limit) {
        constexpr double range = 4294967296.0; // std::mt19937::max() + 1
        const double unit = static_cast<double>(random_engine()) / range;
        return -limit + 2.0 * limit * unit;
    };
    // Glorot scaling for every matrix; normalization gains at one, every bias at zero.
    const auto fill = [&](const std::size_t offset, const std::size_t rows,
                          const std::size_t columns) {
        const double limit = std::sqrt(6.0 / static_cast<double>(rows + columns));
        for (std::size_t index = 0; index < rows * columns; ++index) {
            parameters_[offset + index] = uniform(limit);
        }
    };
    const std::size_t D = config_.model_size;
    const std::size_t F = config_.feed_forward_size;
    fill(0, config_.vocabulary_size, D);
    for (std::size_t layer = 0; layer < config_.layers; ++layer) {
        const Offsets at = offsets(layer);
        std::fill_n(parameters_.begin() + static_cast<std::ptrdiff_t>(at.norm1_gain), D, 1.0);
        fill(at.query, D, D);
        fill(at.key, D, D);
        fill(at.value, D, D);
        fill(at.output, D, D);
        std::fill_n(parameters_.begin() + static_cast<std::ptrdiff_t>(at.norm2_gain), D, 1.0);
        fill(at.hidden, F, D);
        fill(at.down, D, F);
    }
    std::fill_n(parameters_.begin() + static_cast<std::ptrdiff_t>(final_norm_offset()), D, 1.0);
    fill(output_offset(), config_.vocabulary_size, D);

    // Sinusoidal positions: dimension pairs (2i, 2i + 1) carry sin and cos at a wavelength that
    // grows geometrically with i from 2 pi to 10000 * 2 pi, so every position gets a distinct
    // pattern and nearby positions get similar ones.
    positional_.assign(config_.context_length * D, 0.0);
    if (config_.positional_encoding) {
        for (std::size_t position = 0; position < config_.context_length; ++position) {
            for (std::size_t pair = 0; 2 * pair < D; ++pair) {
                const double rate =
                    std::pow(10000.0, -2.0 * static_cast<double>(pair) / static_cast<double>(D));
                const double angle = static_cast<double>(position) * rate;
                positional_[position * D + 2 * pair] = std::sin(angle);
                if (2 * pair + 1 < D) {
                    positional_[position * D + 2 * pair + 1] = std::cos(angle);
                }
            }
        }
    }
}

std::size_t CharTransformer::feed_forward_size() const noexcept {
    return config_.feed_forward_size;
}

std::size_t CharTransformer::block_parameters() const noexcept {
    const std::size_t D = config_.model_size;
    const std::size_t F = config_.feed_forward_size;
    const std::size_t bias =
        config_.relative_position_bias ? config_.heads * config_.context_length : 0;
    return 2 * D + 4 * D * D + 2 * D + F * D + F + D * F + D + bias;
}

CharTransformer::Offsets CharTransformer::offsets(const std::size_t layer) const noexcept {
    const std::size_t D = config_.model_size;
    const std::size_t F = config_.feed_forward_size;
    std::size_t cursor = config_.vocabulary_size * D + layer * block_parameters();
    Offsets at{};
    at.norm1_gain = cursor;
    cursor += D;
    at.norm1_bias = cursor;
    cursor += D;
    at.query = cursor;
    cursor += D * D;
    at.key = cursor;
    cursor += D * D;
    at.value = cursor;
    cursor += D * D;
    at.output = cursor;
    cursor += D * D;
    at.norm2_gain = cursor;
    cursor += D;
    at.norm2_bias = cursor;
    cursor += D;
    at.hidden = cursor;
    cursor += F * D;
    at.hidden_bias = cursor;
    cursor += F;
    at.down = cursor;
    cursor += D * F;
    at.down_bias = cursor;
    cursor += D;
    at.distance_bias = cursor;
    return at;
}

std::size_t CharTransformer::final_norm_offset() const noexcept {
    return config_.vocabulary_size * config_.model_size + config_.layers * block_parameters();
}

std::size_t CharTransformer::output_offset() const noexcept {
    return final_norm_offset() + 2 * config_.model_size;
}

std::size_t CharTransformer::output_bias_offset() const noexcept {
    return output_offset() + config_.vocabulary_size * config_.model_size;
}

void CharTransformer::validate_ids(const TokenSequence& ids, const bool for_prediction) const {
    const std::size_t minimum = for_prediction ? 1 : 2;
    const std::size_t maximum = for_prediction ? config_.context_length : config_.context_length + 1;
    if (ids.size() < minimum) {
        throw std::invalid_argument("the sequence is too short");
    }
    if (ids.size() > maximum) {
        throw std::invalid_argument("the sequence is longer than the context allows");
    }
    for (const std::size_t id : ids) {
        if (id >= config_.vocabulary_size) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
    }
}

double CharTransformer::forward(const TokenSequence& ids, const std::size_t steps,
                                Workspace& workspace, const bool targets) const {
    const std::size_t D = config_.model_size;
    const std::size_t F = config_.feed_forward_size;
    const std::size_t H = config_.heads;
    const std::size_t d = head_size();
    const std::size_t V = config_.vocabulary_size;
    const double* parameters = parameters_.data();
    const double scale = 1.0 / std::sqrt(static_cast<double>(d));

    workspace.blocks.resize(config_.layers);
    for (auto& block : workspace.blocks) {
        block.residual.resize(steps * D);
        block.norm1.resize(steps * D);
        block.norm1_inverse.resize(steps);
        block.attention_input.resize(steps * D);
        block.query.resize(steps * D);
        block.key.resize(steps * D);
        block.value.resize(steps * D);
        block.attention.assign(H * steps * steps, 0.0);
        block.context.resize(steps * D);
        block.after_attention.resize(steps * D);
        block.norm2.resize(steps * D);
        block.norm2_inverse.resize(steps);
        block.feed_forward_input.resize(steps * D);
        block.hidden_pre.resize(steps * F);
        block.hidden.resize(steps * F);
    }
    workspace.final_residual.resize(steps * D);
    workspace.final_norm.resize(steps * D);
    workspace.final_inverse.resize(steps);
    workspace.final_output.resize(steps * D);
    workspace.logits.resize(steps * V);
    workspace.probabilities.resize(steps * V);

    // Embedding plus position.
    std::vector<double>& stream = workspace.blocks.front().residual;
    for (std::size_t i = 0; i < steps; ++i) {
        const double* embedding = parameters + ids[i] * D;
        const double* position = positional_.data() + i * D;
        for (std::size_t k = 0; k < D; ++k) {
            stream[i * D + k] = embedding[k] + position[k];
        }
    }

    for (std::size_t layer = 0; layer < config_.layers; ++layer) {
        Workspace::Block& block = workspace.blocks[layer];
        const Offsets at = offsets(layer);
        // Normalize, then project every position to queries, keys, and values.
        for (std::size_t i = 0; i < steps; ++i) {
            normalize(block.residual.data() + i * D, D, parameters + at.norm1_gain,
                      parameters + at.norm1_bias, block.norm1.data() + i * D,
                      block.norm1_inverse[i], block.attention_input.data() + i * D);
            const double* input = block.attention_input.data() + i * D;
            project(input, parameters + at.query, D, D, block.query.data() + i * D);
            project(input, parameters + at.key, D, D, block.key.data() + i * D);
            project(input, parameters + at.value, D, D, block.value.data() + i * D);
        }
        // Causal attention, one head at a time: each query attends to keys at or before it.
        for (std::size_t head = 0; head < H; ++head) {
            double* attention = block.attention.data() + head * steps * steps;
            const double* distance_bias =
                config_.relative_position_bias
                    ? parameters + at.distance_bias + head * config_.context_length
                    : nullptr;
            for (std::size_t i = 0; i < steps; ++i) {
                const double* query = block.query.data() + i * D + head * d;
                double* row = attention + i * steps;
                double largest = -std::numeric_limits<double>::infinity();
                for (std::size_t j = 0; j <= i; ++j) {
                    const double* key = block.key.data() + j * D + head * d;
                    double score = 0.0;
                    for (std::size_t k = 0; k < d; ++k) {
                        score += query[k] * key[k];
                    }
                    row[j] = score * scale;
                    if (distance_bias != nullptr) {
                        row[j] += distance_bias[i - j];
                    }
                    largest = std::max(largest, row[j]);
                }
                double total = 0.0;
                for (std::size_t j = 0; j <= i; ++j) {
                    row[j] = std::exp(row[j] - largest);
                    total += row[j];
                }
                double* context = block.context.data() + i * D + head * d;
                std::fill_n(context, d, 0.0);
                for (std::size_t j = 0; j <= i; ++j) {
                    row[j] /= total;
                    const double* value = block.value.data() + j * D + head * d;
                    for (std::size_t k = 0; k < d; ++k) {
                        context[k] += row[j] * value[k];
                    }
                }
            }
        }
        // Output projection with the residual, then the feed-forward network with its own.
        std::vector<double>& next =
            layer + 1 < config_.layers ? workspace.blocks[layer + 1].residual
                                       : workspace.final_residual;
        std::vector<double> projected(D);
        for (std::size_t i = 0; i < steps; ++i) {
            project(block.context.data() + i * D, parameters + at.output, D, D, projected.data());
            for (std::size_t k = 0; k < D; ++k) {
                block.after_attention[i * D + k] = block.residual[i * D + k] + projected[k];
            }
            normalize(block.after_attention.data() + i * D, D, parameters + at.norm2_gain,
                      parameters + at.norm2_bias, block.norm2.data() + i * D,
                      block.norm2_inverse[i], block.feed_forward_input.data() + i * D);
            double* pre = block.hidden_pre.data() + i * F;
            double* hidden = block.hidden.data() + i * F;
            project(block.feed_forward_input.data() + i * D, parameters + at.hidden, F, D, pre);
            for (std::size_t k = 0; k < F; ++k) {
                pre[k] += parameters[at.hidden_bias + k];
                hidden[k] = std::max(pre[k], 0.0);
            }
            project(hidden, parameters + at.down, D, F, projected.data());
            for (std::size_t k = 0; k < D; ++k) {
                next[i * D + k] =
                    block.after_attention[i * D + k] + projected[k] + parameters[at.down_bias + k];
            }
        }
    }

    double total = 0.0;
    for (std::size_t i = 0; i < steps; ++i) {
        normalize(workspace.final_residual.data() + i * D, D, parameters + final_norm_offset(),
                  parameters + final_norm_offset() + D, workspace.final_norm.data() + i * D,
                  workspace.final_inverse[i], workspace.final_output.data() + i * D);
        double* logits = workspace.logits.data() + i * V;
        project(workspace.final_output.data() + i * D, parameters + output_offset(), V, D, logits);
        for (std::size_t id = 0; id < V; ++id) {
            logits[id] += parameters[output_bias_offset() + id];
        }
        const double largest = *std::max_element(logits, logits + V);
        double sum_of_exponentials = 0.0;
        for (std::size_t id = 0; id < V; ++id) {
            sum_of_exponentials += std::exp(logits[id] - largest);
        }
        const double log_normalizer = largest + std::log(sum_of_exponentials);
        double* probabilities = workspace.probabilities.data() + i * V;
        for (std::size_t id = 0; id < V; ++id) {
            probabilities[id] = std::exp(logits[id] - log_normalizer);
        }
        if (targets) {
            total += log_normalizer - logits[ids[i + 1]];
        }
    }
    return total;
}

void CharTransformer::backward(const TokenSequence& ids, const std::size_t steps,
                               const double scale, std::vector<double>& flat,
                               Workspace& workspace) const {
    const std::size_t D = config_.model_size;
    const std::size_t F = config_.feed_forward_size;
    const std::size_t H = config_.heads;
    const std::size_t d = head_size();
    const std::size_t V = config_.vocabulary_size;
    const double* parameters = parameters_.data();
    double* gradient = flat.data();
    const double score_scale = 1.0 / std::sqrt(static_cast<double>(d));

    workspace.d_residual.assign(steps * D, 0.0);
    workspace.d_after_attention.assign(steps * D, 0.0);
    workspace.d_input.assign(steps * D, 0.0);
    workspace.d_hidden.assign(steps * F, 0.0);
    workspace.d_context.assign(steps * D, 0.0);
    workspace.d_query.assign(steps * D, 0.0);
    workspace.d_key.assign(steps * D, 0.0);
    workspace.d_value.assign(steps * D, 0.0);
    workspace.d_scores.assign(steps, 0.0);
    workspace.scratch.assign(std::max(D, F), 0.0);
    std::vector<double>& d_residual = workspace.d_residual;
    std::vector<double> d_logits(V);
    std::vector<double> d_output(D);

    // Output layer and final normalization: dL/dz = p - onehot(target) per position.
    for (std::size_t i = 0; i < steps; ++i) {
        const double* probabilities = workspace.probabilities.data() + i * V;
        for (std::size_t id = 0; id < V; ++id) {
            d_logits[id] = scale * (probabilities[id] - (id == ids[i + 1] ? 1.0 : 0.0));
            gradient[output_bias_offset() + id] += d_logits[id];
        }
        std::fill(d_output.begin(), d_output.end(), 0.0);
        project_backward(workspace.final_output.data() + i * D, parameters + output_offset(),
                         d_logits.data(), V, D, gradient + output_offset(), d_output.data());
        normalize_backward(d_output.data(), workspace.final_norm.data() + i * D,
                           workspace.final_inverse[i], parameters + final_norm_offset(), D,
                           gradient + final_norm_offset(), gradient + final_norm_offset() + D,
                           d_residual.data() + i * D, workspace.scratch.data());
    }

    for (std::size_t layer = config_.layers; layer-- > 0;) {
        Workspace::Block& block = workspace.blocks[layer];
        const Offsets at = offsets(layer);
        std::vector<double>& d_after = workspace.d_after_attention;
        std::vector<double>& d_input = workspace.d_input;
        std::vector<double>& d_hidden = workspace.d_hidden;

        // Feed-forward block: the residual passes d_residual straight through to d_after, and
        // the branch adds its own contribution via the second normalization.
        std::copy(d_residual.begin(), d_residual.end(), d_after.begin());
        std::fill(d_input.begin(), d_input.end(), 0.0);
        for (std::size_t i = 0; i < steps; ++i) {
            const double* d_out = d_residual.data() + i * D;
            for (std::size_t k = 0; k < D; ++k) {
                gradient[at.down_bias + k] += d_out[k];
            }
            double* d_hidden_row = d_hidden.data() + i * F;
            std::fill_n(d_hidden_row, F, 0.0);
            project_backward(block.hidden.data() + i * F, parameters + at.down, d_out, D, F,
                             gradient + at.down, d_hidden_row);
            const double* pre = block.hidden_pre.data() + i * F;
            for (std::size_t k = 0; k < F; ++k) {
                d_hidden_row[k] = pre[k] > 0.0 ? d_hidden_row[k] : 0.0;
                gradient[at.hidden_bias + k] += d_hidden_row[k];
            }
            double* d_ffn_input = d_input.data() + i * D;
            project_backward(block.feed_forward_input.data() + i * D, parameters + at.hidden,
                             d_hidden_row, F, D, gradient + at.hidden, d_ffn_input);
            normalize_backward(d_ffn_input, block.norm2.data() + i * D, block.norm2_inverse[i],
                               parameters + at.norm2_gain, D, gradient + at.norm2_gain,
                               gradient + at.norm2_bias, d_after.data() + i * D,
                               workspace.scratch.data());
        }

        // Attention block. d_after is dL/d(after_attention); the residual passes it through to
        // this block's input, and the branch adds its contribution via the first normalization.
        std::copy(d_after.begin(), d_after.end(), d_residual.begin());
        std::vector<double>& d_context = workspace.d_context;
        std::vector<double>& d_query = workspace.d_query;
        std::vector<double>& d_key = workspace.d_key;
        std::vector<double>& d_value = workspace.d_value;
        std::fill(d_context.begin(), d_context.end(), 0.0);
        std::fill(d_query.begin(), d_query.end(), 0.0);
        std::fill(d_key.begin(), d_key.end(), 0.0);
        std::fill(d_value.begin(), d_value.end(), 0.0);
        for (std::size_t i = 0; i < steps; ++i) {
            project_backward(block.context.data() + i * D, parameters + at.output,
                             d_after.data() + i * D, D, D, gradient + at.output,
                             d_context.data() + i * D);
        }
        for (std::size_t head = 0; head < H; ++head) {
            const double* attention = block.attention.data() + head * steps * steps;
            double* d_distance_bias =
                config_.relative_position_bias
                    ? gradient + at.distance_bias + head * config_.context_length
                    : nullptr;
            for (std::size_t i = 0; i < steps; ++i) {
                const double* row = attention + i * steps;
                const double* d_context_row = d_context.data() + i * D + head * d;
                // dL/d(attention weight) = d_context . value, and values gather their share.
                std::vector<double>& d_weights = workspace.d_scores;
                double dot_total = 0.0;
                for (std::size_t j = 0; j <= i; ++j) {
                    const double* value = block.value.data() + j * D + head * d;
                    double* d_value_row = d_value.data() + j * D + head * d;
                    double dot = 0.0;
                    for (std::size_t k = 0; k < d; ++k) {
                        dot += d_context_row[k] * value[k];
                        d_value_row[k] += row[j] * d_context_row[k];
                    }
                    d_weights[j] = dot;
                    dot_total += row[j] * dot;
                }
                // Softmax backward, then the scores' dependence on queries and keys.
                const double* query = block.query.data() + i * D + head * d;
                double* d_query_row = d_query.data() + i * D + head * d;
                for (std::size_t j = 0; j <= i; ++j) {
                    // Softmax backward gives dL/d(score before scaling); the distance bias
                    // takes it as is, the dot product takes it scaled.
                    const double d_logit = row[j] * (d_weights[j] - dot_total);
                    if (d_distance_bias != nullptr) {
                        d_distance_bias[i - j] += d_logit;
                    }
                    const double d_score = d_logit * score_scale;
                    const double* key = block.key.data() + j * D + head * d;
                    double* d_key_row = d_key.data() + j * D + head * d;
                    for (std::size_t k = 0; k < d; ++k) {
                        d_query_row[k] += d_score * key[k];
                        d_key_row[k] += d_score * query[k];
                    }
                }
            }
        }
        std::fill(d_input.begin(), d_input.end(), 0.0);
        for (std::size_t i = 0; i < steps; ++i) {
            const double* input = block.attention_input.data() + i * D;
            double* d_attention_input = d_input.data() + i * D;
            project_backward(input, parameters + at.query, d_query.data() + i * D, D, D,
                             gradient + at.query, d_attention_input);
            project_backward(input, parameters + at.key, d_key.data() + i * D, D, D,
                             gradient + at.key, d_attention_input);
            project_backward(input, parameters + at.value, d_value.data() + i * D, D, D,
                             gradient + at.value, d_attention_input);
            normalize_backward(d_attention_input, block.norm1.data() + i * D,
                               block.norm1_inverse[i], parameters + at.norm1_gain, D,
                               gradient + at.norm1_gain, gradient + at.norm1_bias,
                               d_residual.data() + i * D, workspace.scratch.data());
        }
    }

    // The embedding table: each position's gradient lands on the row of its id.
    for (std::size_t i = 0; i < steps; ++i) {
        double* row = gradient + ids[i] * D;
        const double* d_stream = d_residual.data() + i * D;
        for (std::size_t k = 0; k < D; ++k) {
            row[k] += d_stream[k];
        }
    }
}

double CharTransformer::loss(const TokenSequence& ids) const {
    validate_ids(ids, false);
    Workspace workspace;
    const std::size_t steps = ids.size() - 1;
    return forward(ids, steps, workspace, true) / static_cast<double>(steps);
}

double CharTransformer::evaluate(const TokenSequence& text, std::size_t stride) const {
    if (text.size() < 2) {
        throw std::invalid_argument("evaluation needs at least one prediction");
    }
    const std::size_t T = config_.context_length;
    if (stride == 0) {
        stride = T;
    }
    if (stride > T) {
        throw std::invalid_argument("the stride cannot exceed the context length");
    }
    Workspace workspace;
    double total = 0.0;
    const std::size_t predictions = text.size() - 1;
    const std::size_t V = config_.vocabulary_size;
    // The first window counts every prediction; each later window counts only its last `stride`
    // predictions, the ones no earlier window predicted with as much context.
    std::size_t counted_from = 0;
    for (std::size_t offset = 0; counted_from < predictions; offset += stride) {
        const std::size_t steps = std::min(T, predictions - offset);
        const TokenSequence window(text.begin() + static_cast<std::ptrdiff_t>(offset),
                                   text.begin() + static_cast<std::ptrdiff_t>(offset + steps + 1));
        validate_ids(window, false);
        forward(window, steps, workspace, true);
        for (std::size_t i = counted_from - offset; i < steps; ++i) {
            total -= std::log(workspace.probabilities[i * V + window[i + 1]]);
        }
        counted_from = offset + steps;
    }
    return total / static_cast<double>(predictions);
}

std::vector<double> CharTransformer::predict(const TokenSequence& ids) const {
    validate_ids(ids, true);
    Workspace workspace;
    forward(ids, ids.size(), workspace, false);
    const std::size_t V = config_.vocabulary_size;
    return {workspace.probabilities.end() - static_cast<std::ptrdiff_t>(V),
            workspace.probabilities.end()};
}

std::vector<double> CharTransformer::gradient(const TokenSequence& ids) const {
    validate_ids(ids, false);
    Workspace workspace;
    const std::size_t steps = ids.size() - 1;
    forward(ids, steps, workspace, true);
    std::vector<double> flat(parameters_.size(), 0.0);
    backward(ids, steps, 1.0 / static_cast<double>(steps), flat, workspace);
    return flat;
}

std::vector<double> CharTransformer::numerical_gradient(const TokenSequence& ids,
                                                        const double epsilon) const {
    validate_ids(ids, false);
    CharTransformer probe = *this;
    std::vector<double> numerical(parameters_.size(), 0.0);
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        const double original = parameters_[index];
        probe.parameters_[index] = original + epsilon;
        const double above = probe.loss(ids);
        probe.parameters_[index] = original - epsilon;
        const double below = probe.loss(ids);
        probe.parameters_[index] = original;
        numerical[index] = (above - below) / (2.0 * epsilon);
    }
    return numerical;
}

std::vector<std::vector<std::vector<std::vector<double>>>>
CharTransformer::attention_weights(const TokenSequence& ids) const {
    validate_ids(ids, true);
    Workspace workspace;
    const std::size_t steps = ids.size();
    forward(ids, steps, workspace, false);
    std::vector<std::vector<std::vector<std::vector<double>>>> weights(config_.layers);
    for (std::size_t layer = 0; layer < config_.layers; ++layer) {
        weights[layer].resize(config_.heads);
        for (std::size_t head = 0; head < config_.heads; ++head) {
            weights[layer][head].assign(steps, std::vector<double>(steps, 0.0));
            const double* attention =
                workspace.blocks[layer].attention.data() + head * steps * steps;
            for (std::size_t i = 0; i < steps; ++i) {
                for (std::size_t j = 0; j <= i; ++j) {
                    weights[layer][head][i][j] = attention[i * steps + j];
                }
            }
        }
    }
    return weights;
}

namespace {

void validate_training_config(const TransformerTrainingConfig& config) {
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.epochs == 0 || config.batch_size == 0) {
        throw std::invalid_argument("epochs and batch_size must be positive");
    }
    if (!std::isfinite(config.clip_norm) || config.clip_norm < 0.0) {
        throw std::invalid_argument("clip_norm must be finite and non-negative");
    }
}

} // namespace

TransformerTrainingResult CharTransformer::fit(const TokenSequence& train,
                                               const TransformerTrainingConfig& config) {
    return fit(train, TokenSequence{}, config);
}

TransformerTrainingResult CharTransformer::fit(const TokenSequence& train,
                                               const TokenSequence& validation,
                                               const TransformerTrainingConfig& config) {
    validate_training_config(config);
    const std::size_t T = config_.context_length;
    if (train.size() < T + 2) {
        throw std::invalid_argument("the training text is shorter than one window");
    }
    for (const std::size_t id : train) {
        if (id >= config_.vocabulary_size) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
    }
    const bool has_validation = !validation.empty();

    // An epoch is as many updates as it takes to read the text's worth of characters once,
    // drawn as random windows rather than in order: the model has no state to carry, so
    // consecutive windows would only correlate the updates.
    const std::size_t updates_per_epoch =
        std::max<std::size_t>(1, train.size() / (config.batch_size * T));
    const std::size_t offsets_available = train.size() - T - 1;
    std::mt19937 random_engine{config.seed};

    Optimizer optimizer{config.optimizer, parameters_.size()};
    std::vector<double> flat(parameters_.size(), 0.0);
    std::vector<double> update;
    Workspace workspace;
    TokenSequence window(T + 1);
    const double scale = 1.0 / static_cast<double>(config.batch_size * T);

    TransformerTrainingResult result;
    double training_seconds = 0.0;
    std::size_t characters = 0;
    for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        double loss_total = 0.0;
        double norm_max = 0.0;
        std::size_t clipped = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t step = 0; step < updates_per_epoch; ++step) {
            std::fill(flat.begin(), flat.end(), 0.0);
            double batch_loss = 0.0;
            for (std::size_t sequence = 0; sequence < config.batch_size; ++sequence) {
                const std::size_t offset =
                    static_cast<std::size_t>(random_engine() % (offsets_available + 1));
                std::copy_n(train.begin() + static_cast<std::ptrdiff_t>(offset), T + 1,
                            window.begin());
                batch_loss += forward(window, T, workspace, true);
                backward(window, T, scale, flat, workspace);
            }
            const double norm = clip_gradient(
                flat, config.clip_norm > 0.0 ? config.clip_norm
                                             : std::numeric_limits<double>::infinity());
            if (!std::isfinite(norm)) {
                throw std::runtime_error("training diverged to a non-finite gradient");
            }
            norm_max = std::max(norm_max, norm);
            if (config.clip_norm > 0.0 && norm > config.clip_norm) {
                ++clipped;
            }
            optimizer.compute_update(flat, config.learning_rate, update);
            subtract_update(update);
            loss_total += batch_loss * scale;
            characters += config.batch_size * T;
        }
        const auto finish = std::chrono::steady_clock::now();
        training_seconds += std::chrono::duration<double>(finish - start).count();

        result.epochs = epoch;
        result.updates += updates_per_epoch;
        result.loss_per_epoch.push_back(loss_total / static_cast<double>(updates_per_epoch));
        result.clipped_fraction_per_epoch.push_back(static_cast<double>(clipped) /
                                                    static_cast<double>(updates_per_epoch));
        result.max_gradient_norm_per_epoch.push_back(norm_max);
        if (!std::isfinite(result.loss_per_epoch.back())) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        if (has_validation) {
            result.validation_loss_per_epoch.push_back(evaluate(validation));
        }
    }
    result.characters_per_second =
        training_seconds > 0.0 ? static_cast<double>(characters) / training_seconds : 0.0;
    return result;
}

TokenSequence CharTransformer::generate(const TokenSequence& prompt, const std::size_t length,
                                        const double temperature, std::mt19937& engine) const {
    if (prompt.empty()) {
        throw std::invalid_argument("generation needs a prompt of at least one id");
    }
    if (!std::isfinite(temperature) || temperature < 0.0) {
        throw std::invalid_argument("temperature must be finite and non-negative");
    }
    for (const std::size_t id : prompt) {
        if (id >= config_.vocabulary_size) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
    }
    const std::size_t V = config_.vocabulary_size;
    TokenSequence history = prompt;
    TokenSequence output;
    output.reserve(length);
    Workspace workspace;
    std::vector<double> probabilities(V);
    constexpr double range = 4294967296.0; // std::mt19937::max() + 1
    for (std::size_t index = 0; index < length; ++index) {
        // Only the last context_length ids are visible: the model has no memory beyond them.
        const std::size_t visible = std::min(history.size(), config_.context_length);
        const TokenSequence window(history.end() - static_cast<std::ptrdiff_t>(visible),
                                   history.end());
        forward(window, visible, workspace, false);
        const double* logits = workspace.logits.data() + (visible - 1) * V;
        std::size_t next = 0;
        if (temperature == 0.0) {
            next = static_cast<std::size_t>(std::max_element(logits, logits + V) - logits);
        } else {
            const double largest = *std::max_element(logits, logits + V);
            double total = 0.0;
            for (std::size_t id = 0; id < V; ++id) {
                probabilities[id] = std::exp((logits[id] - largest) / temperature);
                total += probabilities[id];
            }
            const double draw = static_cast<double>(engine()) / range;
            double cumulative = 0.0;
            next = V - 1;
            for (std::size_t id = 0; id < V; ++id) {
                cumulative += probabilities[id] / total;
                if (draw < cumulative) {
                    next = id;
                    break;
                }
            }
        }
        output.push_back(next);
        history.push_back(next);
    }
    return output;
}

void CharTransformer::set_parameters(const std::vector<double>& values) {
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

void CharTransformer::subtract_update(const std::vector<double>& update) {
    if (update.size() != parameters_.size()) {
        throw std::invalid_argument("update size does not match the parameter count");
    }
    for (std::size_t index = 0; index < parameters_.size(); ++index) {
        parameters_[index] -= update[index];
    }
}

void CharTransformer::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    stream << "ml_scratch_char_transformer 1\n"
           << "vocabulary " << config_.vocabulary_size << '\n'
           << "context " << config_.context_length << '\n'
           << "model " << config_.model_size << '\n'
           << "heads " << config_.heads << '\n'
           << "layers " << config_.layers << '\n'
           << "feed_forward " << config_.feed_forward_size << '\n'
           << "positional " << (config_.positional_encoding ? 1 : 0) << '\n'
           << "relative_bias " << (config_.relative_position_bias ? 1 : 0) << '\n'
           << "parameters " << parameters_.size() << '\n'
           << std::setprecision(17);
    for (const double value : parameters_) {
        stream << value << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

CharTransformer CharTransformer::load(const std::string& path) {
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
    expect("ml_scratch_char_transformer");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    TransformerConfig config{};
    expect("vocabulary");
    config.vocabulary_size = read_size();
    expect("context");
    config.context_length = read_size();
    expect("model");
    config.model_size = read_size();
    expect("heads");
    config.heads = read_size();
    expect("layers");
    config.layers = read_size();
    expect("feed_forward");
    config.feed_forward_size = read_size();
    expect("positional");
    config.positional_encoding = read_size() != 0;
    expect("relative_bias");
    config.relative_position_bias = read_size() != 0;
    // The constructor re-validates the architecture.
    CharTransformer network{config};
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
