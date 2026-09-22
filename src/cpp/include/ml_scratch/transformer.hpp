#pragma once

#include "ml_scratch/optimizer.hpp"
#include "ml_scratch/text_corpus.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ml_scratch {

struct TransformerConfig {
    std::size_t vocabulary_size;
    // The longest sequence the model reads: attention is over at most this many positions, and
    // nothing before them can influence a prediction.
    std::size_t context_length;
    // Width of the residual stream, every embedding, and every attention projection.
    std::size_t model_size;
    std::size_t heads;
    std::size_t layers;
    // Width of each block's feed-forward hidden layer; zero means four times model_size.
    std::size_t feed_forward_size{0};
    // Adds a fixed sinusoidal encoding of each position to its token embedding. Off, the model
    // has no notion of order beyond what the causal mask leaks, which is what an ablation
    // measures.
    bool positional_encoding{true};
    // Adds a learned per-head bias to every attention score that depends only on the distance
    // between the query and the key: context_length values per head per block. Where the
    // sinusoidal encoding says where a token IS, this says how far away it is, which is what a
    // head that must look a fixed number of steps back needs.
    bool relative_position_bias{false};

    bool operator==(const TransformerConfig&) const = default;
};

struct TransformerTrainingConfig {
    OptimizerConfig optimizer{.kind = OptimizerKind::adam};
    double learning_rate{0.001};
    std::size_t epochs{1};
    // Sequences per update, each a window of context_length + 1 ids starting at a seeded random
    // offset in the training text. There is no carried state: each window is read from scratch.
    std::size_t batch_size{32};
    // Global-norm clipping threshold; zero disables it.
    double clip_norm{1.0};
    std::uint32_t seed{0};
};

struct TransformerTrainingResult {
    std::size_t epochs{0};
    std::size_t updates{0};
    std::vector<double> loss_per_epoch;
    std::vector<double> validation_loss_per_epoch;
    std::vector<double> clipped_fraction_per_epoch;
    std::vector<double> max_gradient_norm_per_epoch;
    // Characters whose loss was computed and backpropagated per second of the update loop.
    double characters_per_second{0.0};
};

// A decoder-only Transformer over characters: token embeddings plus a sinusoidal positional
// encoding, then `layers` pre-normalization blocks of causal multi-head self-attention and a
// two-layer feed-forward network with residual connections, a final layer normalization, and a
// linear layer to logits over the next character. Every gradient is derived explicitly.
//
// Parameters are exposed as one flat vector: the token embedding table (row per id), then per
// block the first normalization's gain and bias, W_q, W_k, W_v, W_o (each model_size rows by
// model_size columns, applied as x W^T), the second normalization's gain and bias, W_1
// (feed_forward_size by model_size), b_1, W_2 (model_size by feed_forward_size), b_2, and with
// relative_position_bias the distance biases (row per head, column per distance); then the final
// normalization's gain and bias, and the output projection's weights (row per id) and biases.
class CharTransformer {
  public:
    CharTransformer(TransformerConfig config, std::uint32_t seed = 0);

    // Mean cross-entropy over the predictions of ids[1..] from the ids before each, for a
    // sequence of at most context_length + 1 ids. Throws for fewer than two ids or more than
    // the context allows.
    [[nodiscard]] double loss(const TokenSequence& ids) const;
    // The same over a text of any length. Windows of context_length predictions start every
    // `stride` characters; each character is predicted exactly once, by the window in which it
    // has the most context, so with the default stride the windows abut and a character early in
    // a window sees only the few before it, while a smaller stride costs proportionally more
    // forward passes and guarantees every character at least context_length - stride of
    // context. Zero means the default.
    [[nodiscard]] double evaluate(const TokenSequence& text, std::size_t stride = 0) const;
    // The next-character distribution after reading `ids`, which must fit the context.
    [[nodiscard]] std::vector<double> predict(const TokenSequence& ids) const;

    [[nodiscard]] std::vector<double> gradient(const TokenSequence& ids) const;
    [[nodiscard]] std::vector<double> numerical_gradient(const TokenSequence& ids,
                                                         double epsilon = 1e-5) const;

    // The attention weights a forward pass over `ids` produces: [layer][head][query][key], with
    // zeros above the diagonal where the causal mask applies.
    [[nodiscard]] std::vector<std::vector<std::vector<std::vector<double>>>>
    attention_weights(const TokenSequence& ids) const;

    TransformerTrainingResult fit(const TokenSequence& train,
                                  const TransformerTrainingConfig& config = {});
    TransformerTrainingResult fit(const TokenSequence& train, const TokenSequence& validation,
                                  const TransformerTrainingConfig& config);

    // Draws `length` ids after `prompt`, each from softmax(logits / temperature) over the last
    // context_length ids read so far; temperature zero is greedy. The prompt is not returned.
    [[nodiscard]] TokenSequence generate(const TokenSequence& prompt, std::size_t length,
                                         double temperature, std::mt19937& engine) const;

    [[nodiscard]] std::vector<double> parameters() const { return parameters_; }
    void set_parameters(const std::vector<double>& values);
    void subtract_update(const std::vector<double>& update);
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameters_.size(); }
    [[nodiscard]] const TransformerConfig& config() const noexcept { return config_; }

    void save(const std::string& path) const;
    [[nodiscard]] static CharTransformer load(const std::string& path);

  private:
    // Everything one sequence's backward pass needs from its forward pass; sized for the
    // longest sequence and reused.
    struct Workspace;
    struct Offsets;

    [[nodiscard]] std::size_t feed_forward_size() const noexcept;
    [[nodiscard]] std::size_t head_size() const noexcept {
        return config_.model_size / config_.heads;
    }
    [[nodiscard]] Offsets offsets(std::size_t layer) const noexcept;
    [[nodiscard]] std::size_t final_norm_offset() const noexcept;
    [[nodiscard]] std::size_t output_offset() const noexcept;
    [[nodiscard]] std::size_t output_bias_offset() const noexcept;
    [[nodiscard]] std::size_t block_parameters() const noexcept;

    void validate_ids(const TokenSequence& ids, bool for_prediction) const;
    // Forward over ids[0..steps), leaving every intermediate in the workspace and the logits
    // of each position; returns the total cross-entropy against ids[1..steps] when `targets`
    // is true (the sequence then needs steps + 1 ids).
    double forward(const TokenSequence& ids, std::size_t steps, Workspace& workspace,
                   bool targets) const;
    // Adds scale * dL/dtheta for the sequence last forwarded into `workspace` to `flat`.
    void backward(const TokenSequence& ids, std::size_t steps, double scale,
                  std::vector<double>& flat, Workspace& workspace) const;

    TransformerConfig config_;
    std::vector<double> parameters_;
    // positional_[position * model_size + dimension], precomputed for every context position.
    std::vector<double> positional_;
};

} // namespace ml_scratch
