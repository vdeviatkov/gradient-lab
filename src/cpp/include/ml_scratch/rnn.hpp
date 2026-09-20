#pragma once

#include "ml_scratch/optimizer.hpp"
#include "ml_scratch/text_corpus.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ml_scratch {

struct RnnTrainingConfig {
    OptimizerConfig optimizer{.kind = OptimizerKind::adam};
    double learning_rate{0.002};
    std::size_t epochs{1};
    // How many steps each backward pass unrolls: the truncation length of backpropagation through
    // time. The hidden state is carried across consecutive windows but no gradient flows between
    // them, so nothing further back than this can be learned from directly.
    std::size_t sequence_length{50};
    // The training text is cut into this many equal contiguous streams that advance in parallel;
    // one update averages the loss over one window from each of them.
    std::size_t batch_size{32};
    // If the whole gradient's Euclidean norm exceeds this it is scaled down to it. Zero disables
    // clipping. The gradient's norm before any clipping is what the result reports.
    double clip_norm{5.0};
    // There is no seed: the streams are fixed slices of the text and nothing is shuffled, so
    // training is a deterministic function of the text and the network's initial parameters.
};

struct RnnTrainingResult {
    std::size_t epochs{0};
    // Mean cross-entropy per predicted character, in nats, averaged over the epoch's updates.
    std::vector<double> loss_per_epoch;
    // Cross-entropy over the validation sequence after each epoch; empty without one.
    std::vector<double> validation_loss_per_epoch;
    // Of the epoch's updates, the fraction whose gradient norm exceeded clip_norm, and the largest
    // and mean norm before clipping. Recorded whether or not clipping is enabled, so an unclipped
    // run shows what clipping would have done.
    std::vector<double> clipped_fraction_per_epoch;
    std::vector<double> max_gradient_norm_per_epoch;
    std::vector<double> mean_gradient_norm_per_epoch;
    // Training throughput: characters whose loss was computed and backpropagated, per second of
    // wall-clock time in the update loop. Validation passes are excluded.
    double characters_per_second{0.0};
    std::size_t updates{0};
};

// The recurrence that turns the previous state and the input into the next state. Every cell
// reads a one-hot character x_t and produces a hidden vector h_t of hidden_size values that the
// output layer reads; they differ in how h_t is computed and in what else is carried.
enum class RecurrentCell {
    // h_t = tanh(W_x x_t + W_h h_{t-1} + b). One matrix multiply per step and a state that is
    // rewritten in full every step, which is what makes its gradient a product of Jacobians.
    elman,
    // Long short-term memory. A cell state c_t that is updated additively,
    //   c_t = f_t * c_{t-1} + i_t * g_t,   h_t = o_t * tanh(c_t),
    // with sigmoid input, forget and output gates i, f, o and a tanh candidate g, each an affine
    // function of x_t and h_{t-1}. The state carried between steps is (h_t, c_t). The forget
    // gate's bias starts at one, so a fresh cell remembers by default.
    lstm,
    // Gated recurrent unit. A reset gate r and an update gate z, both sigmoid, and a candidate
    //   n_t = tanh(W_xn x_t + r_t * (W_hn h_{t-1}) + b_n),   h_t = (1 - z_t) * n_t + z_t * h_{t-1}.
    // The update gate interpolates between keeping the old state and taking the candidate, which
    // is the additive path again, with one vector of state instead of two.
    gru,
};

[[nodiscard]] const char* recurrent_cell_name(RecurrentCell cell);

// A character-level recurrent network: a recurrent cell over one-hot inputs, and a linear layer
// from its hidden vector to logits over the next character,
//   z_t = W_hy h_t + b_y,
// trained by softmax cross-entropy. Gradients are derived explicitly by backpropagation through
// time for every cell.
//
// Parameters are exposed as one flat vector in the order W_x (one block of hidden_size rows per
// gate, a column per vocabulary id), W_h (the same blocks, a column per previous hidden unit),
// b (one entry per gate row), W_hy (row per vocabulary id), b_y. The elman cell has one gate
// block, the LSTM four in the order i, f, o, g, and the GRU three in the order r, z, n.
class CharRnn {
  public:
    // The state carried between calls: h for the elman and GRU cells, h followed by c for the
    // LSTM; state_size() values.
    using State = std::vector<double>;

    CharRnn(std::size_t vocabulary_size, std::size_t hidden_size, std::uint32_t seed = 0,
            RecurrentCell cell = RecurrentCell::elman);

    [[nodiscard]] State initial_state() const { return State(state_size(), 0.0); }
    [[nodiscard]] std::size_t state_size() const noexcept {
        return cell_ == RecurrentCell::lstm ? 2 * hidden_size_ : hidden_size_;
    }
    [[nodiscard]] RecurrentCell cell() const noexcept { return cell_; }

    // Mean cross-entropy per predicted character over ids[1..], each predicted from everything
    // before it, starting from `state` and advancing it to the end. Any length: the forward pass
    // keeps nothing per step, so a whole split can be scored in one call. Throws when fewer than
    // two ids are given.
    [[nodiscard]] double loss(const TokenSequence& ids, State& state) const;
    [[nodiscard]] double loss(const TokenSequence& ids) const;
    // The next-character distribution after reading all of `ids` from the zero state.
    [[nodiscard]] std::vector<double> predict(const TokenSequence& ids) const;

    // Gradient of loss(ids, state) with respect to every parameter, by full backpropagation
    // through the whole sequence — the truncation window is whatever the caller passes in.
    [[nodiscard]] std::vector<double> gradient(const TokenSequence& ids,
                                               const State& state = {}) const;
    // The same by central differences, to verify `gradient`.
    [[nodiscard]] std::vector<double> numerical_gradient(const TokenSequence& ids,
                                                         const State& state = {},
                                                         double epsilon = 1e-5) const;

    // How far back a single prediction's gradient reaches. Takes the loss of the LAST prediction
    // alone and returns, for lag = 0, 1, ..., the Euclidean norm of its gradient with respect to
    // the state that many steps earlier — the whole carried state, so h and c together for the
    // LSTM. For the elman cell this is a product of `lag` Jacobians, so it shrinks or grows
    // geometrically; how fast, and whether a gated cell escapes that, is the measurement.
    [[nodiscard]] std::vector<double> gradient_reach(const TokenSequence& ids) const;

    RnnTrainingResult fit(const TokenSequence& train, const RnnTrainingConfig& config = {});
    RnnTrainingResult fit(const TokenSequence& train, const TokenSequence& validation,
                          const RnnTrainingConfig& config);

    // Reads `prompt` from the zero state, then draws `length` further ids one at a time, each
    // from softmax(logits / temperature) and fed back as the next input. A temperature of zero
    // takes the most likely id at every step. The returned sequence excludes the prompt.
    [[nodiscard]] TokenSequence generate(const TokenSequence& prompt, std::size_t length,
                                         double temperature, std::mt19937& engine) const;

    [[nodiscard]] std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);
    void subtract_update(const std::vector<double>& update);
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameters_.size(); }
    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return vocabulary_size_; }
    [[nodiscard]] std::size_t hidden_size() const noexcept { return hidden_size_; }

    // Text checkpoints with every parameter at 17 significant digits.
    void save(const std::string& path) const;
    [[nodiscard]] static CharRnn load(const std::string& path);

  private:
    // Everything one backward pass needs from the forward pass, sized for one window and reused.
    struct Workspace {
        std::vector<double> hidden;        // [step][unit], h_t
        std::vector<double> cell;          // [step][unit], c_t; LSTM only
        std::vector<double> gates;         // [step][gate row], the activated gate values
        std::vector<double> extra;         // [step][unit], tanh(c_t) for the LSTM, W_hn h_{t-1} for the GRU
        std::vector<double> probabilities; // [step][id], softmax(z_t)
        std::vector<double> hidden_delta;  // dL/dh_t for the step being visited
        std::vector<double> cell_delta;    // dL/dc_t carried from the step after; LSTM only
        std::vector<double> step_delta;    // dL/d(pre-activation) of every gate row
        std::vector<double> logits;
    };

    // Runs `steps` steps from `state` over inputs ids[offset + t], filling the workspace, and
    // returns the total (not mean) cross-entropy against ids[offset + t + 1]. `state` becomes the
    // final hidden state.
    double forward(const TokenSequence& ids, std::size_t offset, std::size_t steps, State& state,
                   Workspace& workspace) const;
    // Adds scale * dL/dtheta for the window last forwarded into `workspace` to `flat`. With
    // `last_only`, only the final step's loss contributes. `initial` is the state the window
    // started from. `reach`, when given, receives ||dL/dh_t|| from the last step backwards.
    void backward(const TokenSequence& ids, std::size_t offset, std::size_t steps,
                  const State& initial, double scale, bool last_only, std::vector<double>& flat,
                  Workspace& workspace, std::vector<double>* reach = nullptr) const;
    // One cell step forward from (previous_hidden, previous_cell) with input id `input`, writing
    // the step's caches; and its backward, which reads `hidden_delta` (dL/dh_t) and `cell_delta`
    // (dL/dc_t carried in), accumulates the parameter gradient, and leaves dL/dh_{t-1} and
    // dL/dc_{t-1} in the same two buffers.
    void step_forward(std::size_t input, const double* previous_hidden,
                      const double* previous_cell, std::size_t step, Workspace& workspace) const;
    void step_backward(std::size_t input, const double* previous_hidden,
                       const double* previous_cell, std::size_t step, std::vector<double>& flat,
                       Workspace& workspace) const;
    void validate_ids(const TokenSequence& ids) const;
    void validate_state(const State& state) const;
    void softmax_into(const std::vector<double>& logits, double temperature,
                      std::vector<double>& probabilities) const;

    // Gate rows per cell: 1, 4, or 3 blocks of hidden_size.
    [[nodiscard]] std::size_t gate_rows() const noexcept;
    // Offsets of each block inside the flat parameter vector.
    [[nodiscard]] std::size_t input_offset() const noexcept { return 0; }
    [[nodiscard]] std::size_t recurrent_offset() const noexcept {
        return gate_rows() * vocabulary_size_;
    }
    [[nodiscard]] std::size_t hidden_bias_offset() const noexcept {
        return recurrent_offset() + gate_rows() * hidden_size_;
    }
    [[nodiscard]] std::size_t output_offset() const noexcept {
        return hidden_bias_offset() + gate_rows();
    }
    [[nodiscard]] std::size_t output_bias_offset() const noexcept {
        return output_offset() + vocabulary_size_ * hidden_size_;
    }

    std::size_t vocabulary_size_;
    std::size_t hidden_size_;
    RecurrentCell cell_;
    std::vector<double> parameters_;
};

// Scales `gradient` down to Euclidean norm `max_norm` if it is larger, and returns the norm it
// had before. A gradient at or below the threshold is left exactly as it was; an infinite
// threshold only measures.
double clip_gradient(std::vector<double>& gradient, double max_norm);

} // namespace ml_scratch
