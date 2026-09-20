#include "ml_scratch/deterministic_random.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/rnn.hpp"
#include "ml_scratch/text_corpus.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
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

double norm(const std::vector<double>& values) {
    double total = 0.0;
    for (const double value : values) {
        total += value * value;
    }
    return std::sqrt(total);
}

// A repeating pattern over a small vocabulary, long enough to train on.
ml_scratch::TokenSequence periodic(const std::size_t period, const std::size_t length) {
    ml_scratch::TokenSequence ids;
    for (std::size_t index = 0; index < length; ++index) {
        ids.push_back(index % period);
    }
    return ids;
}

void test_forward_pass_by_hand() {
    // Vocabulary of 2, one hidden unit: h_t = tanh(w_x[x_t] + w_h h_{t-1} + b_h),
    // z_t = w_y h_t + b_y over two ids. Parameters in flat order: W_xh (1x2), W_hh (1x1), b_h,
    // W_hy (2x1), b_y (2).
    ml_scratch::CharRnn network{2, 1, 0};
    network.set_parameters({0.5, -0.5, 0.8, 0.1, 1.0, -1.0, 0.2, -0.2});

    // Sequence 0, 1, 0: two predictions.
    const double h1 = std::tanh(0.5 + 0.8 * 0.0 + 0.1);
    const double h2 = std::tanh(-0.5 + 0.8 * h1 + 0.1);
    const auto step_loss = [](const double hidden, const std::size_t target) {
        const double z0 = 1.0 * hidden + 0.2;
        const double z1 = -1.0 * hidden - 0.2;
        const double log_normalizer = std::log(std::exp(z0) + std::exp(z1));
        return log_normalizer - (target == 0 ? z0 : z1);
    };
    const double expected = (step_loss(h1, 1) + step_loss(h2, 0)) / 2.0;
    require_near(network.loss({0, 1, 0}), expected, 1e-14, "the loss should match the hand computation");

    // The distribution after reading 0, 1 is softmax over the logits at h2.
    const auto probabilities = network.predict({0, 1});
    const double z0 = h2 + 0.2;
    const double z1 = -h2 - 0.2;
    require_near(probabilities[0], std::exp(z0) / (std::exp(z0) + std::exp(z1)), 1e-14,
                 "predict should return the softmax of the final logits");

    // The state advances: scoring the sequence in two halves with the state carried gives the
    // same total as scoring it at once.
    ml_scratch::CharRnn::State state = network.initial_state();
    const double first = network.loss({0, 1}, state);
    const double second = network.loss({1, 0}, state);
    require_near((first + second) / 2.0, expected, 1e-14,
                 "carrying the state should make the split scoring agree with the whole");
    require_near(state.front(), h2, 1e-14, "the state should hold the final hidden value");
}

void test_gradient_matches_finite_differences() {
    ml_scratch::CharRnn network{5, 4, 3};
    const ml_scratch::TokenSequence ids{0, 3, 1, 4, 4, 2, 0, 1, 3};

    const auto analytic = network.gradient(ids);
    const auto numerical = network.numerical_gradient(ids);
    const auto verdict = ml_scratch::check_gradient(analytic, numerical);
    require(verdict.passed, "backpropagation through time failed its gradient check");

    // From a non-zero initial state too, since training carries one across windows.
    const ml_scratch::CharRnn::State state{0.3, -0.7, 0.1, 0.9};
    const auto from_state = network.gradient(ids, state);
    require(ml_scratch::check_gradient(from_state, network.numerical_gradient(ids, state)).passed,
            "the gradient from a carried state failed its check");
    require(from_state != analytic, "the initial state should change the gradient");

    // Gradient reach: the last prediction's loss alone, differentiated with respect to the
    // hidden state at each earlier step. Lag 0 is checked against central differences over the
    // final hidden state by perturbing through the recurrence: with the loss of the last step
    // only, the gradient with respect to h_{T-1} equals W_hy^T (p - y), which is verified by
    // computing it directly from predict.
    const auto reach = network.gradient_reach(ids);
    require(reach.size() == ids.size() - 1, "one norm per step, from the last step backwards");
    const auto probabilities = network.predict(ids);
    const auto parameters = network.parameters();
    const std::size_t output_offset = 4 * 5 + 4 * 4 + 4;
    std::vector<double> direct(4, 0.0);
    for (std::size_t id = 0; id < 5; ++id) {
        const double delta = probabilities[id] - (id == ids.back() ? 1.0 : 0.0);
        for (std::size_t unit = 0; unit < 4; ++unit) {
            direct[unit] += delta * parameters[output_offset + id * 4 + unit];
        }
    }
    // predict reads all of ids, one step past what the last prediction saw; recompute for the
    // inputs the last prediction actually used.
    ml_scratch::TokenSequence inputs(ids.begin(), ids.end() - 1);
    const auto last_probabilities = network.predict(inputs);
    std::fill(direct.begin(), direct.end(), 0.0);
    for (std::size_t id = 0; id < 5; ++id) {
        const double delta = last_probabilities[id] - (id == ids.back() ? 1.0 : 0.0);
        for (std::size_t unit = 0; unit < 4; ++unit) {
            direct[unit] += delta * parameters[output_offset + id * 4 + unit];
        }
    }
    require_near(reach.front(), norm(direct), 1e-12,
                 "the lag-0 reach should be the norm of W_hy^T (p - y)");
    for (const double value : reach) {
        require(std::isfinite(value) && value >= 0.0, "reach norms are finite and non-negative");
    }
}

void test_gated_cells_by_hand() {
    // One hidden unit over a vocabulary of two, every weight chosen so the step can be followed
    // by hand. LSTM flat layout: W_x rows i, f, o, g (each 1 x 2), W_h rows i, f, o, g (1 x 1),
    // biases i, f, o, g, then W_hy (2 x 1) and b_y (2).
    ml_scratch::CharRnn lstm{2, 1, 0, ml_scratch::RecurrentCell::lstm};
    require(lstm.state_size() == 2, "an LSTM carries h and c");
    require(lstm.parameter_count() == 4 * 2 + 4 * 1 + 4 + 2 + 2, "LSTM parameter count");
    lstm.set_parameters({0.5, -0.5, 0.3, 0.1, -0.2, 0.4, 0.7, -0.3, // W_x
                         0.6, 0.2, -0.4, 0.9,                       // W_h
                         0.0, 1.0, 0.1, -0.1,                       // b
                         1.0, -1.0, 0.0, 0.0});                     // W_hy, b_y
    const auto sig = [](const double v) { return 1.0 / (1.0 + std::exp(-v)); };
    // Input id 1 from the zero state: h_{t-1} = 0, c_{t-1} = 0.
    const double i = sig(-0.5 + 0.0);
    const double f = sig(0.1 + 1.0);
    const double o = sig(0.4 + 0.1);
    const double g = std::tanh(-0.3 - 0.1);
    const double c = f * 0.0 + i * g;
    const double h = o * std::tanh(c);
    ml_scratch::CharRnn::State state = lstm.initial_state();
    static_cast<void>(lstm.loss({1, 0}, state));
    require_near(state[0], h, 1e-15, "LSTM hidden value by hand");
    require_near(state[1], c, 1e-15, "LSTM cell value by hand");
    // The forget gate's bias starts at one on a fresh cell.
    const ml_scratch::CharRnn fresh{2, 3, 4, ml_scratch::RecurrentCell::lstm};
    const auto fresh_parameters = fresh.parameters();
    const std::size_t bias_offset = 4 * 3 * 2 + 4 * 3 * 3;
    require(fresh_parameters[bias_offset + 3] == 1.0 && fresh_parameters[bias_offset + 4] == 1.0 &&
                fresh_parameters[bias_offset] == 0.0,
            "a fresh LSTM's forget biases are one and its other biases zero");

    // GRU flat layout: W_x rows r, z, n; W_h rows r, z, n; biases r, z, n; W_hy; b_y.
    ml_scratch::CharRnn gru{2, 1, 0, ml_scratch::RecurrentCell::gru};
    require(gru.state_size() == 1, "a GRU carries h alone");
    gru.set_parameters({0.5, -0.5, 0.3, 0.1, -0.2, 0.4, // W_x
                        0.6, 0.2, -0.4,                  // W_h
                        0.0, 0.1, -0.1,                  // b
                        1.0, -1.0, 0.0, 0.0});
    // Two steps, ids 1 then 0, so the second step has a non-zero previous state.
    const double r1 = sig(-0.5 + 0.0);
    const double z1 = sig(0.1 + 0.1);
    const double n1 = std::tanh(0.4 + r1 * (-0.4 * 0.0) - 0.1);
    const double h1 = (1.0 - z1) * n1 + z1 * 0.0;
    const double r2 = sig(0.5 + 0.6 * h1 + 0.0);
    const double z2 = sig(0.3 + 0.2 * h1 + 0.1);
    const double n2 = std::tanh(-0.2 + r2 * (-0.4 * h1) - 0.1);
    const double h2 = (1.0 - z2) * n2 + z2 * h1;
    state = gru.initial_state();
    static_cast<void>(gru.loss({1, 0, 1}, state));
    require_near(state[0], h2, 1e-15, "GRU hidden value by hand after two steps");
}

void test_gated_gradients_match_finite_differences() {
    const ml_scratch::TokenSequence ids{0, 3, 1, 4, 4, 2, 0, 1, 3};
    for (const auto cell : {ml_scratch::RecurrentCell::lstm, ml_scratch::RecurrentCell::gru}) {
        ml_scratch::CharRnn network{5, 4, 3, cell};
        const std::string name = ml_scratch::recurrent_cell_name(cell);
        require(ml_scratch::check_gradient(network.gradient(ids), network.numerical_gradient(ids))
                    .passed,
                name + " backpropagation through time failed its gradient check");
        ml_scratch::CharRnn::State state(network.state_size());
        for (std::size_t index = 0; index < state.size(); ++index) {
            state[index] = 0.3 * static_cast<double>(index + 1) * (index % 2 == 0 ? 1.0 : -1.0);
        }
        require(ml_scratch::check_gradient(network.gradient(ids, state),
                                           network.numerical_gradient(ids, state))
                    .passed,
                name + " gradient from a carried state failed its check");
        // Scoring in halves with the state carried must agree with scoring at once, which is
        // what makes the carried (h, c) pair a complete state.
        ml_scratch::CharRnn::State carried = network.initial_state();
        const double first = network.loss({0, 3, 1, 4, 4}, carried);
        const double second = network.loss({4, 2, 0, 1, 3}, carried);
        require_near((first * 4.0 + second * 4.0) / 8.0, network.loss(ids), 1e-14,
                     name + ": carrying the state should not change the loss");
    }
}

void test_gradient_reach_decays_for_a_contractive_recurrence() {
    // With a small recurrent matrix the Jacobian product shrinks every step, so the gradient of
    // the last prediction with respect to earlier states must fall geometrically with the lag.
    ml_scratch::CharRnn network{4, 6, 5};
    auto parameters = network.parameters();
    const std::size_t recurrent_offset = 6 * 4;
    for (std::size_t index = 0; index < 36; ++index) {
        parameters[recurrent_offset + index] *= 0.3;
    }
    network.set_parameters(parameters);
    const auto reach = network.gradient_reach(periodic(4, 31));
    require(reach.size() == 30, "thirty lags");
    require(reach[10] < 0.1 * reach[0] && reach[29] < 0.01 * reach[0],
            "the gradient should vanish with the lag under a contractive recurrence");
    for (std::size_t lag = 1; lag < reach.size(); ++lag) {
        require(reach[lag] <= reach[lag - 1] * 1.0001, "the decay should be monotone here");
    }
}

void test_gradient_clipping() {
    std::vector<double> small{0.3, -0.4}; // norm 0.5
    require_near(ml_scratch::clip_gradient(small, 1.0), 0.5, 1e-15, "the norm is returned");
    require(small == std::vector<double>{0.3, -0.4}, "a small gradient is left alone");

    std::vector<double> large{3.0, -4.0}; // norm 5
    require_near(ml_scratch::clip_gradient(large, 1.0), 5.0, 1e-15, "the pre-clip norm is returned");
    require_near(norm(large), 1.0, 1e-15, "a large gradient is scaled to the threshold");
    require_near(large[0] / large[1], -0.75, 1e-15, "clipping preserves the direction");
    require_invalid_argument([&] { static_cast<void>(ml_scratch::clip_gradient(large, 0.0)); },
                             "a zero threshold is rejected");
}

void test_training_learns_a_periodic_sequence() {
    // A period-5 pattern needs the network to remember where it is in the cycle; a bigram model
    // on the same text would already be perfect, so the RNN has to at least match it.
    const auto ids = periodic(5, 2'000);
    ml_scratch::CharRnn network{5, 12, 7};
    const double before = network.loss(ids);

    ml_scratch::RnnTrainingConfig config;
    config.learning_rate = 0.01;
    config.epochs = 8;
    config.sequence_length = 10;
    config.batch_size = 4;
    const auto result = network.fit(ids, config);
    require(result.epochs == 8 && result.loss_per_epoch.size() == 8 &&
                result.validation_loss_per_epoch.empty() &&
                result.clipped_fraction_per_epoch.size() == 8 &&
                result.max_gradient_norm_per_epoch.size() == 8 &&
                result.mean_gradient_norm_per_epoch.size() == 8,
            "the result should hold one entry per epoch");
    // 4 streams of 500 ids, 49 windows of 10 predictions each.
    require(result.updates == 8 * 49, "one update per window per epoch");
    require(result.characters_per_second > 0.0, "throughput should be measured");
    const double after = network.loss(ids);
    require(after < 0.05 && after < before / 10.0,
            "training should drive the loss on a periodic sequence near zero");

    // Greedy generation continues the cycle.
    std::mt19937 engine{1};
    const auto continuation = network.generate({0, 1, 2}, 10, 0.0, engine);
    require(continuation == ml_scratch::TokenSequence{3, 4, 0, 1, 2, 3, 4, 0, 1, 2},
            "greedy generation should continue the period-5 cycle");
    // At a low temperature sampling agrees with greedy on a pattern this sharp.
    require(network.generate({0, 1, 2}, 10, 0.2, engine) == continuation,
            "low-temperature sampling should follow the learned cycle");

    // Validation is scored after each epoch when supplied, and it is the same text here.
    ml_scratch::CharRnn other{5, 12, 7};
    const auto with_validation = other.fit(ids, periodic(5, 200), config);
    require(with_validation.validation_loss_per_epoch.size() == 8 &&
                with_validation.loss_per_epoch == result.loss_per_epoch,
            "validation should not change the training trajectory");
    require_near(with_validation.validation_loss_per_epoch.back(), other.loss(periodic(5, 200)),
                 1e-12, "the last validation loss is the loss on the validation text");
}

// Where the truncation window is the lever. The stream alternates a random bit with a copy of
// the random bit written three pairs earlier: r1 c1 r2 c2 ... with c_i = r_{i-3}. The random
// bits cannot be predicted, so the best possible loss is ln 2 / 2 = 0.347 nats: zero on the
// copies, ln 2 on the random bits. Reaching it means recalling a bit that entered six steps
// before, which nothing in the stream rewards until the gradient spans those steps. A periodic pattern
// would not do here: a carried hidden state can learn to count with a window of one, because
// each step's loss already tells the recurrence which state to be in next.
void test_truncation_limits_what_is_learned() {
    constexpr std::size_t delay = 3;
    ml_scratch::DeterministicRandom random{5};
    std::vector<std::size_t> random_bits;
    ml_scratch::TokenSequence ids;
    for (std::size_t index = 0; index < 6'000; ++index) {
        random_bits.push_back(random.bernoulli(0.5) ? 1 : 0);
        ids.push_back(random_bits.back());
        ids.push_back(index >= delay ? random_bits[index - delay] : 0);
    }
    const auto train = [&](const std::size_t window) {
        ml_scratch::CharRnn network{2, 16, 11};
        ml_scratch::RnnTrainingConfig config;
        config.learning_rate = 0.01;
        config.epochs = 8;
        config.sequence_length = window;
        config.batch_size = 8;
        static_cast<void>(network.fit(ids, config));
        return network.loss(ids);
    };
    const double floor = std::log(2.0) / 2.0;
    const double truncated = train(4);
    const double unrolled = train(16);
    require(unrolled < floor + 0.02,
            "a window longer than the dependency should reach the loss floor");
    require(truncated > std::log(2.0) - 0.03,
            "a window shorter than the dependency should learn nothing beyond chance");
}

void test_clipping_is_reported() {
    const auto ids = periodic(3, 600);
    ml_scratch::CharRnn network{3, 8, 2};
    ml_scratch::RnnTrainingConfig config;
    config.epochs = 1;
    config.sequence_length = 5;
    config.batch_size = 2;
    // A threshold below every gradient's norm clips every update.
    config.clip_norm = 1e-6;
    const auto clipped = network.fit(ids, config);
    require_near(clipped.clipped_fraction_per_epoch.front(), 1.0, 0.0,
                 "every update should have been clipped");
    require(clipped.max_gradient_norm_per_epoch.front() > 1e-6 &&
                clipped.mean_gradient_norm_per_epoch.front() <=
                    clipped.max_gradient_norm_per_epoch.front(),
            "the reported norms are the pre-clip ones");

    // With clipping off nothing is counted, but the norms are still measured.
    ml_scratch::CharRnn free_network{3, 8, 2};
    config.clip_norm = 0.0;
    const auto unclipped = free_network.fit(ids, config);
    require_near(unclipped.clipped_fraction_per_epoch.front(), 0.0, 0.0,
                 "nothing is clipped when clipping is off");
    require(unclipped.max_gradient_norm_per_epoch.front() > 0.0 &&
                unclipped.mean_gradient_norm_per_epoch.front() > 0.0,
            "the norms are still measured with clipping off");
}

void test_seed_reproducibility_and_checkpoints() {
    const auto ids = periodic(4, 400);
    ml_scratch::RnnTrainingConfig config;
    config.epochs = 2;
    config.sequence_length = 8;
    config.batch_size = 2;

    ml_scratch::CharRnn first{4, 6, 21};
    ml_scratch::CharRnn second{4, 6, 21};
    require(first.parameters() == second.parameters(), "the same seed gives the same network");
    require(ml_scratch::CharRnn(4, 6, 22).parameters() != first.parameters(),
            "a different seed gives a different network");
    const auto first_result = first.fit(ids, config);
    const auto second_result = second.fit(ids, config);
    // Everything but the throughput, which is wall-clock time.
    require(first_result.loss_per_epoch == second_result.loss_per_epoch &&
                first_result.max_gradient_norm_per_epoch ==
                    second_result.max_gradient_norm_per_epoch &&
                first_result.mean_gradient_norm_per_epoch ==
                    second_result.mean_gradient_norm_per_epoch,
            "training is deterministic");
    require(first.parameters() == second.parameters(), "trained networks agree");

    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_rnn_test.checkpoint").string();
    first.save(path);
    const auto restored = ml_scratch::CharRnn::load(path);
    require(restored.vocabulary_size() == 4 && restored.hidden_size() == 6 &&
                restored.parameters() == first.parameters(),
            "a checkpoint should round-trip exactly");
    std::mt19937 engine{3};
    std::mt19937 same{3};
    require(restored.generate({0}, 12, 1.0, engine) == first.generate({0}, 12, 1.0, same),
            "the restored network should generate the same text");
    std::filesystem::remove(path);

    const std::string broken =
        (std::filesystem::temp_directory_path() / "ml_scratch_rnn_broken.checkpoint").string();
    {
        std::ofstream stream{broken};
        stream << "ml_scratch_char_rnn 1\nvocabulary 4\nhidden 6\nparameters 3\n1 2 3\n";
    }
    bool rejected = false;
    try {
        static_cast<void>(ml_scratch::CharRnn::load(broken));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "a checkpoint with the wrong parameter count should be rejected");
    std::filesystem::remove(broken);
}

void test_input_validation() {
    require_invalid_argument([] { ml_scratch::CharRnn bad{1, 4}; }, "a one-character vocabulary is useless");
    require_invalid_argument([] { ml_scratch::CharRnn bad{4, 0}; }, "a zero hidden size is rejected");
    ml_scratch::CharRnn network{4, 3, 0};
    require_invalid_argument([&] { static_cast<void>(network.loss({0})); },
                             "a single id predicts nothing");
    require_invalid_argument([&] { static_cast<void>(network.loss({0, 9})); },
                             "ids outside the vocabulary are rejected");
    ml_scratch::CharRnn::State wrong(2, 0.0);
    require_invalid_argument([&] { static_cast<void>(network.loss({0, 1}, wrong)); },
                             "a state of the wrong size is rejected");
    std::mt19937 engine{0};
    require_invalid_argument([&] { static_cast<void>(network.generate({}, 3, 1.0, engine)); },
                             "generation needs a prompt");
    require_invalid_argument([&] { static_cast<void>(network.generate({0}, 3, -1.0, engine)); },
                             "a negative temperature is rejected");
    require_invalid_argument([&] { network.set_parameters({1.0}); },
                             "the wrong parameter count is rejected");
    ml_scratch::RnnTrainingConfig config;
    config.sequence_length = 0;
    require_invalid_argument([&] { static_cast<void>(network.fit(periodic(4, 100), config)); },
                             "a zero window is rejected");
    config = {};
    config.batch_size = 200;
    require_invalid_argument([&] { static_cast<void>(network.fit(periodic(4, 100), config)); },
                             "text too short for the streams is rejected");
}

} // namespace

int main() {
    try {
        test_forward_pass_by_hand();
        test_gradient_matches_finite_differences();
        test_gated_cells_by_hand();
        test_gated_gradients_match_finite_differences();
        test_gradient_reach_decays_for_a_contractive_recurrence();
        test_gradient_clipping();
        test_training_learns_a_periodic_sequence();
        test_truncation_limits_what_is_learned();
        test_clipping_is_reported();
        test_seed_reproducibility_and_checkpoints();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
