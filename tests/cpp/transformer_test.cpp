#include "ml_scratch/deterministic_random.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/transformer.hpp"

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

const ml_scratch::TransformerConfig small{5, 8, 8, 2, 2};

void test_configuration_and_parameter_count() {
    const ml_scratch::CharTransformer network{small, 3};
    // Embedding 5 x 8; per block 2 x 8 norm, 4 x 64 projections, 2 x 8 norm, 32 x 8 + 32 hidden,
    // 8 x 32 + 8 down; final norm 2 x 8; output 5 x 8 + 5.
    const std::size_t block = 16 + 256 + 16 + 256 + 32 + 256 + 8;
    require(network.parameter_count() == 40 + 2 * block + 16 + 45,
            "the parameter count should follow the layout");
    require(network.config().feed_forward_size == 32, "feed-forward defaults to four times the model");

    require_invalid_argument([] { ml_scratch::CharTransformer bad{{5, 8, 8, 3, 2}}; },
                             "a model size that is not a multiple of the heads is rejected");
    require_invalid_argument([] { ml_scratch::CharTransformer bad{{1, 8, 8, 2, 2}}; },
                             "a one-character vocabulary is rejected");
    require_invalid_argument([] { ml_scratch::CharTransformer bad{{5, 0, 8, 2, 2}}; },
                             "a zero context is rejected");
    require_invalid_argument([&] { static_cast<void>(network.loss({0})); },
                             "a single id predicts nothing");
    require_invalid_argument([&] { static_cast<void>(network.loss({0, 1, 2, 3, 4, 0, 1, 2, 3, 4})); },
                             "a sequence longer than the context is rejected");
    require_invalid_argument([&] { static_cast<void>(network.loss({0, 9})); },
                             "ids outside the vocabulary are rejected");
}

void test_positional_encoding_and_masking() {
    // With the causal mask, nothing at position i depends on ids after i. Change the last id:
    // every earlier position's attention is unchanged, every future key has weight zero, and
    // each row still sums to one over the past.
    const ml_scratch::CharTransformer network{small, 3};
    const ml_scratch::TokenSequence one{0, 3, 1, 4};
    const ml_scratch::TokenSequence two{0, 3, 1, 2};
    const auto weights_one = network.attention_weights(one);
    const auto weights_two = network.attention_weights(two);
    for (std::size_t layer = 0; layer < 2; ++layer) {
        for (std::size_t head = 0; head < 2; ++head) {
            for (std::size_t i = 0; i < 3; ++i) {
                for (std::size_t j = 0; j < 4; ++j) {
                    require_near(weights_one[layer][head][i][j], weights_two[layer][head][i][j],
                                 1e-15, "earlier positions must not see a later id");
                    if (j > i) {
                        require(weights_one[layer][head][i][j] == 0.0,
                                "the causal mask should zero every future key");
                    }
                }
                double total = 0.0;
                for (std::size_t j = 0; j <= i; ++j) {
                    total += weights_one[layer][head][i][j];
                }
                require_near(total, 1.0, 1e-12, "attention weights sum to one over the past");
            }
        }
    }
    require(network.predict({0, 3, 1}) == network.predict({0, 3, 1}),
            "predict is deterministic");

    // Without the positional encoding a single block is permutation-invariant over the past:
    // the final position attends to values that depend only on each past token, so the order of
    // the past cannot change its prediction. With a second block it can, because each position's
    // first-block output already depends on the prefix the mask lets it see — which is why a
    // deeper stack is not entirely order-blind even without positions.
    ml_scratch::TransformerConfig unordered = small;
    unordered.positional_encoding = false;
    unordered.layers = 1;
    const ml_scratch::CharTransformer no_positions{unordered, 3};
    const auto swapped = no_positions.predict({3, 0, 1, 2});
    const auto original = no_positions.predict({0, 3, 1, 2});
    for (std::size_t id = 0; id < 5; ++id) {
        require_near(swapped[id], original[id], 1e-12,
                     "without positions, one block cannot tell two orders of the past apart");
    }
    ml_scratch::TransformerConfig ordered_by_depth = unordered;
    ordered_by_depth.layers = 2;
    const ml_scratch::CharTransformer deeper{ordered_by_depth, 3};
    const auto deeper_swapped = deeper.predict({3, 0, 1, 2});
    const auto deeper_original = deeper.predict({0, 3, 1, 2});
    double leaked = 0.0;
    for (std::size_t id = 0; id < 5; ++id) {
        leaked += std::abs(deeper_swapped[id] - deeper_original[id]);
    }
    require(leaked > 1e-9, "a second block leaks order through the causal mask");
    const auto with_positions = network.predict({0, 3, 1, 2});
    const auto with_positions_swapped = network.predict({3, 0, 1, 2});
    double difference = 0.0;
    for (std::size_t id = 0; id < 5; ++id) {
        difference += std::abs(with_positions[id] - with_positions_swapped[id]);
    }
    require(difference > 1e-6, "with positions, order should matter");
}

void test_gradient_matches_finite_differences() {
    const ml_scratch::TokenSequence ids{0, 3, 1, 4, 4, 2, 0, 1, 3};
    ml_scratch::CharTransformer network{small, 3};
    require(ml_scratch::check_gradient(network.gradient(ids), network.numerical_gradient(ids)).passed,
            "the transformer gradient failed its check");
    // A shorter sequence than the context, one head, no positional encoding, and the relative
    // position bias instead.
    ml_scratch::TransformerConfig other{5, 8, 4, 1, 3, 6, false, true};
    ml_scratch::CharTransformer variant{other, 9};
    const ml_scratch::TokenSequence shorter{2, 2, 0, 4, 1};
    require(ml_scratch::check_gradient(variant.gradient(shorter), variant.numerical_gradient(shorter))
                .passed,
            "the variant transformer gradient failed its check");

    // The loss is the mean over positions of the softmax cross-entropy, checked against predict.
    double expected = 0.0;
    for (std::size_t position = 1; position < ids.size(); ++position) {
        const ml_scratch::TokenSequence prefix(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(position));
        expected -= std::log(network.predict(prefix)[ids[position]]) / static_cast<double>(ids.size() - 1);
    }
    require_near(network.loss(ids), expected, 1e-12,
                 "loss should be the mean cross-entropy of each prefix's prediction");
    // evaluate over a text of two windows plus a partial one agrees with per-window losses.
    ml_scratch::TokenSequence text;
    for (std::size_t index = 0; index < 21; ++index) {
        text.push_back((index * 7 + 3) % 5);
    }
    double total = 0.0;
    for (std::size_t offset = 0; offset < 20; offset += 8) {
        const std::size_t steps = std::min<std::size_t>(8, 20 - offset);
        const ml_scratch::TokenSequence window(text.begin() + static_cast<std::ptrdiff_t>(offset),
                                               text.begin() + static_cast<std::ptrdiff_t>(offset + steps + 1));
        total += network.loss(window) * static_cast<double>(steps);
    }
    require_near(network.evaluate(text), total / 20.0, 1e-12,
                 "evaluate should be the prediction-weighted mean over windows");
    // With a stride of 4 each character after the first window is predicted with at least four
    // characters of context: the same as scoring, for every position p >= 8, the window ending
    // at p, which is what the sum below does.
    double strided = 0.0;
    for (std::size_t position = 1; position <= 20; ++position) {
        const std::size_t start = position <= 8 ? 0 : ((position - 5) / 4) * 4;
        const ml_scratch::TokenSequence prefix(text.begin() + static_cast<std::ptrdiff_t>(start),
                                               text.begin() + static_cast<std::ptrdiff_t>(position));
        strided -= std::log(network.predict(prefix)[text[position]]);
    }
    require_near(network.evaluate(text, 4), strided / 20.0, 1e-12,
                 "a strided evaluation should predict each character from its widest window");
    require_invalid_argument([&] { static_cast<void>(network.evaluate(text, 9)); },
                             "a stride beyond the context is rejected");
}

void test_training_learns_a_periodic_sequence() {
    ml_scratch::TokenSequence ids;
    for (std::size_t index = 0; index < 2'000; ++index) {
        ids.push_back(index % 5);
    }
    ml_scratch::CharTransformer network{{5, 16, 16, 2, 1}, 7};
    const double before = network.evaluate(ids);
    ml_scratch::TransformerTrainingConfig config;
    config.learning_rate = 0.01;
    config.epochs = 6;
    config.batch_size = 8;
    config.seed = 1;
    const auto result = network.fit(ids, config);
    require(result.epochs == 6 && result.loss_per_epoch.size() == 6 &&
                result.validation_loss_per_epoch.empty() && result.updates == 6 * (2000 / 128),
            "the result should hold one entry per epoch and count its updates");
    require(result.characters_per_second > 0.0, "throughput should be measured");
    const double after = network.evaluate(ids);
    require(after < 0.1 && after < before / 10.0,
            "training should drive the loss on a periodic sequence near zero");
    std::mt19937 engine{1};
    const auto continuation = network.generate({0, 1, 2}, 10, 0.0, engine);
    require(continuation == ml_scratch::TokenSequence{3, 4, 0, 1, 2, 3, 4, 0, 1, 2},
            "greedy generation should continue the cycle");
    // Generation beyond the context keeps working on the last context_length ids.
    require(network.generate({0, 1, 2}, 40, 0.0, engine).back() == 2,
            "generation past the context should keep the cycle");
}

// The reason attention exists: a bit twenty steps back is one direct lookup away, not twenty
// multiplications. The stream is experiment 16's delayed copy at a delay the elman cell could
// not learn, with the copies written as 2 and 3 instead of 0 and 1 so that the current token
// says whether a copy comes next; a recurrent cell tracks that parity in its state for free,
// but a transformer would have to read it off the positional encoding, and a sinusoid has no
// period-two component to read it from.
void test_attention_learns_a_long_copy() {
    constexpr std::size_t delay = 10;
    ml_scratch::DeterministicRandom random{5};
    std::vector<std::size_t> bits;
    ml_scratch::TokenSequence ids;
    for (std::size_t index = 0; index < 4'000; ++index) {
        bits.push_back(random.bernoulli(0.5) ? 1 : 0);
        ids.push_back(bits.back());
        ids.push_back(2 + (index >= delay ? bits[index - delay] : 0));
    }
    ml_scratch::CharTransformer network{{4, 40, 16, 2, 1}, 11};
    ml_scratch::TransformerTrainingConfig config;
    config.learning_rate = 0.005;
    config.epochs = 30;
    config.batch_size = 8;
    static_cast<void>(network.fit(ids, config));
    // Scored with a stride of 20 in a context of 40, every prediction after the first window
    // has the source bit in view, so the floor is ln 2 / 2: the random bits cost ln 2 and the
    // copies nothing.
    const double floor = std::log(2.0) / 2.0;
    require(network.evaluate(ids, 20) < floor + 0.03,
            "attention should learn a twenty-step copy that recurrence could not");
    // The head that does it puts its weight on the source bit: at the last position of a window
    // ending on a copy, some head attends mostly to the position twenty steps back.
    const ml_scratch::TokenSequence window(ids.begin() + 1, ids.begin() + 41);
    const auto weights = network.attention_weights(window);
    double best = 0.0;
    for (std::size_t head = 0; head < 2; ++head) {
        best = std::max(best, weights[0][head][39][19]);
    }
    require(best > 0.5, "some head should put most of its weight on the source bit");
}

void test_seed_reproducibility_and_checkpoints() {
    ml_scratch::TokenSequence ids;
    for (std::size_t index = 0; index < 400; ++index) {
        ids.push_back(index % 4);
    }
    ml_scratch::TransformerTrainingConfig config;
    config.epochs = 2;
    config.batch_size = 4;
    ml_scratch::CharTransformer first{{4, 8, 8, 2, 1}, 21};
    ml_scratch::CharTransformer second{{4, 8, 8, 2, 1}, 21};
    require(first.parameters() == second.parameters(), "the same seed gives the same network");
    require(ml_scratch::CharTransformer({4, 8, 8, 2, 1}, 22).parameters() != first.parameters(),
            "a different seed gives a different network");
    const auto first_result = first.fit(ids, config);
    const auto second_result = second.fit(ids, config);
    require(first_result.loss_per_epoch == second_result.loss_per_epoch &&
                first.parameters() == second.parameters(),
            "training is deterministic");

    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_transformer_test.checkpoint").string();
    first.save(path);
    const auto restored = ml_scratch::CharTransformer::load(path);
    require(restored.config() == first.config() && restored.parameters() == first.parameters(),
            "a checkpoint should round-trip exactly");
    std::mt19937 engine{3};
    std::mt19937 same{3};
    require(restored.generate({0}, 12, 1.0, engine) == first.generate({0}, 12, 1.0, same),
            "the restored network should generate the same text");
    std::filesystem::remove(path);

    const std::string broken =
        (std::filesystem::temp_directory_path() / "ml_scratch_transformer_broken.checkpoint").string();
    {
        std::ofstream stream{broken};
        stream << "ml_scratch_char_transformer 1\nvocabulary 4\ncontext 8\nmodel 8\nheads 2\n"
                  "layers 1\nfeed_forward 32\npositional 1\nparameters 3\n1 2 3\n";
    }
    bool rejected = false;
    try {
        static_cast<void>(ml_scratch::CharTransformer::load(broken));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "a checkpoint with the wrong parameter count should be rejected");
    std::filesystem::remove(broken);
}

} // namespace

int main() {
    try {
        test_configuration_and_parameter_count();
        test_positional_encoding_and_masking();
        test_gradient_matches_finite_differences();
        test_training_learns_a_periodic_sequence();
        test_attention_learns_a_long_copy();
        test_seed_reproducibility_and_checkpoints();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
