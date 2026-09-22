#include "ml_scratch/deterministic_random.hpp"
#include "ml_scratch/rnn.hpp"
#include "ml_scratch/text_corpus.hpp"
#include "ml_scratch/transformer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260921;
// The prefix, splits and vocabulary shared with experiments 15 and 16.
constexpr std::size_t corpus_characters = 500'000;

// The main model, and the reference numbers from experiments 15 and 16 it is read against.
constexpr std::size_t context_length = 64;
constexpr std::size_t model_size = 64;
constexpr std::size_t heads = 4;
constexpr std::size_t layers = 2;
constexpr double learning_rate = 0.003;
constexpr double clip_norm = 1.0;
constexpr std::size_t batch_size = 32;
constexpr std::size_t epochs = 8;
// Validation losses of the 128-unit elman cell and the 70-unit GRU after each epoch on this
// corpus, from experiments 15 and 16, and their test losses after twelve.
constexpr double elman_validation[] = {2.3113, 2.2093, 2.1487, 2.0969, 2.0590, 2.0278,
                                       2.0065, 1.9859, 1.9681, 1.9541, 1.9435, 1.9324};
constexpr double gru_validation[] = {2.3221, 2.1630, 2.0755, 2.0235, 1.9886, 1.9618,
                                     1.9402, 1.9227, 1.9087, 1.8974, 1.8874, 1.8789};
constexpr double elman_test = 1.9637;
constexpr double gru_test = 1.9192;
constexpr double trigram_test = 2.1536;

// Part two: the same architecture at a parameter count matching the recurrent networks, and
// its ablations, each for fewer epochs.
constexpr std::size_t small_model_size = 32;
constexpr std::size_t small_epochs = 6;
constexpr std::size_t short_context = 16;

// Part three: experiment 16's delayed-copy stream with the copies written as 2 and 3, so the
// current token says whether a copy comes next.
constexpr std::size_t copy_pairs = 6'000;
constexpr std::size_t copy_delay = 10;
constexpr std::size_t copy_context = 40;
constexpr std::size_t copy_model_size = 16;
constexpr std::size_t copy_epochs = 30;
constexpr std::size_t rnn_copy_hidden = 32;
constexpr std::size_t rnn_copy_epochs = 60;

constexpr std::size_t sample_length = 300;

std::string data_path(const char* variable, const std::string& fallback) {
    if (const char* override_path = std::getenv(variable)) {
        return override_path;
    }
    return fallback;
}

struct Run {
    std::string label;
    std::size_t parameters;
    std::vector<double> train_loss;
    std::vector<double> validation_loss;
    double test_loss;
    double characters_per_second;
    double seconds;
};

void print_epoch_header() {
    std::cout << "  epoch   train loss   validation   bits/char   clipped   max |grad|   chars/s   "
                 "seconds\n";
}

Run train(const std::string& label, ml_scratch::CharTransformer& network,
          const std::size_t count, const ml_scratch::TextCorpus& corpus, const double rate) {
    ml_scratch::TransformerTrainingConfig config;
    config.learning_rate = rate;
    config.epochs = 1;
    config.batch_size = batch_size;
    config.clip_norm = clip_norm;
    std::cout << "\n  " << label << ": " << network.parameter_count() << " parameters\n";
    print_epoch_header();
    Run run{label, network.parameter_count(), {}, {}, 0.0, 0.0, 0.0};
    for (std::size_t epoch = 1; epoch <= count; ++epoch) {
        // A fresh seed per epoch draws fresh windows; the sequence of seeds is fixed.
        config.seed = seed + static_cast<std::uint32_t>(epoch);
        const auto start = std::chrono::steady_clock::now();
        const auto result = network.fit(corpus.train, corpus.validation, config);
        const auto finish = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(finish - start).count();
        run.seconds += seconds;
        run.characters_per_second += result.characters_per_second / static_cast<double>(count);
        run.train_loss.push_back(result.loss_per_epoch.front());
        run.validation_loss.push_back(result.validation_loss_per_epoch.front());
        std::cout << "  " << std::setw(5) << epoch << std::setw(13) << run.train_loss.back()
                  << std::setw(13) << run.validation_loss.back() << std::setw(12)
                  << ml_scratch::bits_per_character(run.validation_loss.back()) << std::setw(10)
                  << result.clipped_fraction_per_epoch.front() << std::setw(13)
                  << std::setprecision(2) << result.max_gradient_norm_per_epoch.front()
                  << std::setw(10) << std::setprecision(0) << result.characters_per_second
                  << std::setw(10) << std::setprecision(1) << seconds << std::setprecision(4)
                  << '\n';
    }
    run.test_loss = network.evaluate(corpus.test);
    return run;
}

void print_summary_header() {
    std::cout << "  model                       parameters   validation       test   bits/char   "
                 "perplexity   chars/s   seconds\n";
}

void print_summary(const Run& run) {
    std::string padded = run.label;
    padded.resize(26, ' ');
    std::cout << "  " << padded << std::setw(13) << run.parameters << std::setw(13)
              << run.validation_loss.back() << std::setw(11) << run.test_loss << std::setw(12)
              << ml_scratch::bits_per_character(run.test_loss) << std::setw(13)
              << ml_scratch::perplexity(run.test_loss) << std::setw(10) << std::setprecision(0)
              << run.characters_per_second << std::setw(10) << std::setprecision(1) << run.seconds
              << std::setprecision(4) << '\n';
}

ml_scratch::TransformerConfig configuration(const std::size_t width, const std::size_t context,
                                            const std::size_t head_count,
                                            const std::size_t vocabulary) {
    return {vocabulary, context, width, head_count, layers};
}

// Where each head looks: the mean attention weight it places at each distance behind the
// query, over the last position of many validation windows.
struct HeadProfile {
    double mean_distance;
    double within_three; // weight mass at distances 0..3
    std::size_t favourite; // distance with the most weight
};

std::vector<std::vector<HeadProfile>> attention_profiles(const ml_scratch::CharTransformer& network,
                                                         const ml_scratch::TokenSequence& text) {
    const auto& config = network.config();
    const std::size_t T = config.context_length;
    constexpr std::size_t windows = 50;
    std::vector<std::vector<std::vector<double>>> by_distance(
        config.layers, std::vector<std::vector<double>>(config.heads, std::vector<double>(T, 0.0)));
    const std::size_t stride = (text.size() - T) / windows;
    for (std::size_t window = 0; window < windows; ++window) {
        const auto begin = text.begin() + static_cast<std::ptrdiff_t>(window * stride);
        const ml_scratch::TokenSequence ids(begin, begin + static_cast<std::ptrdiff_t>(T));
        const auto weights = network.attention_weights(ids);
        for (std::size_t layer = 0; layer < config.layers; ++layer) {
            for (std::size_t head = 0; head < config.heads; ++head) {
                for (std::size_t j = 0; j < T; ++j) {
                    by_distance[layer][head][T - 1 - j] +=
                        weights[layer][head][T - 1][j] / static_cast<double>(windows);
                }
            }
        }
    }
    std::vector<std::vector<HeadProfile>> profiles(config.layers);
    for (std::size_t layer = 0; layer < config.layers; ++layer) {
        for (std::size_t head = 0; head < config.heads; ++head) {
            const auto& weights = by_distance[layer][head];
            HeadProfile profile{0.0, 0.0, 0};
            for (std::size_t distance = 0; distance < T; ++distance) {
                profile.mean_distance += static_cast<double>(distance) * weights[distance];
                if (distance <= 3) {
                    profile.within_three += weights[distance];
                }
                if (weights[distance] > weights[profile.favourite]) {
                    profile.favourite = distance;
                }
            }
            profiles[layer].push_back(profile);
        }
    }
    return profiles;
}

ml_scratch::TokenSequence marked_copy_stream() {
    ml_scratch::DeterministicRandom random{5};
    std::vector<std::size_t> bits;
    ml_scratch::TokenSequence ids;
    for (std::size_t index = 0; index < copy_pairs; ++index) {
        bits.push_back(random.bernoulli(0.5) ? 1 : 0);
        ids.push_back(bits.back());
        ids.push_back(2 + (index >= copy_delay ? bits[index - copy_delay] : 0));
    }
    return ids;
}

std::string show(const std::string& text) {
    std::string shown;
    for (const char character : text) {
        shown += character == '\n' ? std::string{"\n    "} : std::string{character};
    }
    return shown;
}

} // namespace

int main() {
    const std::string path =
        data_path("ML_SCRATCH_TINY_SHAKESPEARE", "data/tiny_shakespeare/input.txt");
    ml_scratch::TextCorpus corpus;
    try {
        corpus = ml_scratch::load_text_corpus(path, corpus_characters);
    } catch (const std::exception& error) {
        std::cerr << "could not load the corpus from " << path << ": " << error.what() << "\n\n"
                  << "Download it first: ./scripts/download_tiny_shakespeare.sh\n";
        return 1;
    }
    const std::size_t vocabulary = corpus.vocabulary_size();

    std::cout << std::fixed << std::setprecision(4)
              << "C++ decoder-only Transformer on Tiny Shakespeare (seed=" << seed << ")\n\n"
              << "  the same " << corpus_characters << "-character prefix, splits and vocabulary "
              << "as experiments 15 and 16: " << corpus.train.size() << " training, "
              << corpus.validation.size() << " validation and " << corpus.test.size()
              << " test characters, " << vocabulary << " distinct\n"
              << "  every run: random windows of context_length + 1 characters, " << batch_size
              << " per update, Adam, gradient norm clipped at " << clip_norm
              << "; an epoch is one training text's worth of characters\n"
              << "  validation and test are scored in abutting windows, so a character early in "
                 "a window sees only the few before it\n";

    // ---------- Part one: the main model against the recurrent networks ----------
    std::cout << "\npart one: " << layers << " blocks of " << heads << "-head attention and a "
              << 4 * model_size << "-wide feed-forward layer, width " << model_size
              << ", context " << context_length << ", Adam at " << learning_rate << ", "
              << epochs << " epochs\n";
    ml_scratch::CharTransformer main_model{
        configuration(model_size, context_length, heads, vocabulary), seed};
    const Run main_run = train("transformer", main_model, epochs, corpus, learning_rate);
    std::cout << "\n  validation by epoch against the recurrent networks of experiments 15 and 16\n"
              << "  epoch   transformer   elman 128   GRU 70\n";
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "  " << std::setw(5) << epoch + 1 << std::setw(14)
                  << main_run.validation_loss[epoch] << std::setw(12) << elman_validation[epoch]
                  << std::setw(9) << gru_validation[epoch] << '\n';
    }
    std::cout << '\n';
    print_summary_header();
    print_summary(main_run);
    std::cout << "  (experiment 15's elman cell reached " << elman_test
              << " on test after 12 epochs, experiment 16's GRU " << gru_test
              << ", the trigram " << trigram_test << ")\n";

    const std::string checkpoint =
        data_path("ML_SCRATCH_TRANSFORMER_CHECKPOINT", "data/char_transformer.checkpoint");
    try {
        main_model.save(checkpoint);
        const auto restored = ml_scratch::CharTransformer::load(checkpoint);
        std::cout << "  checkpoint " << checkpoint << " round-trips: "
                  << (restored.parameters() == main_model.parameters() ? "yes" : "NO") << '\n';
    } catch (const std::exception& error) {
        std::cout << "  checkpoint not written: " << error.what() << '\n';
    }

    // Where the heads look.
    const auto profiles = attention_profiles(main_model, corpus.validation);
    std::cout << "\n  where each head looks, from the last position of 50 validation windows: "
                 "mean distance behind the query,\n"
              << "  weight within three characters, and the single most attended distance\n"
              << "  block   head   mean distance   within 3   favourite\n";
    double longest_mean = 0.0;
    double most_local = 0.0;
    for (std::size_t layer = 0; layer < layers; ++layer) {
        for (std::size_t head = 0; head < heads; ++head) {
            const HeadProfile& profile = profiles[layer][head];
            std::cout << "  " << std::setw(5) << layer + 1 << std::setw(7) << head + 1
                      << std::setw(16) << std::setprecision(1) << profile.mean_distance
                      << std::setw(11) << std::setprecision(3) << profile.within_three
                      << std::setw(12) << profile.favourite << std::setprecision(4) << '\n';
            longest_mean = std::max(longest_mean, profile.mean_distance);
            most_local = std::max(most_local, profile.within_three);
        }
    }

    // A sample.
    {
        const std::string prompt = "ROMEO:\n";
        std::mt19937 engine{seed};
        const auto ids = main_model.generate(corpus.encode(prompt), sample_length, 0.5, engine);
        std::cout << "\n  sample after the prompt \"ROMEO:\", temperature 0.5, seeded:\n    "
                  << show(prompt + corpus.decode(ids)) << "\n";
    }

    // ---------- Part two: a recurrent-sized model and its ablations ----------
    std::cout << "\npart two: width " << small_model_size << " (the recurrent networks' parameter "
              << "count) and its ablations, " << small_epochs << " epochs each\n";
    ml_scratch::CharTransformer small{
        configuration(small_model_size, context_length, heads, vocabulary), seed};
    const Run small_run = train("width 32", small, small_epochs, corpus, learning_rate);
    ml_scratch::TransformerConfig no_positions =
        configuration(small_model_size, context_length, heads, vocabulary);
    no_positions.positional_encoding = false;
    ml_scratch::CharTransformer unordered{no_positions, seed};
    const Run no_positions_run =
        train("no positional encoding", unordered, small_epochs, corpus, learning_rate);
    ml_scratch::TransformerConfig relative =
        configuration(small_model_size, context_length, heads, vocabulary);
    relative.relative_position_bias = true;
    ml_scratch::CharTransformer with_bias{relative, seed};
    const Run relative_run =
        train("plus relative position bias", with_bias, small_epochs, corpus, learning_rate);
    ml_scratch::CharTransformer one_head{
        configuration(small_model_size, context_length, 1, vocabulary), seed};
    const Run one_head_run = train("one head", one_head, small_epochs, corpus, learning_rate);
    ml_scratch::CharTransformer short_model{
        configuration(small_model_size, short_context, heads, vocabulary), seed};
    const Run short_run =
        train("context 16", short_model, small_epochs, corpus, learning_rate);
    std::cout << '\n';
    print_summary_header();
    for (const Run* run : {&small_run, &no_positions_run, &relative_run, &one_head_run, &short_run}) {
        print_summary(*run);
    }
    std::cout << "  (the elman cell's validation at epoch " << small_epochs << " was "
              << elman_validation[small_epochs - 1] << ", the GRU's "
              << gru_validation[small_epochs - 1] << ")\n";

    // ---------- Part three: the marked copy stream ----------
    const double floor = std::log(2.0) / 2.0;
    std::cout << "\npart three: a delayed-copy stream, copies written as 2 and 3 so the current "
              << "token says whether a copy comes next\n"
              << "  every copy repeats the random bit that entered " << 2 * copy_delay
              << " steps before; the floor is ln 2 / 2 = " << floor
              << " and chance on the copies is ln 2 = " << std::log(2.0) << '\n'
              << "  recurrent cells: " << rnn_copy_hidden << " units, window " << copy_context
              << ", " << rnn_copy_epochs << " epochs; transformers: width " << copy_model_size
              << ", one block, two heads, context " << copy_context << ", " << copy_epochs
              << " epochs, scored with a stride of " << copy_context / 2 << "\n\n"
              << "  model                              loss   weight on the source bit\n";
    const auto stream = marked_copy_stream();
    struct CopyResult {
        std::string label;
        double loss;
        double source_weight;
    };
    std::vector<CopyResult> copy_results;
    for (const auto cell : {ml_scratch::RecurrentCell::elman, ml_scratch::RecurrentCell::lstm,
                            ml_scratch::RecurrentCell::gru}) {
        ml_scratch::CharRnn network{4, rnn_copy_hidden, seed, cell};
        ml_scratch::RnnTrainingConfig config;
        config.learning_rate = 0.01;
        config.epochs = rnn_copy_epochs;
        config.sequence_length = copy_context;
        config.batch_size = 8;
        static_cast<void>(network.fit(stream, config));
        copy_results.push_back({ml_scratch::recurrent_cell_name(cell), network.loss(stream), -1.0});
    }
    const auto copy_transformer = [&](const std::string& label, const bool positions,
                                      const bool bias) {
        ml_scratch::TransformerConfig config{4, copy_context, copy_model_size, 2, 1, 0, positions,
                                             bias};
        ml_scratch::CharTransformer network{config, seed};
        ml_scratch::TransformerTrainingConfig training;
        training.learning_rate = 0.005;
        training.epochs = copy_epochs;
        training.batch_size = 8;
        training.seed = seed;
        static_cast<void>(network.fit(stream, training));
        // At the last position of a window ending on a copy, the weight a head puts on the
        // source bit, twenty steps back; the best head is reported.
        const ml_scratch::TokenSequence window(
            stream.begin() + 1, stream.begin() + 1 + static_cast<std::ptrdiff_t>(copy_context));
        const auto weights = network.attention_weights(window);
        double best = 0.0;
        for (std::size_t head = 0; head < 2; ++head) {
            best = std::max(best, weights[0][head][copy_context - 1][copy_context - 1 - 2 * copy_delay]);
        }
        copy_results.push_back({label, network.evaluate(stream, copy_context / 2), best});
    };
    copy_transformer("transformer, sinusoidal", true, false);
    copy_transformer("transformer, relative bias only", false, true);
    copy_transformer("transformer, both", true, true);
    copy_transformer("transformer, neither", false, false);
    for (const CopyResult& result : copy_results) {
        std::string padded = result.label;
        padded.resize(32, ' ');
        std::cout << "  " << padded << std::setw(9) << result.loss;
        if (result.source_weight >= 0.0) {
            std::cout << std::setw(13) << result.source_weight;
        }
        std::cout << '\n';
    }

    // ---------- Summary and criteria ----------
    std::cout << "\nsummary\n"
              << "  transformer test " << main_run.test_loss << " with " << main_run.parameters
              << " parameters at " << std::setprecision(0) << main_run.characters_per_second
              << " chars/s; elman " << std::setprecision(4) << elman_test << ", GRU " << gru_test
              << ", trigram " << trigram_test << '\n'
              << "  width 32 at epoch " << small_epochs << ": " << small_run.validation_loss.back()
              << " validation; no positions " << no_positions_run.validation_loss.back()
              << ", relative bias " << relative_run.validation_loss.back() << ", one head "
              << one_head_run.validation_loss.back() << ", context 16 "
              << short_run.validation_loss.back() << '\n'
              << "  heads: longest mean distance " << std::setprecision(1) << longest_mean
              << ", most local head keeps " << std::setprecision(3) << most_local
              << " of its weight within three characters\n"
              << std::setprecision(4) << "  copy at " << 2 * copy_delay << " steps: elman "
              << copy_results[0].loss << ", LSTM " << copy_results[1].loss << ", GRU "
              << copy_results[2].loss << ", transformer " << copy_results[3].loss
              << " (source weight " << copy_results[3].source_weight << "), without positions "
              << copy_results[6].loss << '\n';

    const bool beat_the_trigram = main_run.test_loss < trigram_test;
    // At this budget the transformer's headline claim against the recurrent networks is
    // measured, not assumed: it has to beat the elman cell on test.
    const bool beat_the_elman = main_run.test_loss < elman_test;
    const bool still_improving =
        main_run.validation_loss.back() == *std::min_element(main_run.validation_loss.begin(),
                                                             main_run.validation_loss.end());
    // Order matters: without a positional encoding the model must do markedly worse, and
    // telling heads how far away a key is must help. One head against four and a 16-character
    // context against 64 are reported, not asserted: at this size the run found both ablations
    // AHEAD of the full model, and the README says why that is not a surprise.
    const bool positions_matter =
        no_positions_run.validation_loss.back() > small_run.validation_loss.back() + 0.2;
    const bool relative_bias_helps =
        relative_run.validation_loss.back() < small_run.validation_loss.back();
    // Attention reaches the source bit in one lookup, and the head that does it says so.
    const bool copy_learned = copy_results[3].loss < floor + 0.03 &&
                              copy_results[3].source_weight > 0.5;
    const bool copy_needs_positions = copy_results[6].loss > copy_results[3].loss + 0.1;
    const bool elman_still_cannot = copy_results[0].loss > std::log(2.0) - 0.05;

    if (!beat_the_trigram || !beat_the_elman || !still_improving || !positions_matter ||
        !relative_bias_helps || !copy_learned || !copy_needs_positions || !elman_still_cannot) {
        std::cerr << "experiment failed its criteria: test " << main_run.test_loss << ", width 32 "
                  << small_run.validation_loss.back() << '/'
                  << no_positions_run.validation_loss.back() << '/'
                  << one_head_run.validation_loss.back() << '/' << short_run.validation_loss.back()
                  << ", copy";
        for (const CopyResult& result : copy_results) {
            std::cerr << ' ' << result.loss;
        }
        std::cerr << '\n';
        return 1;
    }
    return 0;
}
