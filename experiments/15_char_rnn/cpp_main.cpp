#include "ml_scratch/rnn.hpp"
#include "ml_scratch/text_corpus.hpp"

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

constexpr std::uint32_t seed = 20260919;
// The sequence milestones share this prefix of the corpus, its splits, and its vocabulary.
constexpr std::size_t corpus_characters = 500'000;
constexpr std::size_t hidden_size = 128;
constexpr std::size_t sequence_length = 50;
constexpr std::size_t batch_size = 32;
constexpr double learning_rate = 0.002;
constexpr double clip_norm = 5.0;
constexpr std::size_t epochs = 12;
// Part two trains the same network with shorter truncation windows for the same number of epochs.
constexpr std::size_t short_window = 5;
constexpr std::size_t shortest_window = 1;
// Part three trains with plain gradient descent at a rate that needs clipping, for fewer epochs.
constexpr double sgd_learning_rate = 0.7;
constexpr std::size_t sgd_epochs = 4;
constexpr double tight_clip_norm = 1.0;
// Gradient reach is averaged over this many windows of this many characters from the validation
// split.
constexpr std::size_t reach_windows = 20;
constexpr std::size_t reach_length = 60;
constexpr std::size_t sample_length = 300;

std::string data_path(const char* variable, const std::string& fallback) {
    if (const char* override_path = std::getenv(variable)) {
        return override_path;
    }
    return fallback;
}

struct Scores {
    double validation;
    double test;
};

void print_scores_header() {
    std::cout << "  model                 parameters   validation   bits/char   perplexity       "
                 "test   bits/char   perplexity\n";
}

void print_scores(const std::string& label, const std::size_t parameters, const Scores& scores) {
    std::string padded = label;
    padded.resize(20, ' ');
    std::cout << "  " << padded << std::setw(12) << parameters << std::setw(13) << scores.validation
              << std::setw(12) << ml_scratch::bits_per_character(scores.validation)
              << std::setw(13) << ml_scratch::perplexity(scores.validation) << std::setw(11)
              << scores.test << std::setw(12) << ml_scratch::bits_per_character(scores.test)
              << std::setw(13) << ml_scratch::perplexity(scores.test) << '\n';
}

void print_epoch_header() {
    std::cout << "  epoch   train loss   validation   bits/char   clipped   max |grad|   mean "
                 "|grad|   chars/s   seconds\n";
}

void print_epoch(const std::size_t epoch, const ml_scratch::RnnTrainingResult& result,
                 const double seconds) {
    const std::size_t index = result.epochs - 1;
    std::cout << "  " << std::setw(5) << epoch << std::setw(13) << result.loss_per_epoch[index]
              << std::setw(13) << result.validation_loss_per_epoch[index] << std::setw(12)
              << ml_scratch::bits_per_character(result.validation_loss_per_epoch[index])
              << std::setw(10) << result.clipped_fraction_per_epoch[index] << std::setw(13)
              << std::setprecision(2) << result.max_gradient_norm_per_epoch[index] << std::setw(13)
              << result.mean_gradient_norm_per_epoch[index] << std::setw(10) << std::setprecision(0)
              << result.characters_per_second << std::setw(10) << std::setprecision(1) << seconds
              << std::setprecision(4) << '\n';
}

// Mean gradient-reach profile over fixed windows of the validation text: the norm of the last
// prediction's gradient with respect to the hidden state `lag` steps earlier, relative to lag 0.
std::vector<double> mean_reach(const ml_scratch::CharRnn& network,
                               const ml_scratch::TokenSequence& text) {
    std::vector<double> total(reach_length, 0.0);
    const std::size_t stride = text.size() / reach_windows;
    for (std::size_t window = 0; window < reach_windows; ++window) {
        const auto begin = text.begin() + static_cast<std::ptrdiff_t>(window * stride);
        const ml_scratch::TokenSequence ids(begin, begin + reach_length + 1);
        const auto reach = network.gradient_reach(ids);
        for (std::size_t lag = 0; lag < reach_length; ++lag) {
            total[lag] += reach[lag] / reach[0] / static_cast<double>(reach_windows);
        }
    }
    return total;
}

void print_reach(const std::string& label, const std::vector<double>& reach) {
    std::string padded = label;
    padded.resize(18, ' ');
    std::cout << "  " << padded;
    for (const std::size_t lag : {0, 1, 2, 5, 10, 20, 30, 50}) {
        std::cout << std::setw(11) << std::scientific << std::setprecision(2) << reach[lag];
    }
    std::cout << std::defaultfloat << std::fixed << std::setprecision(4) << '\n';
}

std::string show(const std::string& text) {
    // Newlines are shown as a visible marker plus an indented line, so a sample stays readable.
    std::string shown;
    for (const char character : text) {
        if (character == '\n') {
            shown += "\n    ";
        } else {
            shown += character;
        }
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
              << "C++ character-level RNN on Tiny Shakespeare (seed=" << seed << ")\n\n"
              << "  the first " << corpus_characters << " characters of the corpus: "
              << corpus.train.size() << " training, " << corpus.validation.size()
              << " validation and " << corpus.test.size() << " test characters, contiguous in "
              << "that order; " << vocabulary << " distinct characters\n"
              << "  network: one-hot " << vocabulary << " -> " << hidden_size << " tanh -> "
              << vocabulary << " logits; truncated BPTT over " << sequence_length << " steps, "
              << batch_size << " streams, Adam at " << learning_rate << ", gradient norm clipped "
              << "at " << clip_norm << "\n"
              << "  losses are mean cross-entropy per predicted character in nats; "
                 "bits/char divides by ln 2 and perplexity is exp\n\n";

    // ---------- Count-based baselines ----------
    std::cout << "count-based baselines, fit on the training split with add-one smoothing\n";
    print_scores_header();
    std::vector<Scores> ngram_scores;
    for (const std::size_t order : {1, 2, 3}) {
        ml_scratch::CharNgram model{order, vocabulary, 1.0};
        model.fit(corpus.train);
        const Scores scores{model.cross_entropy(corpus.validation),
                            model.cross_entropy(corpus.test)};
        ngram_scores.push_back(scores);
        std::size_t contexts = 1;
        for (std::size_t index = 1; index < order; ++index) {
            contexts *= vocabulary;
        }
        print_scores(order == 1 ? "unigram" : order == 2 ? "bigram" : "trigram",
                     contexts * vocabulary, scores);
    }
    const double uniform = std::log(static_cast<double>(vocabulary));
    std::cout << "  (a uniform guess over " << vocabulary << " characters scores " << uniform
              << " nats, " << ml_scratch::bits_per_character(uniform) << " bits/char)\n\n";

    // ---------- The recurrent network ----------
    ml_scratch::CharRnn network{vocabulary, hidden_size, seed};
    const auto initial_parameters = network.parameters();
    const Scores untrained{network.loss(corpus.validation), network.loss(corpus.test)};
    const std::vector<double> reach_before = mean_reach(network, corpus.validation);

    std::cout << "training, one epoch at a time so the validation split is scored after each\n";
    print_epoch_header();
    ml_scratch::RnnTrainingConfig config;
    config.learning_rate = learning_rate;
    config.epochs = 1;
    config.sequence_length = sequence_length;
    config.batch_size = batch_size;
    config.clip_norm = clip_norm;
    std::vector<double> validation_history;
    std::vector<double> max_norm_history;
    double clipped_total = 0.0;
    double throughput_total = 0.0;
    double training_seconds = 0.0;
    for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
        const auto start = std::chrono::steady_clock::now();
        const auto result = network.fit(corpus.train, corpus.validation, config);
        const auto finish = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(finish - start).count();
        training_seconds += seconds;
        print_epoch(epoch, result, seconds);
        validation_history.push_back(result.validation_loss_per_epoch.front());
        max_norm_history.push_back(result.max_gradient_norm_per_epoch.front());
        clipped_total += result.clipped_fraction_per_epoch.front() / static_cast<double>(epochs);
        throughput_total += result.characters_per_second / static_cast<double>(epochs);
    }
    const Scores trained{network.loss(corpus.validation), network.loss(corpus.test)};
    std::cout << '\n';
    print_scores_header();
    print_scores("untrained RNN", network.parameter_count(), untrained);
    print_scores("trained RNN", network.parameter_count(), trained);
    std::cout << "  " << std::setprecision(0) << throughput_total
              << " training characters per second, " << std::setprecision(1) << training_seconds
              << " s of training in total\n"
              << std::setprecision(4);

    const std::string checkpoint =
        data_path("ML_SCRATCH_CHAR_RNN_CHECKPOINT", "data/char_rnn.checkpoint");
    try {
        network.save(checkpoint);
        const auto restored = ml_scratch::CharRnn::load(checkpoint);
        std::cout << "  checkpoint " << checkpoint << " round-trips: "
                  << (restored.parameters() == network.parameters() ? "yes" : "NO") << "\n";
    } catch (const std::exception& error) {
        std::cout << "  checkpoint not written: " << error.what() << '\n';
    }

    // ---------- Samples ----------
    const std::string prompt = "ROMEO:\n";
    std::cout << "\nsamples after the prompt \"ROMEO:\", " << sample_length
              << " characters each, seeded\n";
    for (const double temperature : {0.5, 1.0}) {
        std::mt19937 engine{seed};
        const auto ids = network.generate(corpus.encode(prompt), sample_length, temperature, engine);
        std::cout << "\n  temperature " << std::setprecision(1) << temperature
                  << std::setprecision(4) << ":\n    " << show(prompt + corpus.decode(ids))
                  << "\n";
    }

    // ---------- Gradient reach ----------
    const std::vector<double> reach_after = mean_reach(network, corpus.validation);
    std::cout << "\ngradient reach: |dL_T/dh_(T-lag)| relative to lag 0, for the last prediction "
                 "of "
              << reach_windows << " validation windows\n"
              << "  lag               ";
    for (const std::size_t lag : {0, 1, 2, 5, 10, 20, 30, 50}) {
        std::cout << std::setw(11) << lag;
    }
    std::cout << '\n';
    print_reach("before training", reach_before);
    print_reach("after training", reach_after);

    // ---------- Part two: the truncation window ----------
    // Every run below starts from the same initial parameters as the main one.
    const auto train_epochs = [&](const std::string& label,
                                  const ml_scratch::RnnTrainingConfig& variant,
                                  const std::size_t count) {
        std::cout << "\n  " << label << '\n';
        print_epoch_header();
        ml_scratch::CharRnn variant_network{vocabulary, hidden_size, seed};
        variant_network.set_parameters(initial_parameters);
        ml_scratch::RnnTrainingConfig one_epoch = variant;
        one_epoch.epochs = 1;
        std::vector<ml_scratch::RnnTrainingResult> history;
        for (std::size_t epoch = 1; epoch <= count; ++epoch) {
            const auto start = std::chrono::steady_clock::now();
            history.push_back(variant_network.fit(corpus.train, corpus.validation, one_epoch));
            const auto finish = std::chrono::steady_clock::now();
            print_epoch(epoch, history.back(),
                        std::chrono::duration<double>(finish - start).count());
        }
        return history;
    };

    std::cout << "\npart two: the truncation window, same network and optimizer, " << epochs
              << " epochs each\n"
              << "  a shorter window makes proportionally more updates per epoch, so the runs are "
                 "compared both at\n"
              << "  equal epochs (equal characters seen) and at equal update counts\n";
    ml_scratch::RnnTrainingConfig short_config = config;
    short_config.sequence_length = short_window;
    const auto short_history =
        train_epochs("window " + std::to_string(short_window), short_config, epochs);
    ml_scratch::RnnTrainingConfig shortest_config = config;
    shortest_config.sequence_length = shortest_window;
    const auto shortest_history =
        train_epochs("window " + std::to_string(shortest_window), shortest_config, epochs);

    // ---------- Part three: clipping under plain gradient descent ----------
    std::cout << "\npart three: gradient clipping under plain gradient descent at "
              << sgd_learning_rate << ", " << sgd_epochs << " epochs each\n"
              << "  Adam divides every step by a running gradient scale, which is why clipping "
                 "was idle above;\n"
              << "  plain gradient descent takes the gradient at face value\n";
    ml_scratch::RnnTrainingConfig sgd_config = config;
    sgd_config.optimizer.kind = ml_scratch::OptimizerKind::gradient_descent;
    sgd_config.learning_rate = sgd_learning_rate;
    sgd_config.clip_norm = 0.0;
    const auto sgd_free = train_epochs("no clipping", sgd_config, sgd_epochs);
    sgd_config.clip_norm = clip_norm;
    const auto sgd_clipped = train_epochs("clipped at " + std::to_string(int(clip_norm)),
                                          sgd_config, sgd_epochs);
    sgd_config.clip_norm = tight_clip_norm;
    const auto sgd_tight = train_epochs("clipped at " + std::to_string(int(tight_clip_norm)),
                                        sgd_config, sgd_epochs);

    // ---------- Summary and criteria ----------
    const auto final_validation = [](const std::vector<ml_scratch::RnnTrainingResult>& history) {
        return history.back().validation_loss_per_epoch.front();
    };
    const auto largest_norm = [](const std::vector<ml_scratch::RnnTrainingResult>& history) {
        double largest = 0.0;
        for (const auto& result : history) {
            largest = std::max(largest, result.max_gradient_norm_per_epoch.front());
        }
        return largest;
    };
    const double max_norm_overall =
        *std::max_element(max_norm_history.begin(), max_norm_history.end());
    const double best_validation =
        *std::min_element(validation_history.begin(), validation_history.end());
    // Equal update counts: one epoch of the short window makes as many updates as
    // sequence_length / short_window epochs of the main run.
    const std::size_t matching_epoch = sequence_length / short_window;
    const double short_after_one_epoch = short_history.front().validation_loss_per_epoch.front();
    const double main_at_matching = validation_history[matching_epoch - 1];

    std::cout << "\nsummary\n"
              << "  test cross-entropy: unigram " << ngram_scores[0].test << ", bigram "
              << ngram_scores[1].test << ", trigram " << ngram_scores[2].test << ", RNN "
              << trained.test << " (" << ml_scratch::bits_per_character(trained.test)
              << " bits/char, perplexity " << ml_scratch::perplexity(trained.test) << ")\n"
              << "  validation went from " << untrained.validation << " untrained to "
              << trained.validation << "; best epoch " << best_validation << '\n'
              << "  " << std::setprecision(2) << 100.0 * clipped_total << "% of the main run's "
              << "updates were clipped; largest pre-clip gradient norm " << max_norm_overall
              << " against a threshold of " << clip_norm << std::setprecision(4) << '\n'
              << "  gradient reach at lag 20: " << std::scientific << std::setprecision(2)
              << reach_before[20] << " before training, " << reach_after[20] << " after"
              << std::defaultfloat << std::fixed << std::setprecision(4) << '\n'
              << "  after " << epochs << " epochs: validation " << trained.validation
              << " with a " << sequence_length << "-step window, "
              << final_validation(short_history) << " with " << short_window << ", "
              << final_validation(shortest_history) << " with " << shortest_window << '\n'
              << "  at equal updates (" << short_history.front().updates << "): "
              << main_at_matching << " with " << sequence_length << " steps after "
              << matching_epoch << " epochs, " << short_after_one_epoch << " with "
              << short_window << " steps after 1\n"
              << "  gradient descent at " << sgd_learning_rate << ": validation "
              << final_validation(sgd_free) << " unclipped (largest norm " << std::setprecision(1)
              << largest_norm(sgd_free) << std::setprecision(4) << "), "
              << final_validation(sgd_clipped) << " clipped at " << clip_norm << ", "
              << final_validation(sgd_tight) << " clipped at " << tight_clip_norm << '\n';

    const bool beat_every_ngram = trained.test < ngram_scores[2].test &&
                                  trained.test < ngram_scores[1].test &&
                                  trained.test < ngram_scores[0].test;
    const bool still_improving = validation_history.back() == best_validation;
    const bool learned_something = trained.validation < 0.75 * untrained.validation;
    const bool test_matches_validation = std::abs(trained.test - trained.validation) < 0.2;
    // The gradient reaching 20 steps back is a small fraction of what reaches the last state:
    // the vanishing gradient, measured.
    const bool gradient_vanishes = reach_after[20] < 0.1;
    // A one-step window cannot use context beyond what the carried state happens to hold, and
    // has to lose to both longer windows at equal epochs; at equal update counts the longer
    // window has to beat the shorter one, since each of its updates sees ten times the context.
    const bool one_step_window_is_worst =
        final_validation(shortest_history) > final_validation(short_history) &&
        final_validation(shortest_history) > trained.validation;
    const bool longer_window_wins_per_update = main_at_matching < short_after_one_epoch;
    // Under plain gradient descent the unclipped run has to show the explosion — a largest norm
    // more than an order of magnitude above the tight threshold — and end worse than both
    // clipped runs; the tighter threshold is expected to be the best of the three.
    const bool unclipped_exploded = largest_norm(sgd_free) > 10.0 * tight_clip_norm;
    const bool clipping_rescued =
        final_validation(sgd_clipped) < final_validation(sgd_free) &&
        final_validation(sgd_tight) < final_validation(sgd_clipped);

    if (!beat_every_ngram || !still_improving || !learned_something || !test_matches_validation ||
        !gradient_vanishes || !one_step_window_is_worst || !longer_window_wins_per_update ||
        !unclipped_exploded || !clipping_rescued) {
        std::cerr << "experiment failed its criteria: test=" << trained.test
                  << ", trigram=" << ngram_scores[2].test << ", validation history";
        for (const double value : validation_history) {
            std::cerr << ' ' << value;
        }
        std::cerr << ", reach[20]=" << reach_after[20]
                  << ", windows=" << trained.validation << '/' << final_validation(short_history)
                  << '/' << final_validation(shortest_history)
                  << ", sgd=" << final_validation(sgd_free) << '/'
                  << final_validation(sgd_clipped) << '/' << final_validation(sgd_tight) << '\n';
        return 1;
    }
    return 0;
}
