#include "ml_scratch/deterministic_random.hpp"
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
// The same prefix, splits, vocabulary, optimizer, window and streams as experiment 15.
constexpr std::size_t corpus_characters = 500'000;
constexpr std::size_t sequence_length = 50;
constexpr std::size_t batch_size = 32;
constexpr double learning_rate = 0.002;
constexpr double clip_norm = 5.0;
constexpr std::size_t epochs = 12;

// Part one holds the parameter count near experiment 15's elman network: 128 elman units come
// to 32703 parameters, 58 LSTM units to 32021, and 70 GRU units to 32613.
constexpr std::size_t elman_hidden = 128;
constexpr std::size_t lstm_matched_hidden = 58;
constexpr std::size_t gru_matched_hidden = 70;
// Part two holds the hidden size at 128 instead, for fewer epochs, since a gated cell of that
// width costs three to four times the elman's arithmetic per character.
constexpr std::size_t wide_hidden = 128;
constexpr std::size_t wide_epochs = 4;

// Gradient reach, as in experiment 15.
constexpr std::size_t reach_windows = 20;
constexpr std::size_t reach_length = 60;

// Part three: a stream alternating a random bit with a copy of the bit written `delay` pairs
// earlier. The copy at index 2i + 1 repeats the bit at index 2(i - delay), which entered the
// network 2 * delay steps before the prediction is made. The random bits are unpredictable, so
// the best possible loss is ln 2 / 2.
constexpr std::size_t copy_pairs = 6'000;
constexpr std::size_t copy_hidden = 32;
constexpr std::size_t copy_window = 40;
constexpr std::size_t copy_epochs = 60;
constexpr double copy_learning_rate = 0.01;

constexpr std::size_t sample_length = 300;

std::string data_path(const char* variable, const std::string& fallback) {
    if (const char* override_path = std::getenv(variable)) {
        return override_path;
    }
    return fallback;
}

struct Run {
    std::string label;
    ml_scratch::RecurrentCell cell;
    std::size_t hidden;
    std::size_t parameters;
    std::vector<double> train_loss;
    std::vector<double> validation_loss;
    double test_loss;
    double characters_per_second;
    double seconds;
};

void print_epoch_header() {
    std::cout << "  epoch   train loss   validation   bits/char   max |grad|   chars/s   seconds\n";
}

// Trains one network epoch by epoch, scoring validation after each, and returns its record.
Run train(const std::string& label, const ml_scratch::RecurrentCell cell, const std::size_t hidden,
          const std::size_t count, const ml_scratch::TextCorpus& corpus,
          ml_scratch::CharRnn* keep = nullptr) {
    ml_scratch::CharRnn network{corpus.vocabulary_size(), hidden, seed, cell};
    ml_scratch::RnnTrainingConfig config;
    config.learning_rate = learning_rate;
    config.epochs = 1;
    config.sequence_length = sequence_length;
    config.batch_size = batch_size;
    config.clip_norm = clip_norm;

    std::cout << "\n  " << label << ": " << hidden << " units, " << network.parameter_count()
              << " parameters\n";
    print_epoch_header();
    Run run{label, cell, hidden, network.parameter_count(), {}, {}, 0.0, 0.0, 0.0};
    for (std::size_t epoch = 1; epoch <= count; ++epoch) {
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
                  << ml_scratch::bits_per_character(run.validation_loss.back()) << std::setw(13)
                  << std::setprecision(2) << result.max_gradient_norm_per_epoch.front()
                  << std::setw(10) << std::setprecision(0) << result.characters_per_second
                  << std::setw(10) << std::setprecision(1) << seconds << std::setprecision(4)
                  << '\n';
    }
    run.test_loss = network.loss(corpus.test);
    if (keep != nullptr) {
        *keep = network;
    }
    return run;
}

void print_summary_header() {
    std::cout << "  model                 units   parameters   validation       test   bits/char   "
                 "perplexity   chars/s   seconds\n";
}

void print_summary(const Run& run) {
    std::string padded = run.label;
    padded.resize(20, ' ');
    std::cout << "  " << padded << std::setw(7) << run.hidden << std::setw(13) << run.parameters
              << std::setw(13) << run.validation_loss.back() << std::setw(11) << run.test_loss
              << std::setw(12) << ml_scratch::bits_per_character(run.test_loss) << std::setw(13)
              << ml_scratch::perplexity(run.test_loss) << std::setw(10) << std::setprecision(0)
              << run.characters_per_second << std::setw(10) << std::setprecision(1) << run.seconds
              << std::setprecision(4) << '\n';
}

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

ml_scratch::TokenSequence delayed_copy_stream(const std::size_t delay) {
    ml_scratch::DeterministicRandom random{5};
    std::vector<std::size_t> bits;
    ml_scratch::TokenSequence ids;
    for (std::size_t index = 0; index < copy_pairs; ++index) {
        bits.push_back(random.bernoulli(0.5) ? 1 : 0);
        ids.push_back(bits.back());
        ids.push_back(index >= delay ? bits[index - delay] : 0);
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

    std::cout << std::fixed << std::setprecision(4)
              << "C++ LSTM and GRU against the elman cell on Tiny Shakespeare (seed=" << seed
              << ")\n\n"
              << "  the same " << corpus_characters << "-character prefix, splits and vocabulary "
              << "as experiment 15: " << corpus.train.size() << " training, "
              << corpus.validation.size() << " validation and " << corpus.test.size()
              << " test characters, " << corpus.vocabulary_size() << " distinct\n"
              << "  every run: truncated BPTT over " << sequence_length << " steps, " << batch_size
              << " streams, Adam at " << learning_rate << ", gradient norm clipped at "
              << clip_norm << "; losses in nats per character\n";

    // ---------- Part one: equal parameter counts ----------
    std::cout << "\npart one: three cells at the same parameter count, " << epochs
              << " epochs each\n";
    ml_scratch::CharRnn elman{corpus.vocabulary_size(), elman_hidden, seed};
    ml_scratch::CharRnn lstm{corpus.vocabulary_size(), lstm_matched_hidden, seed,
                             ml_scratch::RecurrentCell::lstm};
    ml_scratch::CharRnn gru{corpus.vocabulary_size(), gru_matched_hidden, seed,
                            ml_scratch::RecurrentCell::gru};
    const Run elman_run =
        train("elman", ml_scratch::RecurrentCell::elman, elman_hidden, epochs, corpus, &elman);
    const Run lstm_run = train("LSTM", ml_scratch::RecurrentCell::lstm, lstm_matched_hidden,
                               epochs, corpus, &lstm);
    const Run gru_run =
        train("GRU", ml_scratch::RecurrentCell::gru, gru_matched_hidden, epochs, corpus, &gru);
    std::cout << '\n';
    print_summary_header();
    print_summary(elman_run);
    print_summary(lstm_run);
    print_summary(gru_run);

    // ---------- Part two: equal hidden size ----------
    std::cout << "\npart two: the gated cells at " << wide_hidden << " units, " << wide_epochs
              << " epochs, against the elman run at epoch " << wide_epochs << '\n';
    const Run wide_lstm =
        train("LSTM", ml_scratch::RecurrentCell::lstm, wide_hidden, wide_epochs, corpus);
    const Run wide_gru =
        train("GRU", ml_scratch::RecurrentCell::gru, wide_hidden, wide_epochs, corpus);
    std::cout << '\n';
    print_summary_header();
    print_summary(wide_lstm);
    print_summary(wide_gru);
    std::cout << "  (the elman run's validation at epoch " << wide_epochs << " was "
              << elman_run.validation_loss[wide_epochs - 1] << ")\n";

    // ---------- Gradient reach of the part-one networks ----------
    std::cout << "\ngradient reach after training: |dL_T/d(state)_(T-lag)| relative to lag 0, "
                 "mean over "
              << reach_windows << " validation windows\n"
              << "  lag               ";
    for (const std::size_t lag : {0, 1, 2, 5, 10, 20, 30, 50}) {
        std::cout << std::setw(11) << lag;
    }
    std::cout << '\n';
    const auto elman_reach = mean_reach(elman, corpus.validation);
    const auto lstm_reach = mean_reach(lstm, corpus.validation);
    const auto gru_reach = mean_reach(gru, corpus.validation);
    print_reach("elman", elman_reach);
    print_reach("LSTM", lstm_reach);
    print_reach("GRU", gru_reach);

    // ---------- A sample from the best part-one model ----------
    const Run* best = &elman_run;
    ml_scratch::CharRnn* best_network = &elman;
    if (lstm_run.test_loss < best->test_loss) {
        best = &lstm_run;
        best_network = &lstm;
    }
    if (gru_run.test_loss < best->test_loss) {
        best = &gru_run;
        best_network = &gru;
    }
    {
        const std::string prompt = "ROMEO:\n";
        std::mt19937 engine{seed};
        const auto ids = best_network->generate(corpus.encode(prompt), sample_length, 0.5, engine);
        std::cout << "\nsample from the " << best->label << " (best test loss in part one) after "
                  << "the prompt \"ROMEO:\", temperature 0.5, seeded:\n    "
                  << show(prompt + corpus.decode(ids)) << "\n";
    }

    // ---------- Part three: the delayed-copy stream ----------
    std::cout << "\npart three: a delayed-copy stream, " << copy_hidden << " units, window "
              << copy_window << ", Adam at " << copy_learning_rate << ", " << copy_epochs
              << " epochs\n"
              << "  every other character copies the random bit written `delay` pairs earlier, "
                 "which entered 2 * delay steps before;\n"
              << "  the floor is ln 2 / 2 = " << std::log(2.0) / 2.0
              << " (the random bits cost ln 2, the copies nothing) and chance is ln 2 = "
              << std::log(2.0) << "\n\n"
              << "  delay   steps back      elman       LSTM        GRU     gradient reach of each "
                 "trained network at the source bit's lag\n";
    struct CopyRow {
        std::size_t delay;
        double elman;
        double lstm;
        double gru;
        // Gradient reach of each trained network on its own stream, at the lag of the copy.
        double elman_reach;
        double lstm_reach;
        double gru_reach;
    };
    std::vector<CopyRow> copy_rows;
    for (const std::size_t delay : {3, 6, 10}) {
        const auto ids = delayed_copy_stream(delay);
        const std::size_t steps_back = 2 * delay;
        CopyRow row{delay, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (const auto cell : {ml_scratch::RecurrentCell::elman, ml_scratch::RecurrentCell::lstm,
                                ml_scratch::RecurrentCell::gru}) {
            ml_scratch::CharRnn network{2, copy_hidden, seed, cell};
            ml_scratch::RnnTrainingConfig config;
            config.learning_rate = copy_learning_rate;
            config.epochs = copy_epochs;
            config.sequence_length = copy_window;
            config.batch_size = 8;
            static_cast<void>(network.fit(ids, config));
            const double loss = network.loss(ids);
            // Reach on the same stream: the last prediction of each window is a copy, so its
            // gradient's norm at the lag of the source bit is what learning the copy needed.
            std::vector<double> reach_total(copy_window, 0.0);
            constexpr std::size_t windows = 20;
            for (std::size_t window = 0; window < windows; ++window) {
                // Windows end on a copy position: odd indices are copies.
                const std::size_t end = 2 * (100 + 200 * window) + 1;
                const ml_scratch::TokenSequence slice(ids.begin() + static_cast<std::ptrdiff_t>(end + 1 - copy_window - 1),
                                                      ids.begin() + static_cast<std::ptrdiff_t>(end + 1));
                const auto reach = network.gradient_reach(slice);
                for (std::size_t lag = 0; lag < copy_window; ++lag) {
                    reach_total[lag] += reach[lag] / reach[0] / static_cast<double>(windows);
                }
            }
            const double reach_at_source = reach_total[steps_back];
            switch (cell) {
            case ml_scratch::RecurrentCell::elman:
                row.elman = loss;
                row.elman_reach = reach_at_source;
                break;
            case ml_scratch::RecurrentCell::lstm:
                row.lstm = loss;
                row.lstm_reach = reach_at_source;
                break;
            case ml_scratch::RecurrentCell::gru:
                row.gru = loss;
                row.gru_reach = reach_at_source;
                break;
            }
        }
        copy_rows.push_back(row);
        std::cout << "  " << std::setw(5) << delay << std::setw(13) << steps_back << std::setw(11)
                  << row.elman << std::setw(11) << row.lstm << std::setw(11) << row.gru
                  << "     reach at " << std::setw(2) << steps_back << " steps: " << std::scientific
                  << std::setprecision(2) << row.elman_reach << ' ' << row.lstm_reach << ' '
                  << row.gru_reach << std::defaultfloat << std::fixed << std::setprecision(4)
                  << '\n';
    }

    // ---------- Summary and criteria ----------
    const double floor = std::log(2.0) / 2.0;
    std::cout << "\nsummary\n"
              << "  equal parameters, test: elman " << elman_run.test_loss << ", LSTM "
              << lstm_run.test_loss << ", GRU " << gru_run.test_loss << '\n'
              << "  " << wide_hidden << " units at epoch " << wide_epochs << ", validation: elman "
              << elman_run.validation_loss[wide_epochs - 1] << ", LSTM "
              << wide_lstm.validation_loss.back() << ", GRU " << wide_gru.validation_loss.back()
              << '\n'
              << "  reach at lag 20: elman " << std::scientific << std::setprecision(2)
              << elman_reach[20] << ", LSTM " << lstm_reach[20] << ", GRU " << gru_reach[20]
              << std::defaultfloat << std::fixed << std::setprecision(4) << '\n'
              << "  delayed copy at " << 2 * copy_rows[1].delay << " steps: elman "
              << copy_rows[1].elman << ", LSTM " << copy_rows[1].lstm << ", GRU "
              << copy_rows[1].gru << " (floor " << floor << "); at "
              << 2 * copy_rows[2].delay << " steps: elman " << copy_rows[2].elman
              << ", LSTM " << copy_rows[2].lstm << ", GRU " << copy_rows[2].gru << '\n'
              << "  reach at the source bit, " << 2 * copy_rows[2].delay
              << " steps back: elman " << std::scientific << std::setprecision(2)
              << copy_rows[2].elman_reach << ", LSTM " << copy_rows[2].lstm_reach << ", GRU "
              << copy_rows[2].gru_reach << std::defaultfloat << std::fixed << std::setprecision(4)
              << '\n';

    // Reach is a property of the trained network on its data, not of the cell alone. On text the
    // GRU carries gradient an order of magnitude further than the elman cell; on the copy stream
    // at twenty steps the LSTM, which learned the copy, carries gradient to the source bit an
    // order of magnitude better than the elman cell, which did not.
    const bool gru_reaches_further_on_text = gru_reach[20] > 10.0 * elman_reach[20];
    const bool lstm_reaches_the_source_bit = copy_rows[2].lstm_reach > 10.0 * copy_rows[2].elman_reach;
    // On the copy stream the dependency at twelve steps is one the elman cell cannot learn at
    // this budget and both gated cells can; at six steps every cell can; at twenty the LSTM
    // still can and the elman cell still cannot.
    const bool short_delay_learned_by_all = copy_rows[0].elman < floor + 0.05 &&
                                            copy_rows[0].lstm < floor + 0.05 &&
                                            copy_rows[0].gru < floor + 0.05;
    const bool medium_delay_separates_the_cells =
        copy_rows[1].elman > std::log(2.0) - 0.05 && copy_rows[1].lstm < floor + 0.05 &&
        copy_rows[1].gru < floor + 0.05;
    const bool long_delay_needs_the_lstm =
        copy_rows[2].elman > std::log(2.0) - 0.05 && copy_rows[2].lstm < floor + 0.05;
    // At equal parameters the GRU beats the elman cell on test and the LSTM does not; at equal
    // width both gated cells beat it. Both are measured claims the run could falsify.
    const bool gru_wins_at_equal_parameters = gru_run.test_loss < elman_run.test_loss;
    const bool lstm_loses_at_equal_parameters = lstm_run.test_loss > elman_run.test_loss;
    const bool gates_win_at_equal_width =
        wide_lstm.validation_loss.back() < elman_run.validation_loss[wide_epochs - 1] &&
        wide_gru.validation_loss.back() < elman_run.validation_loss[wide_epochs - 1];
    const bool elman_matches_experiment_15 =
        std::abs(elman_run.validation_loss.back() - 1.9324) < 5e-4;

    if (!gru_reaches_further_on_text || !lstm_reaches_the_source_bit ||
        !short_delay_learned_by_all || !medium_delay_separates_the_cells ||
        !long_delay_needs_the_lstm || !gru_wins_at_equal_parameters ||
        !lstm_loses_at_equal_parameters || !gates_win_at_equal_width ||
        !elman_matches_experiment_15) {
        std::cerr << "experiment failed its criteria: test " << elman_run.test_loss << '/'
                  << lstm_run.test_loss << '/' << gru_run.test_loss << ", wide "
                  << wide_lstm.validation_loss.back() << '/' << wide_gru.validation_loss.back()
                  << ", reach " << elman_reach[20] << '/' << lstm_reach[20] << '/'
                  << gru_reach[20] << ", copy";
        for (const CopyRow& row : copy_rows) {
            std::cerr << ' ' << row.elman << '/' << row.lstm << '/' << row.gru;
        }
        std::cerr << '\n';
        return 1;
    }
    return 0;
}
