#include "ml_scratch/benchmark.hpp"
#include "ml_scratch/convolution.hpp"
#include "ml_scratch/deterministic_random.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/reinforcement.hpp"
#include "ml_scratch/rnn.hpp"
#include "ml_scratch/snake.hpp"
#include "ml_scratch/softmax_regression.hpp"
#include "ml_scratch/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260922;

std::string data_path(const char* variable, const std::string& fallback) {
    if (const char* override_path = std::getenv(variable)) {
        return override_path;
    }
    return fallback;
}

// A deterministic feature matrix, so the benchmarked work does not depend on a downloaded file.
ml_scratch::FeatureMatrix synthetic_features(const std::size_t samples, const std::size_t width,
                                             const std::uint64_t stream) {
    ml_scratch::DeterministicRandom random{stream};
    ml_scratch::FeatureMatrix data(samples, std::vector<double>(width, 0.0));
    for (auto& row : data) {
        for (double& value : row) {
            value = random.uniform(-1.0, 1.0);
        }
    }
    return data;
}

ml_scratch::LabeledDataset synthetic_labeled(const std::size_t samples, const std::size_t width,
                                             const std::size_t classes,
                                             const std::uint64_t stream) {
    ml_scratch::DeterministicRandom random{stream};
    ml_scratch::LabeledDataset data;
    data.reserve(samples);
    for (std::size_t index = 0; index < samples; ++index) {
        std::vector<double> features(width);
        for (double& value : features) {
            value = random.uniform(0.0, 1.0);
        }
        data.push_back({std::move(features), random.index(classes)});
    }
    return data;
}

ml_scratch::TokenSequence synthetic_text(const std::size_t length, const std::size_t vocabulary) {
    ml_scratch::DeterministicRandom random{7};
    ml_scratch::TokenSequence ids;
    ids.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
        ids.push_back(random.index(vocabulary));
    }
    return ids;
}

void print_header() {
    std::cout << "  benchmark                              per unit   units/s    cold/steady   "
                 "spread   iters   reps\n";
}

// Times per unit are printed in whatever scale keeps them readable.
std::string scaled(const double seconds) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(2);
    if (seconds >= 1e-3) {
        text << seconds * 1e3 << " ms";
    } else if (seconds >= 1e-6) {
        text << seconds * 1e6 << " us";
    } else {
        text << seconds * 1e9 << " ns";
    }
    return text.str();
}

std::string rate(const double per_second) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(per_second >= 100.0 ? 0 : 1);
    if (per_second >= 1e6) {
        text << per_second / 1e6 << "M";
    } else if (per_second >= 1e3) {
        text << per_second / 1e3 << "k";
    } else {
        text << per_second;
    }
    return text.str();
}

void print_result(const ml_scratch::BenchmarkResult& result, const double clock_resolution) {
    std::string padded = result.name;
    padded.resize(38, ' ');
    const double per_unit = result.work_per_iteration > 0.0
                                ? result.median() / result.work_per_iteration
                                : result.median();
    std::string time = scaled(per_unit);
    time.resize(10, ' ');
    std::string units = rate(result.work_per_second());
    units.resize(10, ' ');
    // A body faster than the clock can resolve has no measurable first call: one call is one
    // call, and it cannot be averaged over anything.
    std::ostringstream cold;
    if (result.cold_seconds < clock_resolution) {
        cold << "below res";
    } else {
        cold << std::fixed << std::setprecision(1) << result.cold_ratio() << "x";
    }
    std::cout << "  " << padded << time << units << std::setw(13) << cold.str()
              << std::setw(9) << std::setprecision(1)
              << 100.0 * result.relative_spread() << "%" << std::setw(8) << result.iterations
              << std::setw(7) << result.repetitions << '\n';
}

} // namespace

int main() {
    const ml_scratch::Environment environment = ml_scratch::current_environment();
    const ml_scratch::ClockCharacteristics clock = ml_scratch::measure_clock();

    std::cout << std::fixed << std::setprecision(2)
              << "C++ implementation benchmarks (seed=" << seed << ")\n\n"
              << "  " << environment.compiler << " " << environment.compiler_version << ", "
              << environment.language_standard << ", " << environment.optimization << ", "
              << environment.platform << " " << environment.architecture << ", "
              << environment.pointer_bits << "-bit, " << environment.date << '\n'
              << "  steady_clock resolution " << scaled(clock.resolution_seconds)
              << ", cost of one reading " << scaled(clock.overhead_seconds) << '\n'
              << "  every figure is the median over repetitions of the mean over iterations, "
                 "after a warm-up;\n"
              << "  cold/steady is the very first call divided by that median, and spread is the "
                 "standard deviation\n"
              << "  across repetitions as a fraction of their mean\n\n";

    std::vector<ml_scratch::BenchmarkResult> results;
    const auto record = [&results, &clock](ml_scratch::BenchmarkResult result) {
        print_result(result, clock.resolution_seconds);
        results.push_back(std::move(result));
        return &results.back();
    };

    // ---------- Inference ----------
    std::cout << "inference: one sample through a trained-shape model\n";
    print_header();

    ml_scratch::FeedForwardNetwork mlp{{{784, 128, ml_scratch::Activation::rectified_linear},
                                        {128, 10, ml_scratch::Activation::identity}},
                                       ml_scratch::Loss::softmax_cross_entropy,
                                       seed};
    const std::vector<double> image = synthetic_features(1, 784, 1).front();
    record(ml_scratch::benchmark("MLP 784-128-10 forward",
                                 [&] { return mlp.forward(image); }));

    // The same computation with its scratch space reused instead of allocated per call, which
    // isolates what the per-call allocation costs. At this size the answer turns out to be
    // nothing measurable, and the row is kept because that is worth knowing.
    ml_scratch::FeedForwardNetwork::Workspace workspace;
    record(ml_scratch::benchmark("MLP forward, workspace reused",
                                 [&] { return mlp.forward(image, workspace).front(); }));

    ml_scratch::SoftmaxRegression linear{784, 10};
    record(ml_scratch::benchmark("softmax regression 784-10 forward",
                                 [&] { return linear.predict(image); }));

    ml_scratch::ConvolutionalNetwork cnn{
        {1, 28, 28},
        {ml_scratch::convolution(8, 3, ml_scratch::Activation::rectified_linear, 1, 1),
         ml_scratch::max_pooling(2), ml_scratch::flatten(),
         ml_scratch::dense(10, ml_scratch::Activation::identity)},
        ml_scratch::Loss::softmax_cross_entropy,
        seed};
    record(ml_scratch::benchmark("CNN 8x3x3 + pool + dense forward",
                                 [&] { return cnn.forward(image); }));

    // ---------- Training ----------
    std::cout << "\ntraining: one gradient over a batch of 32\n";
    print_header();
    const auto batch = synthetic_labeled(32, 784, 10, 2);
    {
        auto result = ml_scratch::benchmark("MLP gradient, batch 32",
                                            [&] { return mlp.gradient(batch); });
        result.work_per_iteration = 32.0;
        result.work_unit = "samples";
        record(std::move(result));
    }
    {
        auto result = ml_scratch::benchmark("CNN gradient, batch 32",
                                            [&] { return cnn.gradient(batch); });
        result.work_per_iteration = 32.0;
        result.work_unit = "samples";
        record(std::move(result));
    }

    // ---------- Sequence models ----------
    std::cout << "\nsequence models: one 50-character window, forward and backward\n";
    print_header();
    constexpr std::size_t vocabulary = 64;
    constexpr std::size_t window = 50;
    const ml_scratch::TokenSequence text = synthetic_text(window + 1, vocabulary);
    for (const auto cell : {ml_scratch::RecurrentCell::elman, ml_scratch::RecurrentCell::lstm,
                            ml_scratch::RecurrentCell::gru}) {
        ml_scratch::CharRnn network{vocabulary, 128, seed, cell};
        auto result = ml_scratch::benchmark(
            std::string{ml_scratch::recurrent_cell_name(cell)} + " 128, BPTT over 50",
            [&] { return network.gradient(text); });
        result.work_per_iteration = window;
        result.work_unit = "characters";
        record(std::move(result));
    }
    {
        ml_scratch::CharTransformer transformer{{vocabulary, window, 64, 4, 2}, seed};
        auto result = ml_scratch::benchmark("transformer 2x4x64, context 50",
                                            [&] { return transformer.gradient(text); });
        result.work_per_iteration = window;
        result.work_unit = "characters";
        record(std::move(result));
    }

    // ---------- Environment and agents ----------
    std::cout << "\nreinforcement learning: one environment step, and one agent update\n";
    print_header();
    {
        ml_scratch::SnakeGame game{{10, 10}, 1};
        // The reset is outside the timed region, so what is measured is a step rather than the
        // cost of restarting an episode that the step ended.
        auto result = ml_scratch::benchmark(
            "snake step",
            [&] {
                if (game.finished()) {
                    game.reset();
                }
                const auto outcome = game.step(static_cast<ml_scratch::SnakeAction>(
                    game.steps() % ml_scratch::snake_action_count));
                return outcome.reward;
            },
            [&game] { game.reset(1); }, {});
        result.work_unit = "steps";
        record(std::move(result));
    }
    {
        ml_scratch::SnakeGame game{{10, 10}, 1};
        auto result = ml_scratch::benchmark("snake observation",
                                            [&] { return game.observation(); });
        result.work_unit = "observations";
        record(std::move(result));
    }
    {
        ml_scratch::TabularQLearning agent{};
        const ml_scratch::Transition sample{{}, 17, 1, 1.0, {}, 33, false};
        auto result = ml_scratch::benchmark("tabular Q update",
                                            [&] { return agent.learn(sample); });
        result.work_unit = "updates";
        record(std::move(result));
    }
    {
        ml_scratch::DeepQLearning::Config config;
        config.warmup = 64;
        config.replay_capacity = 4'000;
        config.seed = seed;
        ml_scratch::DeepQLearning agent{config};
        const std::vector<double> features(ml_scratch::SnakeGame::observation_size, 0.5);
        for (std::size_t index = 0; index < 256; ++index) {
            agent.observe({features, 0, index % 3, 0.5, features, 0, false});
        }
        auto result = ml_scratch::benchmark("DQN update, batch 32", [&] {
            return agent.observe({features, 0, 1, 0.5, features, 0, false});
        });
        result.work_per_iteration = 32.0;
        result.work_unit = "transitions";
        record(std::move(result));
    }

    // ---------- Where the time goes, and what it cost to write ----------
    std::cout << "\nimplementation size, as a rough stand-in for complexity: lines that are "
                 "neither blank nor a comment\n"
              << "  module                              header   source   total\n";
    const std::string root = data_path("ML_SCRATCH_SOURCE_ROOT", "src/cpp");
    struct Module {
        std::string name;
        std::string header;
        std::string source;
    };
    const std::vector<Module> modules{
        {"neural_network", "neural_network.hpp", "neural_network.cpp"},
        {"convolution", "convolution.hpp", "convolution.cpp"},
        {"rnn (three cells)", "rnn.hpp", "rnn.cpp"},
        {"transformer", "transformer.hpp", "transformer.cpp"},
        {"reinforcement", "reinforcement.hpp", "reinforcement.cpp"},
        {"snake", "snake.hpp", "snake.cpp"},
        {"benchmark", "benchmark.hpp", "benchmark.cpp"},
    };
    std::size_t total_code = 0;
    bool sources_readable = true;
    for (const Module& module : modules) {
        try {
            const auto header =
                ml_scratch::measure_source(root + "/include/ml_scratch/" + module.header);
            const auto source = ml_scratch::measure_source(root + "/src/" + module.source);
            std::string padded = module.name;
            padded.resize(36, ' ');
            std::cout << "  " << padded << std::setw(7) << header.code_lines << std::setw(9)
                      << source.code_lines << std::setw(8)
                      << header.code_lines + source.code_lines << '\n';
            total_code += header.code_lines + source.code_lines;
        } catch (const std::exception& error) {
            std::cout << "  (source sizes unavailable: " << error.what() << ")\n";
            sources_readable = false;
            break;
        }
    }
    if (sources_readable) {
        std::cout << "  the seven modules above come to " << total_code
                  << " lines of code, and every gradient in them is derived by hand\n";
    }

    // ---------- Output ----------
    const std::string output_directory =
        data_path("ML_SCRATCH_BENCHMARK_DIR", "results/benchmarks");
    const std::string output = output_directory + "/cpp.json";
    bool written = false;
    try {
        std::filesystem::create_directories(output_directory);
        ml_scratch::write_benchmark_json(output, environment, clock, results);
        written = true;
        std::cout << "\n  wrote " << output << ": every repetition, so the summary can be "
                  << "recomputed and another implementation compared against it\n";
    } catch (const std::exception& error) {
        std::cout << "\n  could not write " << output << ": " << error.what() << '\n';
    }
    if (const auto peak = ml_scratch::peak_resident_bytes()) {
        std::cout << "  peak resident memory over the whole run: " << *peak / (1024 * 1024)
                  << " MiB\n";
    } else {
        std::cout << "  peak resident memory: not available on this platform\n";
    }

    // ---------- Summary and criteria ----------
    const auto find = [&results](const std::string& name) -> const ml_scratch::BenchmarkResult& {
        const auto match = std::find_if(results.begin(), results.end(),
                                        [&name](const ml_scratch::BenchmarkResult& result) {
                                            return result.name == name;
                                        });
        if (match == results.end()) {
            throw std::logic_error("no benchmark named " + name);
        }
        return *match;
    };
    double worst_spread = 0.0;
    double largest_cold_ratio = 0.0;
    double shortest_repetition_ratio = std::numeric_limits<double>::infinity();
    std::string coldest;
    for (const auto& result : results) {
        worst_spread = std::max(worst_spread, result.relative_spread());
        shortest_repetition_ratio =
            std::min(shortest_repetition_ratio, result.median() *
                                                    static_cast<double>(result.iterations) /
                                                    clock.resolution_seconds);
        if (result.cold_ratio() > largest_cold_ratio) {
            largest_cold_ratio = result.cold_ratio();
            coldest = result.name;
        }
    }
    const auto& elman = find("elman 128, BPTT over 50");
    const auto& lstm = find("lstm 128, BPTT over 50");
    const auto& transformer = find("transformer 2x4x64, context 50");
    const auto& tabular = find("tabular Q update");
    const auto& deep = find("DQN update, batch 32");
    const auto& snake_step = find("snake step");
    const auto& allocating = find("MLP 784-128-10 forward");
    const auto& reusing = find("MLP forward, workspace reused");

    std::cout << "\nsummary\n"
              << "  the slowest first call was " << coldest << " at " << std::setprecision(1)
              << largest_cold_ratio << "x its steady-state median; the widest spread across "
              << "repetitions was " << 100.0 * worst_spread << "%, and the shortest timed "
              << "repetition was " << std::setprecision(0) << shortest_repetition_ratio
              << "x the clock's resolution\n"
              << std::setprecision(2) << "  per character of a 50-step window: elman "
              << scaled(elman.median() / 50.0) << ", LSTM " << scaled(lstm.median() / 50.0)
              << ", transformer " << scaled(transformer.median() / 50.0) << '\n'
              << "  one Q-learning update: table " << scaled(tabular.median()) << ", network "
              << scaled(deep.median()) << " for a batch of 32 — "
              << std::setprecision(0) << deep.median() / tabular.median() << "x"
              << std::setprecision(2) << '\n'
              << "  a snake step costs " << scaled(snake_step.median()) << ", so an episode of "
                 "150 steps is "
              << scaled(snake_step.median() * 150.0) << " of environment against "
              << scaled(deep.median() * 150.0) << " of learning\n"
              << "  the same MLP forward pass costs " << scaled(allocating.median())
              << " allocating its cache and " << scaled(reusing.median())
              << " reusing one, a difference of " << std::setprecision(1)
              << 100.0 * std::abs(allocating.median() - reusing.median()) / allocating.median()
              << "% against a spread of " << 100.0 * allocating.relative_spread() << "%"
              << std::setprecision(2) << '\n';

    // One iteration may well be faster than the clock can resolve — that is what the iteration
    // count is for. What has to hold is that each timed REPETITION is long compared with the
    // resolution, so the division by the iteration count is dividing a real duration.
    const bool above_the_clock = std::all_of(
        results.begin(), results.end(), [&clock](const ml_scratch::BenchmarkResult& result) {
            return result.median() * static_cast<double>(result.iterations) >
                   1000.0 * clock.resolution_seconds;
        });
    // A measurement whose repetitions disagree by more than a fifth is not a measurement.
    const bool repeatable = worst_spread < 0.2;
    // Warm-up is not ceremony: some first call in this set has to be markedly slower than its
    // steady state, or there would be nothing for it to hide.
    const bool warmup_earns_its_place = largest_cold_ratio > 3.0;
    // The orderings the earlier milestones' wall-clock columns implied, measured directly here.
    const bool gates_cost_more = lstm.median() > 2.0 * elman.median();
    const bool attention_costs_more = transformer.median() > 2.0 * elman.median();
    const bool table_beats_network = deep.median() > 100.0 * tabular.median();
    const bool environment_is_cheap = snake_step.median() < 0.05 * deep.median();
    // The two rows that differ only in whether the scratch space is reused have to agree: at
    // this size the allocation is below the noise floor, and a difference would mean one of them
    // is measuring something else.
    const bool allocation_is_negligible =
        std::abs(allocating.median() - reusing.median()) < 0.1 * allocating.median();

    if (!repeatable) {
        // Worth saying plainly, because it is the common failure and it is not a bug: a
        // benchmark run alongside a build or another benchmark measures the contention, and the
        // honest response is to refuse the numbers rather than publish them.
        std::cerr << "the repetitions of at least one benchmark disagree by "
                  << 100.0 * worst_spread
                  << "%, which means the machine was not idle; rerun with nothing else "
                     "running\n";
    }
    if (!above_the_clock || !repeatable || !warmup_earns_its_place || !gates_cost_more ||
        !attention_costs_more || !table_beats_network || !environment_is_cheap ||
        !allocation_is_negligible || !written || !sources_readable) {
        std::cerr << "experiment failed its criteria: worst spread " << worst_spread
                  << ", largest cold ratio " << largest_cold_ratio << ", elman "
                  << elman.median() << ", lstm " << lstm.median() << ", transformer "
                  << transformer.median() << ", tabular " << tabular.median() << ", dqn "
                  << deep.median() << ", step " << snake_step.median() << ", written " << written
                  << '\n';
        return 1;
    }
    return 0;
}
