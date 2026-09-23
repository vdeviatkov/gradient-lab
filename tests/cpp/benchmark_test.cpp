#include "ml_scratch/benchmark.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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

void test_statistics_by_hand() {
    ml_scratch::BenchmarkResult result;
    result.repetition_seconds = {4.0, 1.0, 3.0, 2.0, 10.0};
    require_near(result.best(), 1.0, 0.0, "best is the smallest repetition");
    require_near(result.worst(), 10.0, 0.0, "worst is the largest");
    require_near(result.median(), 3.0, 0.0, "the median ignores the outlier");
    require_near(result.mean(), 4.0, 1e-15, "the mean does not");
    // sd of {4,1,3,2,10} about 4 is sqrt((0+9+1+4+36)/5) = sqrt(10).
    require_near(result.relative_spread(), std::sqrt(10.0) / 4.0, 1e-12, "relative spread");

    result.repetition_seconds = {2.0, 4.0};
    require_near(result.median(), 3.0, 1e-15, "an even count averages the middle two");

    result.repetition_seconds = {0.5};
    require_near(result.relative_spread(), 0.0, 0.0, "one repetition has no spread");
    result.work_per_iteration = 100.0;
    require_near(result.work_per_second(), 200.0, 1e-12, "a rate is work over the median");
    result.cold_seconds = 5.0;
    require_near(result.cold_ratio(), 10.0, 1e-12, "the cold ratio compares the first call");

    const ml_scratch::BenchmarkResult empty;
    require(empty.median() == 0.0 && empty.best() == 0.0 && empty.work_per_second() == 0.0 &&
                empty.cold_ratio() == 0.0,
            "an empty result reports zeros rather than reading past its data");
}

// The harness has to measure a known duration correctly: a body that sleeps for a set time must
// come back at about that time per iteration.
void test_measures_a_known_duration() {
    constexpr auto nap = std::chrono::microseconds{300};
    ml_scratch::BenchmarkOptions options;
    options.iterations = 3;
    options.repetitions = 5;
    options.warmup_iterations = 1;
    const auto result = ml_scratch::benchmark(
        "sleep", [&] { std::this_thread::sleep_for(nap); }, options);
    require(result.iterations == 3 && result.repetitions == 5 &&
                result.repetition_seconds.size() == 5,
            "the configured counts are honoured");
    // Sleeping is only accurate from below, so the floor is the assertion and the ceiling is
    // loose enough to survive a busy machine.
    require(result.median() >= 0.0003 && result.median() < 0.01,
            "a 300 microsecond sleep measures at least 300 microseconds per iteration");
    require(result.name == "sleep", "the name is carried through");
}

void test_auto_tuning_and_warmup() {
    // A body far faster than the clock's resolution must be run many times per repetition,
    // otherwise the measurement is of the clock.
    std::size_t calls = 0;
    ml_scratch::BenchmarkOptions options;
    options.repetitions = 3;
    options.target_seconds = 0.01;
    const auto result = ml_scratch::benchmark(
        "increment", [&calls] { return ++calls; }, options);
    require(result.iterations > 1000,
            "a trivial body should be auto-tuned to many iterations per repetition");
    require(result.median() < 1e-6, "and its per-iteration time should be tiny");
    // Every iteration really ran: the cold call, the tuning rounds, the warm-up, and the
    // repetitions all call the body.
    require(calls > result.iterations * options.repetitions,
            "the body ran at least once per timed iteration, plus tuning and warm-up");

    // With an explicit warm-up of zero the harness still warms up by one iteration's worth;
    // with an explicit count it uses exactly that.
    std::size_t warm_calls = 0;
    ml_scratch::BenchmarkOptions explicit_options;
    explicit_options.iterations = 10;
    explicit_options.repetitions = 2;
    explicit_options.warmup_iterations = 5;
    const auto explicit_result = ml_scratch::benchmark(
        "counted", [&warm_calls] { return ++warm_calls; }, explicit_options);
    require(explicit_result.iterations == 10, "an explicit iteration count skips auto-tuning");
    // One cold call, five warm-up calls, then two repetitions of ten.
    require(warm_calls == 1 + 5 + 2 * 10, "exactly the configured calls were made");

    require_invalid_argument(
        [] {
            ml_scratch::BenchmarkOptions bad;
            bad.repetitions = 0;
            return ml_scratch::benchmark("bad", [] { return 1; }, bad);
        },
        "zero repetitions are rejected");
    require_invalid_argument(
        [] {
            ml_scratch::BenchmarkOptions bad;
            bad.target_seconds = 0.0;
            return ml_scratch::benchmark("bad", [] { return 1; }, bad);
        },
        "a zero target duration is rejected");
}

// The reset callback runs outside the timed region, so a benchmark can consume state it needs
// restored without paying for the restoration.
void test_reset_runs_outside_the_measurement() {
    std::size_t resets = 0;
    std::size_t body_calls = 0;
    ml_scratch::BenchmarkOptions options;
    options.iterations = 4;
    options.repetitions = 3;
    options.warmup_iterations = 2;
    const auto result = ml_scratch::benchmark(
        "with reset", [&body_calls] { return ++body_calls; }, [&resets] { ++resets; }, options);
    // One before the cold call, one before the warm-up, one before each repetition.
    require(resets == 1 + 1 + 3, "reset runs once per timed region and before the cold call");
    require(body_calls == 1 + 2 + 3 * 4, "the body ran the configured number of times");
    require(result.repetition_seconds.size() == 3, "one figure per repetition");
}

// do_not_optimize has to actually keep the work: a loop whose result is discarded can be deleted
// outright, and a benchmark of deleted code reports the empty loop.
void test_do_not_optimize_keeps_the_work() {
    ml_scratch::BenchmarkOptions options;
    options.iterations = 200;
    options.repetitions = 5;
    options.warmup_iterations = 10;

    // A body that does a measurable amount of arithmetic, with its result observed.
    const auto kept = ml_scratch::benchmark(
        "kept",
        [] {
            double total = 0.0;
            for (std::size_t index = 0; index < 4'000; ++index) {
                total += std::sqrt(static_cast<double>(index));
            }
            return total;
        },
        options);
    // An empty body, as the floor.
    const auto empty = ml_scratch::benchmark("empty", [] { return 0; }, options);
    require(kept.median() > 20.0 * empty.median(),
            "work whose result is observed is not optimized away");
    require(kept.median() > 1e-7, "and four thousand square roots take measurable time");
}

void test_clock_and_environment() {
    const auto clock = ml_scratch::measure_clock();
    require(clock.resolution_seconds > 0.0 && clock.resolution_seconds < 1e-3,
            "the steady clock resolves better than a millisecond");
    require(clock.overhead_seconds > 0.0 && clock.overhead_seconds < 1e-4,
            "and a reading costs less than a tenth of a millisecond");

    const auto environment = ml_scratch::current_environment();
    require(!environment.compiler.empty() && !environment.platform.empty() &&
                !environment.architecture.empty(),
            "the environment is populated");
    require(environment.pointer_bits == sizeof(void*) * 8, "the pointer width is the real one");
    require(environment.language_standard.rfind("C++", 0) == 0, "the standard is named");
    require(environment.date.size() == 10 && environment.date[4] == '-',
            "the date is an ISO day");

    // Memory reporting is optional, but when it is available it has to be plausible.
    if (const auto peak = ml_scratch::peak_resident_bytes()) {
        require(*peak > 100'000, "a running process holds at least a hundred kilobytes");
        require(*peak < 100'000'000'000ULL, "and not a hundred gigabytes");
    }
}

void test_source_measurement() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_source_test.cpp").string();
    {
        std::ofstream stream{path};
        stream << "// a comment\n"
                  "\n"
                  "int main() {\n"
                  "    /* a block\n"
                  "       comment */\n"
                  "    return 0; // trailing\n"
                  "}\n"
                  "   \n"
                  "/* one line block */\n";
    }
    const auto size = ml_scratch::measure_source(path);
    require(size.lines == 9, "every line is counted");
    // Code lines: `int main() {`, `return 0; // trailing`, `}`.
    require(size.code_lines == 3, "blank lines and comments are not code");
    std::filesystem::remove(path);

    bool threw = false;
    try {
        static_cast<void>(ml_scratch::measure_source(path));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "a missing file is reported");
}

void test_json_output() {
    ml_scratch::BenchmarkResult result;
    result.name = "a \"quoted\" name";
    result.work_unit = "characters";
    result.work_per_iteration = 64.0;
    result.iterations = 10;
    result.repetitions = 2;
    result.cold_seconds = 0.5;
    result.repetition_seconds = {0.25, 0.75};

    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_benchmark_test.json").string();
    ml_scratch::write_benchmark_json(path, ml_scratch::current_environment(),
                                     ml_scratch::measure_clock(), {result});
    std::ifstream stream{path};
    const std::string text{std::istreambuf_iterator<char>{stream},
                           std::istreambuf_iterator<char>{}};
    require(text.find("\"implementation\": \"cpp\"") != std::string::npos,
            "the implementation is named");
    require(text.find("a \\\"quoted\\\" name") != std::string::npos, "quotes are escaped");
    require(text.find("\"work_unit\": \"characters\"") != std::string::npos,
            "the unit is recorded");
    require(text.find("\"median_seconds\": 0.5") != std::string::npos,
            "the median of 0.25 and 0.75 is written");
    require(text.find("\"work_per_second\": 128") != std::string::npos,
            "64 units at half a second each is 128 a second");
    require(text.find("\"repetition_seconds\": [0.25, 0.75]") != std::string::npos,
            "every repetition is kept, so the summary can be recomputed");
    require(text.find("\"compiler\"") != std::string::npos &&
                text.find("\"resolution_seconds\"") != std::string::npos,
            "the environment and clock travel with the results");
    // Braces balance, which is the cheapest check that the document is well formed.
    require(std::count(text.begin(), text.end(), '{') == std::count(text.begin(), text.end(), '}'),
            "the JSON braces balance");
    require(std::count(text.begin(), text.end(), '[') == std::count(text.begin(), text.end(), ']'),
            "the JSON brackets balance");
    std::filesystem::remove(path);
}

} // namespace

int main() {
    try {
        test_statistics_by_hand();
        test_measures_a_known_duration();
        test_auto_tuning_and_warmup();
        test_reset_runs_outside_the_measurement();
        test_do_not_optimize_keeps_the_work();
        test_clock_and_environment();
        test_source_measurement();
        test_json_output();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
