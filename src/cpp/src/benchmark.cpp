#include "ml_scratch/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <ostream>
#include <string_view>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

#if __has_include(<sys/resource.h>)
#include <sys/resource.h>
#define ML_SCRATCH_HAS_RUSAGE 1
#endif

namespace ml_scratch {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(const Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

double BenchmarkResult::best() const {
    return repetition_seconds.empty()
               ? 0.0
               : *std::min_element(repetition_seconds.begin(), repetition_seconds.end());
}

double BenchmarkResult::worst() const {
    return repetition_seconds.empty()
               ? 0.0
               : *std::max_element(repetition_seconds.begin(), repetition_seconds.end());
}

double BenchmarkResult::median() const {
    if (repetition_seconds.empty()) {
        return 0.0;
    }
    // The median rather than the mean: one repetition interrupted by the operating system moves
    // a mean and leaves a median alone, and an interrupted repetition is noise rather than a
    // property of the code.
    std::vector<double> sorted = repetition_seconds;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t middle = sorted.size() / 2;
    return sorted.size() % 2 == 1 ? sorted[middle]
                                  : 0.5 * (sorted[middle - 1] + sorted[middle]);
}

double BenchmarkResult::mean() const {
    if (repetition_seconds.empty()) {
        return 0.0;
    }
    return std::accumulate(repetition_seconds.begin(), repetition_seconds.end(), 0.0) /
           static_cast<double>(repetition_seconds.size());
}

double BenchmarkResult::relative_spread() const {
    if (repetition_seconds.size() < 2) {
        return 0.0;
    }
    const double average = mean();
    if (average <= 0.0) {
        return 0.0;
    }
    double total = 0.0;
    for (const double value : repetition_seconds) {
        const double difference = value - average;
        total += difference * difference;
    }
    return std::sqrt(total / static_cast<double>(repetition_seconds.size())) / average;
}

double BenchmarkResult::work_per_second() const {
    const double per_iteration = median();
    return per_iteration > 0.0 ? work_per_iteration / per_iteration : 0.0;
}

double BenchmarkResult::cold_ratio() const {
    const double steady = median();
    return steady > 0.0 ? cold_seconds / steady : 0.0;
}

namespace detail {

BenchmarkResult run_benchmark(std::string name, const BenchmarkOptions& options,
                              void* body_context, void (*run_once)(void*), void* reset_context,
                              void (*reset)(void*)) {
    if (options.repetitions == 0) {
        throw std::invalid_argument("a benchmark needs at least one repetition");
    }
    if (!std::isfinite(options.target_seconds) || options.target_seconds <= 0.0) {
        throw std::invalid_argument("target_seconds must be finite and positive");
    }

    BenchmarkResult result;
    result.name = std::move(name);
    result.repetitions = options.repetitions;

    // The cold measurement comes first, before anything has touched the code or the data, which
    // is the only moment at which it can be taken.
    reset(reset_context);
    const auto cold_start = Clock::now();
    run_once(body_context);
    result.cold_seconds = seconds_since(cold_start);

    // Auto-tuning: double the iteration count until one batch reaches the target duration, so
    // the measurement is long compared with the clock's resolution whatever the body costs.
    std::size_t iterations = options.iterations;
    if (iterations == 0) {
        iterations = 1;
        for (;;) {
            reset(reset_context);
            const auto start = Clock::now();
            for (std::size_t index = 0; index < iterations; ++index) {
                run_once(body_context);
            }
            const double elapsed = seconds_since(start);
            if (elapsed >= options.target_seconds || iterations >= (1ULL << 30)) {
                break;
            }
            // Jump straight to an estimate of the right count rather than doubling blindly,
            // with a floor so a body faster than the clock's resolution still makes progress.
            const double factor = elapsed > 0.0 ? options.target_seconds / elapsed : 64.0;
            iterations = std::max(iterations + 1,
                                  static_cast<std::size_t>(static_cast<double>(iterations) *
                                                           std::min(factor * 1.2, 64.0)));
        }
    }
    result.iterations = iterations;

    std::size_t warmup = options.warmup_iterations;
    if (options.warmup_iterations == 0) {
        warmup = iterations;
    }
    reset(reset_context);
    for (std::size_t index = 0; index < warmup; ++index) {
        run_once(body_context);
    }

    result.repetition_seconds.reserve(options.repetitions);
    for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
        reset(reset_context);
        const auto start = Clock::now();
        for (std::size_t index = 0; index < iterations; ++index) {
            run_once(body_context);
        }
        result.repetition_seconds.push_back(seconds_since(start) /
                                            static_cast<double>(iterations));
    }
    return result;
}

} // namespace detail

ClockCharacteristics measure_clock() {
    // Resolution: the smallest non-zero gap between consecutive readings, found by reading in a
    // tight loop until the value changes.
    double resolution = std::numeric_limits<double>::infinity();
    for (std::size_t trial = 0; trial < 20; ++trial) {
        const auto first = Clock::now();
        Clock::time_point second = first;
        while (second == first) {
            second = Clock::now();
        }
        resolution = std::min(resolution, std::chrono::duration<double>(second - first).count());
    }

    // Overhead: the cost of a reading, measured over many of them so the figure is not itself
    // dominated by one reading's cost.
    constexpr std::size_t readings = 100'000;
    const auto start = Clock::now();
    for (std::size_t index = 0; index < readings; ++index) {
        do_not_optimize(Clock::now());
    }
    const double overhead = seconds_since(start) / static_cast<double>(readings);
    return {resolution, overhead};
}

Environment current_environment() {
    Environment environment;
#if defined(__clang__)
    environment.compiler = "clang";
    environment.compiler_version = std::to_string(__clang_major__) + "." +
                                   std::to_string(__clang_minor__) + "." +
                                   std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
    environment.compiler = "gcc";
    environment.compiler_version = std::to_string(__GNUC__) + "." +
                                   std::to_string(__GNUC_MINOR__) + "." +
                                   std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    environment.compiler = "msvc";
    environment.compiler_version = std::to_string(_MSC_VER);
#else
    environment.compiler = "unknown";
#endif

    environment.language_standard = "C++" + std::to_string(__cplusplus / 100 % 100);
    // The build type is not visible to the compiler, so it is passed in by the build system.
    if (const char* build = std::getenv("ML_SCRATCH_BUILD_TYPE")) {
        environment.optimization = build;
    } else {
#ifdef NDEBUG
        // NDEBUG is the one thing the source can see: it says assertions are off, which every
        // optimized configuration sets and a debug one does not.
        environment.optimization = "NDEBUG set, build type not reported";
#else
        environment.optimization = "NDEBUG unset, build type not reported";
#endif
    }

#if defined(__APPLE__)
    environment.platform = "macOS";
#elif defined(__linux__)
    environment.platform = "Linux";
#elif defined(_WIN32)
    environment.platform = "Windows";
#else
    environment.platform = "unknown";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    environment.architecture = "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    environment.architecture = "x86_64";
#else
    environment.architecture = "unknown";
#endif

    environment.pointer_bits = sizeof(void*) * 8;

    const std::time_t now = std::time(nullptr);
    std::tm parts{};
#if defined(_WIN32)
    localtime_s(&parts, &now);
#else
    localtime_r(&now, &parts);
#endif
    std::ostringstream date;
    date << std::put_time(&parts, "%Y-%m-%d");
    environment.date = date.str();
    return environment;
}

std::optional<std::size_t> peak_resident_bytes() {
#ifdef ML_SCRATCH_HAS_RUSAGE
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return std::nullopt;
    }
    // Linux reports kilobytes and the BSDs, macOS included, report bytes.
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
#else
    return std::nullopt;
#endif
}

SourceSize measure_source(const std::string& path) {
    std::ifstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open source file: " + path);
    }
    SourceSize size{0, 0};
    std::string line;
    bool in_block_comment = false;
    while (std::getline(stream, line)) {
        ++size.lines;
        const std::size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos) {
            continue; // blank
        }
        const std::string_view trimmed{line.data() + first, line.size() - first};
        if (in_block_comment) {
            if (trimmed.find("*/") != std::string_view::npos) {
                in_block_comment = false;
            }
            continue;
        }
        if (trimmed.starts_with("//")) {
            continue;
        }
        if (trimmed.starts_with("/*")) {
            if (trimmed.find("*/") == std::string_view::npos) {
                in_block_comment = true;
            }
            continue;
        }
        ++size.code_lines;
    }
    return size;
}

namespace {

void write_string(std::ostream& stream, const std::string& value) {
    stream << '"';
    for (const char character : value) {
        if (character == '"' || character == '\\') {
            stream << '\\' << character;
        } else if (character == '\n') {
            stream << "\\n";
        } else {
            stream << character;
        }
    }
    stream << '"';
}

} // namespace

void write_benchmark_json(const std::string& path, const Environment& environment,
                          const ClockCharacteristics& clock,
                          const std::vector<BenchmarkResult>& results) {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open benchmark output for writing: " + path);
    }
    stream << std::setprecision(12);
    stream << "{\n  \"implementation\": \"cpp\",\n  \"environment\": {\n";
    const auto field = [&stream](const char* key, const std::string& value, const bool last) {
        stream << "    \"" << key << "\": ";
        write_string(stream, value);
        stream << (last ? "\n" : ",\n");
    };
    field("compiler", environment.compiler, false);
    field("compiler_version", environment.compiler_version, false);
    field("language_standard", environment.language_standard, false);
    field("optimization", environment.optimization, false);
    field("platform", environment.platform, false);
    field("architecture", environment.architecture, false);
    field("date", environment.date, false);
    stream << "    \"pointer_bits\": " << environment.pointer_bits << "\n  },\n";
    stream << "  \"clock\": {\n    \"resolution_seconds\": " << clock.resolution_seconds
           << ",\n    \"overhead_seconds\": " << clock.overhead_seconds << "\n  },\n";
    if (const auto peak = peak_resident_bytes()) {
        stream << "  \"peak_resident_bytes\": " << *peak << ",\n";
    } else {
        stream << "  \"peak_resident_bytes\": null,\n";
    }
    stream << "  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const BenchmarkResult& result = results[index];
        stream << "    {\n      \"name\": ";
        write_string(stream, result.name);
        stream << ",\n      \"work_unit\": ";
        write_string(stream, result.work_unit);
        stream << ",\n      \"work_per_iteration\": " << result.work_per_iteration
               << ",\n      \"iterations\": " << result.iterations
               << ",\n      \"repetitions\": " << result.repetitions
               << ",\n      \"cold_seconds\": " << result.cold_seconds
               << ",\n      \"median_seconds\": " << result.median()
               << ",\n      \"best_seconds\": " << result.best()
               << ",\n      \"worst_seconds\": " << result.worst()
               << ",\n      \"relative_spread\": " << result.relative_spread()
               << ",\n      \"work_per_second\": " << result.work_per_second()
               << ",\n      \"repetition_seconds\": [";
        for (std::size_t sample = 0; sample < result.repetition_seconds.size(); ++sample) {
            stream << (sample == 0 ? "" : ", ") << result.repetition_seconds[sample];
        }
        stream << "]\n    }" << (index + 1 == results.size() ? "\n" : ",\n");
    }
    stream << "  ]\n}\n";
    if (!stream) {
        throw std::runtime_error("failed while writing benchmark output: " + path);
    }
}

} // namespace ml_scratch
