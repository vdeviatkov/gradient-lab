#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ml_scratch {

// Keeps the optimizer from deleting the work a benchmark exists to measure. A benchmark body
// whose result is never read is dead code, and a compiler is entitled to remove it; passing the
// result here makes it observable without costing a store to memory.
template <typename T>
inline void do_not_optimize(T&& value) {
#if defined(__clang__) || defined(__GNUC__)
    asm volatile("" : : "r,m"(value) : "memory");
#else
    // No portable equivalent: a volatile write is the fallback, which does cost a store.
    static volatile char sink = 0;
    sink = *reinterpret_cast<const volatile char*>(&value);
    static_cast<void>(sink);
#endif
}

struct BenchmarkOptions {
    // Iterations per repetition. Zero auto-tunes until one repetition takes at least
    // `target_seconds`, so a fast operation is measured over enough work to rise above the
    // clock's resolution and a slow one is not run needlessly often.
    std::size_t iterations{0};
    std::size_t repetitions{7};
    double target_seconds{0.05};
    // Iterations run and discarded before the first timed repetition. What they pay for is the
    // first-call cost — page faults on freshly allocated buffers, cold instruction and data
    // caches, and the branch predictor's ignorance — which is real but is not what a steady-state
    // figure is meant to report.
    std::size_t warmup_iterations{0}; // zero means one iteration's worth of `target_seconds`

    bool operator==(const BenchmarkOptions&) const = default;
};

struct BenchmarkResult {
    std::string name;
    // What one iteration processes, for a rate: characters, samples, steps. Zero means the
    // iteration itself is the unit.
    double work_per_iteration{1.0};
    std::string work_unit{"iteration"};

    std::size_t iterations{0};
    std::size_t repetitions{0};
    // The very first call, on a completely cold state and before any warm-up. Reported next to
    // the steady-state figure because the gap between them is the measurement warm-up hides.
    double cold_seconds{0.0};
    // Mean seconds per iteration within each timed repetition.
    std::vector<double> repetition_seconds;

    [[nodiscard]] double best() const;
    [[nodiscard]] double median() const;
    [[nodiscard]] double mean() const;
    [[nodiscard]] double worst() const;
    // Standard deviation over the repetitions divided by the mean: how repeatable the figure is.
    // A benchmark whose spread is a large fraction of its value is not measuring what it thinks.
    [[nodiscard]] double relative_spread() const;
    [[nodiscard]] double work_per_second() const;
    // How many times slower the first call was than the steady-state median.
    [[nodiscard]] double cold_ratio() const;
};

// Runs `body` under the options above. `body` is called with no arguments; whatever it returns is
// passed through do_not_optimize. `reset`, when supplied, is called before the cold measurement
// and before every repetition, outside the timed region, so state a benchmark consumes can be
// restored without being charged for it.
template <typename Body>
[[nodiscard]] BenchmarkResult benchmark(std::string name, Body&& body,
                                        BenchmarkOptions options = {});
template <typename Body, typename Reset>
[[nodiscard]] BenchmarkResult benchmark(std::string name, Body&& body, Reset&& reset,
                                        BenchmarkOptions options);

// Resolution and call overhead of the clock the harness uses, measured rather than assumed. Any
// benchmark whose per-iteration time is close to these is measuring the clock.
struct ClockCharacteristics {
    // Smallest non-zero difference between two consecutive readings.
    double resolution_seconds;
    // Mean cost of taking one reading.
    double overhead_seconds;
};

[[nodiscard]] ClockCharacteristics measure_clock();

// Everything about the machine and build that a number here depends on. Recorded with every
// result set, because a benchmark without it is not reproducible.
struct Environment {
    std::string compiler;
    std::string compiler_version;
    std::string language_standard;
    std::string optimization; // what the build system said, via ML_SCRATCH_BUILD_TYPE
    std::string platform;
    std::string architecture;
    std::size_t pointer_bits;
    // Wall-clock date of the run, from the system clock, as YYYY-MM-DD.
    std::string date;
};

[[nodiscard]] Environment current_environment();

// Peak resident set size in bytes, where the platform offers it. Empty otherwise, which is
// reported as such rather than guessed at.
[[nodiscard]] std::optional<std::size_t> peak_resident_bytes();

// Source size of one file, as a stand-in for implementation complexity: total lines, and lines
// that are neither blank nor a comment. Counting is deliberately crude — it does not parse C++ —
// and the experiment says so, because the number is a rough comparison between implementations
// of the same algorithm and nothing more.
struct SourceSize {
    std::size_t lines;
    std::size_t code_lines;
};

[[nodiscard]] SourceSize measure_source(const std::string& path);

// Writes a result set as JSON, so the same measurements can be compared against the other
// language implementations when they exist.
void write_benchmark_json(const std::string& path, const Environment& environment,
                          const ClockCharacteristics& clock,
                          const std::vector<BenchmarkResult>& results);

// ---------------------------------------------------------------------------------------------

namespace detail {

struct NoReset {
    void operator()() const noexcept {}
};

BenchmarkResult run_benchmark(std::string name, const BenchmarkOptions& options,
                              void* body_context, void (*run_once)(void*),
                              void* reset_context, void (*reset)(void*));

} // namespace detail

template <typename Body, typename Reset>
BenchmarkResult benchmark(std::string name, Body&& body, Reset&& reset,
                          BenchmarkOptions options) {
    auto run_once = [](void* context) {
        auto& callable = *static_cast<std::remove_reference_t<Body>*>(context);
        if constexpr (std::is_void_v<decltype(callable())>) {
            callable();
        } else {
            do_not_optimize(callable());
        }
    };
    auto run_reset = [](void* context) {
        (*static_cast<std::remove_reference_t<Reset>*>(context))();
    };
    return detail::run_benchmark(std::move(name), options, &body, run_once, &reset, run_reset);
}

template <typename Body>
BenchmarkResult benchmark(std::string name, Body&& body, BenchmarkOptions options) {
    detail::NoReset reset;
    return benchmark(std::move(name), std::forward<Body>(body), reset, options);
}

} // namespace ml_scratch
