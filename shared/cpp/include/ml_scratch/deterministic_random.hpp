#pragma once

#include <cstddef>
#include <cstdint>

namespace ml_scratch {

// A self-contained splitmix64 stream. Experiments use it instead of the standard distributions,
// whose implementations are not specified bit-for-bit, so a seeded dataset is identical on every
// standard library. It is a data-generation utility, not a source of model randomness: the models
// themselves seed their own local engines.
class DeterministicRandom {
  public:
    explicit DeterministicRandom(std::uint64_t seed) noexcept : state_(seed) {}

    // Uniform in [0, 1).
    [[nodiscard]] double uniform() noexcept;
    // Uniform in [low, high).
    [[nodiscard]] double uniform(double low, double high) noexcept;
    // Standard normal deviate via the Box-Muller transform.
    [[nodiscard]] double gaussian() noexcept;
    // Uniform integer in [0, bound). Throws std::invalid_argument when bound is zero.
    [[nodiscard]] std::size_t index(std::size_t bound);
    // True with probability `probability`. Throws std::invalid_argument outside [0, 1].
    [[nodiscard]] bool bernoulli(double probability);

  private:
    [[nodiscard]] std::uint64_t next() noexcept;

    std::uint64_t state_;
};

} // namespace ml_scratch
