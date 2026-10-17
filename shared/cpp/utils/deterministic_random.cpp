#include "ml_scratch/deterministic_random.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ml_scratch {
namespace {

constexpr double two_pi = 6.283185307179586476925286766559;
// 2^53, the number of representable doubles in [0, 1) with a 53-bit mantissa.
constexpr double mantissa_scale = 9007199254740992.0;

} // namespace

std::uint64_t DeterministicRandom::next() noexcept {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

double DeterministicRandom::uniform() noexcept {
    return static_cast<double>(next() >> 11) / mantissa_scale;
}

double DeterministicRandom::uniform(const double low, const double high) noexcept {
    return low + (high - low) * uniform();
}

double DeterministicRandom::gaussian() noexcept {
    // Box-Muller consumes two uniforms and returns one of the two normal deviates. The first
    // draw is clamped away from zero because log(0) is undefined.
    const double first = std::max(uniform(), 1e-12);
    const double second = uniform();
    return std::sqrt(-2.0 * std::log(first)) * std::cos(two_pi * second);
}

std::size_t DeterministicRandom::index(const std::size_t bound) {
    if (bound == 0) {
        throw std::invalid_argument("bound must be positive");
    }
    return static_cast<std::size_t>(uniform() * static_cast<double>(bound)) % bound;
}

bool DeterministicRandom::bernoulli(const double probability) {
    if (!(probability >= 0.0 && probability <= 1.0)) {
        throw std::invalid_argument("probability must be between zero and one");
    }
    return uniform() < probability;
}

} // namespace ml_scratch
