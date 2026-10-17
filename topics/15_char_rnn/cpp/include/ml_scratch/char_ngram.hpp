#pragma once

#include "ml_scratch/text_corpus.hpp"

namespace ml_scratch {

// A count-based n-gram baseline: the next character's probability given the previous order - 1
// characters, estimated from counts with additive smoothing. Order 1 is the unigram model, which
// ignores context altogether; order 2 is the bigram model. Anything that claims to model text has
// to beat these.
class CharNgram {
  public:
    // `alpha` is the pseudo-count added to every possible next character, so that a context or
    // continuation never seen in training still has a finite cross-entropy.
    CharNgram(std::size_t order, std::size_t vocabulary_size, double alpha = 1.0);

    void fit(const TokenSequence& ids);
    // P(next | the preceding order - 1 ids of `context`). Shorter contexts are padded as if the
    // text began with id 0, which is what fit does at the start of its sequence.
    [[nodiscard]] double probability(const TokenSequence& context, std::size_t next) const;
    // Mean negative log probability per predicted character, in nats, over ids[1..].
    [[nodiscard]] double cross_entropy(const TokenSequence& ids) const;

    [[nodiscard]] std::size_t order() const noexcept { return order_; }

  private:
    [[nodiscard]] std::size_t context_index(const TokenSequence& ids, std::size_t position) const;

    std::size_t order_;
    std::size_t vocabulary_size_;
    double alpha_;
    // counts_[context][next] and totals_[context], with the context flattened in base
    // vocabulary_size.
    std::vector<std::vector<double>> counts_;
    std::vector<double> totals_;
};

} // namespace ml_scratch
