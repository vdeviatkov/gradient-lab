#include "ml_scratch/char_ngram.hpp"
#include "ml_scratch/text_corpus.hpp"

#include <cmath>
#include <stdexcept>

namespace ml_scratch {

CharNgram::CharNgram(const std::size_t order, const std::size_t vocabulary_size, const double alpha)
    : order_(order), vocabulary_size_(vocabulary_size), alpha_(alpha) {
    if (order == 0) {
        throw std::invalid_argument("n-gram order must be positive");
    }
    if (vocabulary_size == 0) {
        throw std::invalid_argument("vocabulary must be non-empty");
    }
    if (!std::isfinite(alpha) || alpha <= 0.0) {
        throw std::invalid_argument("the smoothing pseudo-count must be positive");
    }
    std::size_t contexts = 1;
    for (std::size_t index = 1; index < order; ++index) {
        if (contexts > 1'000'000 / vocabulary_size) {
            throw std::invalid_argument("n-gram order is too large for a dense count table");
        }
        contexts *= vocabulary_size;
    }
    counts_.assign(contexts, std::vector<double>(vocabulary_size, 0.0));
    totals_.assign(contexts, 0.0);
}

std::size_t CharNgram::context_index(const TokenSequence& ids, const std::size_t position) const {
    // The order - 1 ids ending just before `position`, flattened most-recent-last in base
    // vocabulary_size; positions before the start of the sequence read as id 0.
    std::size_t index = 0;
    for (std::size_t back = order_ - 1; back >= 1; --back) {
        const std::size_t id = position >= back ? ids[position - back] : 0;
        if (id >= vocabulary_size_) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
        index = index * vocabulary_size_ + id;
    }
    return index;
}

void CharNgram::fit(const TokenSequence& ids) {
    for (std::size_t position = 0; position < ids.size(); ++position) {
        if (ids[position] >= vocabulary_size_) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
        const std::size_t context = context_index(ids, position);
        counts_[context][ids[position]] += 1.0;
        totals_[context] += 1.0;
    }
}

double CharNgram::probability(const TokenSequence& context, const std::size_t next) const {
    if (next >= vocabulary_size_) {
        throw std::invalid_argument("id is outside the vocabulary");
    }
    const std::size_t index = context_index(context, context.size());
    return (counts_[index][next] + alpha_) /
           (totals_[index] + alpha_ * static_cast<double>(vocabulary_size_));
}

double CharNgram::cross_entropy(const TokenSequence& ids) const {
    if (ids.size() < 2) {
        throw std::invalid_argument("cross-entropy needs at least one prediction");
    }
    double total = 0.0;
    for (std::size_t position = 1; position < ids.size(); ++position) {
        if (ids[position] >= vocabulary_size_) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
        const std::size_t context = context_index(ids, position);
        total -= std::log((counts_[context][ids[position]] + alpha_) /
                          (totals_[context] + alpha_ * static_cast<double>(vocabulary_size_)));
    }
    return total / static_cast<double>(ids.size() - 1);
}

} // namespace ml_scratch
