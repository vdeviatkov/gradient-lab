#include "ml_scratch/text_corpus.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace ml_scratch {

std::size_t TextCorpus::encode(const char character) const {
    const auto position = std::lower_bound(vocabulary.begin(), vocabulary.end(), character);
    if (position == vocabulary.end() || *position != character) {
        throw std::invalid_argument("character is not in the vocabulary");
    }
    return static_cast<std::size_t>(position - vocabulary.begin());
}

TokenSequence TextCorpus::encode(const std::string& text) const {
    TokenSequence ids;
    ids.reserve(text.size());
    for (const char character : text) {
        ids.push_back(encode(character));
    }
    return ids;
}

std::string TextCorpus::decode(const TokenSequence& ids) const {
    std::string text;
    text.reserve(ids.size());
    for (const std::size_t id : ids) {
        if (id >= vocabulary.size()) {
            throw std::invalid_argument("id is outside the vocabulary");
        }
        text.push_back(vocabulary[id]);
    }
    return text;
}

TextCorpus build_text_corpus(const std::string& text, const double validation_fraction,
                             const double test_fraction) {
    if (text.empty()) {
        throw std::invalid_argument("the corpus text is empty");
    }
    if (!std::isfinite(validation_fraction) || !std::isfinite(test_fraction) ||
        validation_fraction <= 0.0 || test_fraction <= 0.0 ||
        validation_fraction + test_fraction >= 1.0) {
        throw std::invalid_argument("split fractions must be positive and sum to less than one");
    }

    TextCorpus corpus;
    corpus.vocabulary = text;
    std::sort(corpus.vocabulary.begin(), corpus.vocabulary.end());
    corpus.vocabulary.erase(std::unique(corpus.vocabulary.begin(), corpus.vocabulary.end()),
                            corpus.vocabulary.end());

    const auto total = static_cast<double>(text.size());
    const auto validation_count = static_cast<std::size_t>(validation_fraction * total);
    const auto test_count = static_cast<std::size_t>(test_fraction * total);
    if (validation_count == 0 || test_count == 0 || validation_count + test_count >= text.size()) {
        throw std::invalid_argument("the text is too short for the requested splits");
    }
    const std::size_t train_count = text.size() - validation_count - test_count;

    const TokenSequence ids = corpus.encode(text);
    const auto train_end = ids.begin() + static_cast<std::ptrdiff_t>(train_count);
    const auto validation_end = train_end + static_cast<std::ptrdiff_t>(validation_count);
    corpus.train.assign(ids.begin(), train_end);
    corpus.validation.assign(train_end, validation_end);
    corpus.test.assign(validation_end, ids.end());
    return corpus;
}

TextCorpus load_text_corpus(const std::string& path, const std::size_t max_characters,
                            const double validation_fraction, const double test_fraction) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        throw std::runtime_error("cannot open corpus " + path);
    }
    std::string text{std::istreambuf_iterator<char>{stream}, std::istreambuf_iterator<char>{}};
    if (text.empty()) {
        throw std::runtime_error("corpus is empty: " + path);
    }
    if (max_characters > 0 && text.size() > max_characters) {
        text.resize(max_characters);
    }
    return build_text_corpus(text, validation_fraction, test_fraction);
}

double perplexity(const double cross_entropy) { return std::exp(cross_entropy); }

double bits_per_character(const double cross_entropy) { return cross_entropy / std::log(2.0); }

} // namespace ml_scratch
