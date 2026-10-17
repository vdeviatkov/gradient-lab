#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ml_scratch {

// A sequence of character ids: indices into a TextCorpus vocabulary.
using TokenSequence = std::vector<std::size_t>;

// A character-level modeling corpus with a fixed vocabulary and fixed contiguous splits. The
// sequence milestones — recurrent, gated, and attention-based — all model the same corpus under
// the same next-character objective, so this is what makes them comparable.
struct TextCorpus {
    // Every distinct character in the loaded text, sorted by byte value; a character's id is its
    // index here. Built from the whole loaded text rather than the training split alone, so no
    // split contains a character the model cannot represent.
    std::string vocabulary;
    TokenSequence train;
    TokenSequence validation;
    TokenSequence test;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept { return vocabulary.size(); }
    // Throws std::invalid_argument for a character outside the vocabulary.
    [[nodiscard]] std::size_t encode(char character) const;
    [[nodiscard]] TokenSequence encode(const std::string& text) const;
    // Throws std::invalid_argument for an id outside the vocabulary.
    [[nodiscard]] std::string decode(const TokenSequence& ids) const;
};

// Builds a corpus from `text`: the vocabulary from every character, then contiguous splits with
// the first `validation_fraction` and `test_fraction` of the text taken from its END for
// validation and test, and everything before them for training. Contiguity matters for a
// sequence model, since a random split would let the model see the characters on either side of
// every held-out one. Throws std::invalid_argument when the fractions leave any split empty.
[[nodiscard]] TextCorpus build_text_corpus(const std::string& text,
                                           double validation_fraction = 0.05,
                                           double test_fraction = 0.05);

// Reads a text file and builds a corpus from its first `max_characters` bytes, or all of it when
// that is zero. Throws std::runtime_error when the file is missing or empty.
[[nodiscard]] TextCorpus load_text_corpus(const std::string& path, std::size_t max_characters = 0,
                                          double validation_fraction = 0.05,
                                          double test_fraction = 0.05);

// exp of a per-character cross-entropy in nats: the effective number of equally likely characters
// the model is choosing between.
[[nodiscard]] double perplexity(double cross_entropy);
// Cross-entropy in bits per character.
[[nodiscard]] double bits_per_character(double cross_entropy);

} // namespace ml_scratch
