#include "ml_scratch/text_corpus.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

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

void test_vocabulary_and_round_trip() {
    const auto corpus = ml_scratch::build_text_corpus("hello world! hello again.", 0.2, 0.2);
    // Sorted distinct characters, by byte value: space and punctuation before letters.
    require(corpus.vocabulary == " !.adeghilnorw", "the vocabulary should be the sorted distinct characters");
    require(corpus.vocabulary_size() == 14, "vocabulary size");
    require(corpus.encode('h') == 7 && corpus.encode(' ') == 0, "ids index the vocabulary");
    require(corpus.decode(corpus.encode("hello")) == "hello", "encode and decode should round-trip");
    require_invalid_argument([&] { static_cast<void>(corpus.encode('z')); },
                             "a character outside the vocabulary should be rejected");
    require_invalid_argument([&] { static_cast<void>(corpus.decode({99})); },
                             "an id outside the vocabulary should be rejected");
}

void test_splits_are_contiguous_and_in_order() {
    const std::string text = "abcdefghijklmnopqrst"; // 20 characters
    const auto corpus = ml_scratch::build_text_corpus(text, 0.25, 0.1);
    // 25% of 20 is 5 validation, 10% is 2 test, the leading 13 are training.
    require(corpus.train.size() == 13 && corpus.validation.size() == 5 && corpus.test.size() == 2,
            "split sizes should follow the fractions");
    require(corpus.decode(corpus.train) == "abcdefghijklm" &&
                corpus.decode(corpus.validation) == "nopqr" && corpus.decode(corpus.test) == "st",
            "splits should be contiguous slices in text order");

    require_invalid_argument([&] { static_cast<void>(ml_scratch::build_text_corpus("", 0.1, 0.1)); },
                             "empty text should be rejected");
    require_invalid_argument([&] { static_cast<void>(ml_scratch::build_text_corpus(text, 0.5, 0.5)); },
                             "fractions that leave no training text should be rejected");
    require_invalid_argument([&] { static_cast<void>(ml_scratch::build_text_corpus("ab", 0.1, 0.1)); },
                             "text too short for a split should be rejected");
}

void test_loading_a_file_with_a_prefix() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_corpus_test.txt").string();
    {
        std::ofstream stream{path};
        stream << "the quick brown fox jumps over the lazy dog";
    }
    const auto whole = ml_scratch::load_text_corpus(path, 0, 0.1, 0.1);
    const auto prefix = ml_scratch::load_text_corpus(path, 20, 0.1, 0.1);
    require(whole.train.size() + whole.validation.size() + whole.test.size() == 43,
            "the whole file should be loaded when no limit is given");
    require(prefix.train.size() + prefix.validation.size() + prefix.test.size() == 20,
            "only the prefix should be loaded when a limit is given");
    require(prefix.vocabulary.size() < whole.vocabulary.size(),
            "the prefix's vocabulary should be built from the prefix alone");
    std::filesystem::remove(path);

    bool rejected = false;
    try {
        static_cast<void>(ml_scratch::load_text_corpus(path));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "a missing corpus should be reported");
}

void test_ngram_probabilities_by_hand() {
    // Vocabulary of three ids; text 0 1 0 1 0 1 1.
    const ml_scratch::TokenSequence ids{0, 1, 0, 1, 0, 1, 1};

    ml_scratch::CharNgram unigram{1, 3, 1.0};
    unigram.fit(ids);
    // Counts: id 0 three times, id 1 four times, id 2 never; alpha 1 over 3 ids.
    require_near(unigram.probability({}, 0), (3.0 + 1.0) / (7.0 + 3.0), 1e-15, "unigram P(0)");
    require_near(unigram.probability({}, 2), 1.0 / 10.0, 1e-15, "unigram smoothing gives unseen ids mass");
    double expected = 0.0;
    for (std::size_t position = 1; position < ids.size(); ++position) {
        expected -= std::log(unigram.probability({}, ids[position])) / 6.0;
    }
    require_near(unigram.cross_entropy(ids), expected, 1e-15, "unigram cross-entropy by hand");

    ml_scratch::CharNgram bigram{2, 3, 0.5};
    bigram.fit(ids);
    // After a 0 the next id was 1 three times and nothing else; after a 1 it was 0 twice and 1
    // once. The first id is counted after an imaginary leading 0.
    require_near(bigram.probability({0}, 1), (3.0 + 0.5) / (4.0 + 1.5), 1e-15, "bigram P(1 | 0)");
    require_near(bigram.probability({1}, 0), (2.0 + 0.5) / (3.0 + 1.5), 1e-15, "bigram P(0 | 1)");
    require_near(bigram.probability({1}, 2), 0.5 / 4.5, 1e-15, "bigram unseen continuation");
    // Only the most recent order - 1 ids of a longer context matter.
    require_near(bigram.probability({2, 2, 1}, 0), bigram.probability({1}, 0), 0.0,
                 "only the last id of the context should matter to a bigram");

    // On perfectly periodic text the bigram model is nearly certain while the unigram is not.
    ml_scratch::TokenSequence periodic;
    for (std::size_t index = 0; index < 600; ++index) {
        periodic.push_back(index % 3);
    }
    ml_scratch::CharNgram flat{1, 3, 1.0};
    ml_scratch::CharNgram sharp{2, 3, 1.0};
    flat.fit(periodic);
    sharp.fit(periodic);
    require(sharp.cross_entropy(periodic) < 0.05 && flat.cross_entropy(periodic) > 1.0,
            "a bigram model should capture a period-3 pattern that a unigram cannot");

    require_invalid_argument([] { ml_scratch::CharNgram bad{0, 3}; }, "order zero should be rejected");
    require_invalid_argument([] { ml_scratch::CharNgram bad{2, 3, 0.0}; },
                             "a zero pseudo-count should be rejected");
    require_invalid_argument([&] { static_cast<void>(unigram.cross_entropy({0})); },
                             "cross-entropy needs a prediction");
    require_invalid_argument([&] { unigram.fit({0, 7}); }, "ids outside the vocabulary are rejected");

    require_near(ml_scratch::perplexity(std::log(4.0)), 4.0, 1e-12,
                 "perplexity is the exponential of the cross-entropy");
    require_near(ml_scratch::bits_per_character(std::log(8.0)), 3.0, 1e-12,
                 "eight equally likely characters are three bits");
}

} // namespace

int main() {
    try {
        test_vocabulary_and_round_trip();
        test_splits_are_contiguous_and_in_order();
        test_loading_a_file_with_a_prefix();
        test_ngram_probabilities_by_hand();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
