#include "ml_scratch/dataset.hpp"
#include "ml_scratch/decision_tree.hpp"
#include "ml_scratch/deterministic_random.hpp"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t seed = 20260908;
constexpr std::size_t class_count = 2;
constexpr std::size_t samples_per_class = 300;
constexpr std::size_t train_per_class = 150;
constexpr std::size_t validation_per_class = 75;
constexpr std::size_t noise_feature_count = 3;
constexpr double label_noise_rate = 0.08;
constexpr std::size_t max_swept_depth = 12;

struct Splits {
    ml_scratch::LabeledDataset train;
    ml_scratch::LabeledDataset validation;
    ml_scratch::LabeledDataset test;
};

// Two overlapping Gaussian classes described by two informative features, followed by three pure
// noise features. A fraction of the labels is then flipped. The flips belong to the data-generating
// process, so every split carries them at the same rate and the noise cannot be learned from the
// features. A tree deep enough to isolate individual training points will fit those flips and the
// noise features, which is exactly the overfitting this experiment measures.
Splits build_dataset() {
    constexpr double centers[class_count][2]{{0.0, 0.0}, {2.2, 2.2}};
    constexpr double spread = 1.2;

    ml_scratch::DeterministicRandom random{seed};
    Splits splits;
    for (std::size_t label = 0; label < class_count; ++label) {
        for (std::size_t index = 0; index < samples_per_class; ++index) {
            std::vector<double> features{centers[label][0] + spread * random.gaussian(),
                                         centers[label][1] + spread * random.gaussian()};
            for (std::size_t noise = 0; noise < noise_feature_count; ++noise) {
                features.push_back(random.gaussian());
            }

            const std::size_t observed =
                random.bernoulli(label_noise_rate) ? class_count - 1 - label : label;
            ml_scratch::LabeledDataset& split = index < train_per_class ? splits.train
                                                : index < train_per_class + validation_per_class
                                                    ? splits.validation
                                                    : splits.test;
            split.push_back({std::move(features), observed});
        }
    }
    return splits;
}

struct DepthReport {
    std::size_t depth;
    double train_accuracy;
    double validation_accuracy;
    double test_accuracy;
    std::size_t leaves;
    std::size_t nodes;
};

DepthReport evaluate_depth(const Splits& splits, const std::size_t max_depth,
                           const ml_scratch::SplitCriterion criterion) {
    ml_scratch::DecisionTreeConfig config;
    config.criterion = criterion;
    config.max_depth = max_depth;

    ml_scratch::DecisionTreeClassifier tree{class_count};
    tree.fit(splits.train, config);
    return {tree.depth(),
            tree.accuracy(splits.train),
            tree.accuracy(splits.validation),
            tree.accuracy(splits.test),
            tree.leaf_count(),
            tree.node_count()};
}

void print_depth_report(const std::string_view label, const DepthReport& report) {
    std::cout << "  " << label << ": depth=" << report.depth << ", leaves=" << report.leaves
              << ", train=" << report.train_accuracy
              << ", validation=" << report.validation_accuracy << ", test=" << report.test_accuracy
              << '\n';
}

double majority_baseline_accuracy(const Splits& splits) {
    const auto counts = ml_scratch::class_counts(splits.train, class_count);
    const auto majority = static_cast<std::size_t>(
        std::distance(counts.begin(), std::max_element(counts.begin(), counts.end())));

    std::size_t correct = 0;
    for (const auto& sample : splits.test) {
        if (sample.label == majority) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(splits.test.size());
}

} // namespace

int main() {
    const Splits splits = build_dataset();

    std::cout << std::fixed << std::setprecision(4) << "C++ decision tree experiment (seed=" << seed
              << ")\n"
              << "dataset: " << class_count * samples_per_class << " samples, "
              << splits.train.front().features.size() << " features (2 informative, "
              << noise_feature_count << " noise), " << class_count << " classes, "
              << label_noise_rate * 100.0 << "% label noise\n"
              << "splits: train=" << splits.train.size()
              << ", validation=" << splits.validation.size() << ", test=" << splits.test.size()
              << "\n\n";

    std::cout << "depth sweep (Gini)\n";
    const DepthReport unlimited = evaluate_depth(splits, 0, ml_scratch::SplitCriterion::gini);
    std::size_t selected_depth = 1;
    double best_validation = -1.0;
    for (std::size_t max_depth = 1; max_depth <= max_swept_depth; ++max_depth) {
        const DepthReport report =
            evaluate_depth(splits, max_depth, ml_scratch::SplitCriterion::gini);
        std::string label = "max_depth=" + std::to_string(max_depth);
        label.resize(std::max<std::size_t>(label.size(), 15), ' ');
        print_depth_report(label, report);
        // The shallowest depth wins a tie, since a smaller tree is the simpler explanation.
        if (report.validation_accuracy > best_validation) {
            best_validation = report.validation_accuracy;
            selected_depth = max_depth;
        }
    }
    print_depth_report("unlimited      ", unlimited);
    std::cout << "  selected max_depth=" << selected_depth << " by validation accuracy\n\n";

    const DepthReport selected_gini =
        evaluate_depth(splits, selected_depth, ml_scratch::SplitCriterion::gini);
    const DepthReport selected_entropy =
        evaluate_depth(splits, selected_depth, ml_scratch::SplitCriterion::entropy);
    std::cout << "criterion comparison at max_depth=" << selected_depth << '\n';
    print_depth_report("Gini           ", selected_gini);
    print_depth_report("entropy        ", selected_entropy);

    ml_scratch::DecisionTreeClassifier pruned{class_count};
    pruned.fit(splits.train);
    const std::size_t nodes_before = pruned.node_count();
    const std::size_t collapsed = pruned.prune(splits.validation);

    std::cout << "\nreduced-error pruning of the unlimited tree\n"
              << "  before: nodes=" << nodes_before << ", leaves=" << unlimited.leaves
              << ", depth=" << unlimited.depth << ", train=" << unlimited.train_accuracy
              << ", validation=" << unlimited.validation_accuracy
              << ", test=" << unlimited.test_accuracy << '\n'
              << "  after : nodes=" << pruned.node_count() << ", leaves=" << pruned.leaf_count()
              << ", depth=" << pruned.depth() << ", train=" << pruned.accuracy(splits.train)
              << ", validation=" << pruned.accuracy(splits.validation)
              << ", test=" << pruned.accuracy(splits.test) << '\n'
              << "  collapsed " << collapsed << " internal nodes\n";

    const double baseline = majority_baseline_accuracy(splits);
    const double pruned_test = pruned.accuracy(splits.test);
    std::cout << "\nmajority-class baseline test accuracy=" << baseline << '\n';

    const bool memorized_training_data = unlimited.train_accuracy == 1.0;
    const bool depth_limit_generalized_better =
        selected_gini.test_accuracy > unlimited.test_accuracy;
    const bool pruning_shrank_the_tree = pruned.node_count() < nodes_before;
    const bool pruning_kept_accuracy = pruned_test >= unlimited.test_accuracy;
    const bool beat_the_baseline = selected_gini.test_accuracy > baseline;

    if (!memorized_training_data || !depth_limit_generalized_better || !pruning_shrank_the_tree ||
        !pruning_kept_accuracy || !beat_the_baseline) {
        std::cerr << "experiment failed its criteria: unlimited train=" << unlimited.train_accuracy
                  << ", unlimited test=" << unlimited.test_accuracy
                  << ", selected test=" << selected_gini.test_accuracy
                  << ", pruned test=" << pruned_test << ", baseline=" << baseline << '\n';
        return 1;
    }
    return 0;
}
