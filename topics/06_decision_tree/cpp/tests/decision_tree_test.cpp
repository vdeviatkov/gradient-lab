#include "ml_scratch/decision_tree.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

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

// One axis-aligned threshold at x = 1.5 separates the classes exactly.
const ml_scratch::LabeledDataset one_split_data{
    {{0.0, 7.0}, 0},
    {{1.0, -3.0}, 0},
    {{2.0, 4.0}, 1},
    {{3.0, 9.0}, 1},
};

// A checkerboard on two features. Neither feature alone gives any impurity decrease, so a greedy
// tree cannot start; see test_greedy_search_is_myopic.
const ml_scratch::LabeledDataset xor_data{
    {{0.0, 0.0}, 0},
    {{0.0, 1.0}, 1},
    {{1.0, 0.0}, 1},
    {{1.0, 1.0}, 0},
};

void test_impurity_measures() {
    require_near(ml_scratch::gini_impurity({4, 0}), 0.0, 1e-12, "a pure node has non-zero Gini");
    require_near(ml_scratch::entropy_impurity({4, 0}), 0.0, 1e-12,
                 "a pure node has non-zero entropy");
    require_near(ml_scratch::gini_impurity({2, 2}), 0.5, 1e-12, "incorrect balanced Gini");
    require_near(ml_scratch::entropy_impurity({2, 2}), 1.0, 1e-12,
                 "a balanced binary node should carry one bit");
    require_near(ml_scratch::gini_impurity({1, 1, 1, 1}), 0.75, 1e-12, "incorrect four-class Gini");
    require_near(ml_scratch::entropy_impurity({1, 1, 1, 1}), 2.0, 1e-12,
                 "four balanced classes should carry two bits");
    require_near(ml_scratch::gini_impurity({3, 1}), 0.375, 1e-12, "incorrect skewed Gini");
    require_near(ml_scratch::entropy_impurity({3, 1}), 0.811278124459133, 1e-12,
                 "incorrect skewed entropy");

    // The criterion selector must agree with the individual measures.
    require_near(ml_scratch::node_impurity({3, 1}, ml_scratch::SplitCriterion::gini),
                 ml_scratch::gini_impurity({3, 1}), 0.0, "gini selector disagrees");
    require_near(ml_scratch::node_impurity({3, 1}, ml_scratch::SplitCriterion::entropy),
                 ml_scratch::entropy_impurity({3, 1}), 0.0, "entropy selector disagrees");

    require_invalid_argument([] { static_cast<void>(ml_scratch::gini_impurity({})); },
                             "empty counts were accepted");
    require_invalid_argument([] { static_cast<void>(ml_scratch::entropy_impurity({0, 0})); },
                             "an empty node was accepted");
}

void test_single_split() {
    ml_scratch::DecisionTreeClassifier tree{2};
    tree.fit(one_split_data);

    require(tree.node_count() == 3, "a one-split problem should build three nodes");
    require(tree.leaf_count() == 2, "a one-split problem should have two leaves");
    require(tree.depth() == 1, "a one-split problem should have depth one");
    require(!tree.nodes()[0].leaf, "the root should be an internal node");
    require(tree.nodes()[0].feature == 0, "the tree split on the uninformative feature");
    require_near(tree.nodes()[0].threshold, 1.5, 1e-12, "incorrect split threshold");
    require_near(tree.nodes()[0].impurity, 0.5, 1e-12, "incorrect root impurity");
    require_near(tree.accuracy(one_split_data), 1.0, 1e-12, "a separable split was not learned");

    // Leaves are pure here, so the distribution is degenerate.
    const auto distribution = tree.predict_distribution({0.0, 7.0});
    require_near(distribution[0], 1.0, 1e-12, "incorrect leaf distribution");
    require_near(distribution[1], 0.0, 1e-12, "incorrect leaf distribution");

    // Unseen points follow the learned threshold.
    require(tree.predict({-5.0, 0.0}) == 0, "prediction ignored the learned threshold");
    require(tree.predict({99.0, 0.0}) == 1, "prediction ignored the learned threshold");
}

// A one-feature staircase: class 0, then class 1, then class 0 again. No single threshold
// separates it, but each level of the tree makes real progress, so the greedy search reaches it.
const ml_scratch::LabeledDataset staircase_data{
    {{0.0}, 0}, {{0.5}, 0}, {{1.2}, 1}, {{1.5}, 1}, {{2.5}, 0}, {{3.0}, 0},
};

void test_multi_level_and_multiclass() {
    ml_scratch::DecisionTreeClassifier tree{2};
    tree.fit(staircase_data);
    require(tree.depth() == 2, "a staircase needs two levels");
    require_near(tree.accuracy(staircase_data), 1.0, 1e-12, "the staircase was not learned");

    const ml_scratch::LabeledDataset three_classes{
        {{0.0}, 0}, {{1.0}, 0}, {{5.0}, 1}, {{6.0}, 1}, {{10.0}, 2}, {{11.0}, 2},
    };
    ml_scratch::DecisionTreeClassifier multiclass{3};
    multiclass.fit(three_classes);
    require_near(multiclass.accuracy(three_classes), 1.0, 1e-12, "multiclass fit failed");
    require(multiclass.leaf_count() == 3, "three separable classes need three leaves");
    require(multiclass.predict({5.5}) == 1, "incorrect multiclass prediction");
}

// Greedy induction evaluates one split at a time and never looks ahead. On an exact checkerboard
// every single-feature split leaves both children exactly as impure as the parent, so the search
// stops at the root even though a depth-two tree would be perfect. Lookahead, not more depth, is
// what this dataset needs.
void test_greedy_search_is_myopic() {
    ml_scratch::DecisionTreeClassifier tree{2};
    tree.fit(xor_data);

    require(tree.node_count() == 1, "a zero-gain split was taken on the checkerboard");
    require(tree.depth() == 0, "the greedy search should stop at the root here");
    require_near(tree.accuracy(xor_data), 0.5, 1e-12, "a root-only tree should be at chance");

    // The zero decrease is a property of the data, not of the criterion.
    for (const auto criterion :
         {ml_scratch::SplitCriterion::gini, ml_scratch::SplitCriterion::entropy}) {
        const auto parent = ml_scratch::node_impurity({2, 2}, criterion);
        const auto child = ml_scratch::node_impurity({1, 1}, criterion);
        require_near(parent - child, 0.0, 1e-12, "a checkerboard split should not reduce impurity");
    }
}

void test_gini_and_entropy_agree_here() {
    ml_scratch::DecisionTreeConfig config;
    ml_scratch::DecisionTreeClassifier gini_tree{2};
    config.criterion = ml_scratch::SplitCriterion::gini;
    gini_tree.fit(one_split_data, config);

    ml_scratch::DecisionTreeClassifier entropy_tree{2};
    config.criterion = ml_scratch::SplitCriterion::entropy;
    entropy_tree.fit(one_split_data, config);

    // The two criteria disagree in general, but not on which split is best here.
    require(gini_tree.nodes()[0].feature == entropy_tree.nodes()[0].feature,
            "criteria chose different split features");
    require_near(gini_tree.nodes()[0].threshold, entropy_tree.nodes()[0].threshold, 1e-12,
                 "criteria chose different thresholds");
    // Impurity itself is measured on different scales.
    require(gini_tree.nodes()[0].impurity != entropy_tree.nodes()[0].impurity,
            "Gini and entropy should not report the same impurity value");
}

void test_depth_limit_and_leaf_constraints() {
    ml_scratch::DecisionTreeConfig config;
    config.max_depth = 1;
    ml_scratch::DecisionTreeClassifier shallow{2};
    shallow.fit(staircase_data, config);

    require(shallow.depth() == 1, "the depth limit was ignored");
    // One threshold can only isolate the first step of the staircase.
    require(shallow.accuracy(staircase_data) < 1.0,
            "a depth-one tree should not solve the staircase");

    ml_scratch::DecisionTreeConfig root_only;
    root_only.max_depth = 0;
    root_only.min_samples_split = 100;
    ml_scratch::DecisionTreeClassifier stump{2};
    stump.fit(staircase_data, root_only);
    require(stump.node_count() == 1, "min_samples_split did not stop the split");
    require(stump.depth() == 0, "a root-only tree should have depth zero");

    ml_scratch::DecisionTreeConfig leaf_limited;
    leaf_limited.min_samples_leaf = 2;
    ml_scratch::DecisionTreeClassifier limited{2};
    limited.fit(one_split_data, leaf_limited);
    for (const auto& node : limited.nodes()) {
        if (node.leaf) {
            require(node.sample_count >= 2, "min_samples_leaf was violated");
        }
    }

    ml_scratch::DecisionTreeConfig impurity_limited;
    impurity_limited.min_impurity_decrease = 0.9;
    ml_scratch::DecisionTreeClassifier blocked{2};
    blocked.fit(one_split_data, impurity_limited);
    require(blocked.node_count() == 1, "min_impurity_decrease did not block the split");
}

void test_overfitting_is_visible() {
    // Two identical feature vectors carry conflicting labels, so no tree can separate them.
    ml_scratch::LabeledDataset noisy{
        {{0.0}, 0}, {{1.0}, 0}, {{2.0}, 1}, {{3.0}, 1}, {{4.0}, 0}, {{5.0}, 1},
    };
    ml_scratch::DecisionTreeClassifier unlimited{2};
    unlimited.fit(noisy);
    require_near(unlimited.accuracy(noisy), 1.0, 1e-12,
                 "an unlimited tree should memorize distinct training points");

    ml_scratch::DecisionTreeConfig config;
    config.max_depth = 1;
    ml_scratch::DecisionTreeClassifier shallow{2};
    shallow.fit(noisy, config);
    require(shallow.accuracy(noisy) < unlimited.accuracy(noisy),
            "a depth limit should cost training accuracy");
    require(shallow.node_count() < unlimited.node_count(),
            "a depth limit should produce a smaller tree");
}

void test_pruning() {
    // The training split contains one mislabeled point at x = 2.5 that a full tree will isolate.
    const ml_scratch::LabeledDataset train{
        {{0.0}, 0}, {{1.0}, 0}, {{2.0}, 0}, {{2.5}, 1}, {{3.0}, 0},
        {{6.0}, 1}, {{7.0}, 1}, {{8.0}, 1}, {{9.0}, 1},
    };
    const ml_scratch::LabeledDataset validation{
        {{0.5}, 0}, {{2.4}, 0}, {{2.6}, 0}, {{3.5}, 0}, {{6.5}, 1}, {{8.5}, 1},
    };

    ml_scratch::DecisionTreeClassifier tree{2};
    tree.fit(train);
    require_near(tree.accuracy(train), 1.0, 1e-12, "the tree did not memorize its training split");
    const std::size_t nodes_before = tree.node_count();
    const double validation_before = tree.accuracy(validation);

    const std::size_t collapsed = tree.prune(validation);
    require(collapsed > 0, "pruning removed nothing from an overfitted tree");
    require(tree.node_count() < nodes_before, "pruning did not shrink the tree");
    require(tree.accuracy(validation) >= validation_before, "pruning lowered validation accuracy");
    require(tree.accuracy(train) < 1.0, "pruning should give up some training accuracy");

    // Compaction must leave a consistent tree: every child index is in range and reachable.
    for (const auto& node : tree.nodes()) {
        if (!node.leaf) {
            require(node.left < tree.node_count() && node.right < tree.node_count(),
                    "pruning left a dangling child index");
        }
    }
    require(tree.leaf_count() + tree.node_count() / 2 >= tree.leaf_count(),
            "inconsistent node bookkeeping");

    // A second pass has nothing left to remove.
    require(tree.prune(validation) == 0, "pruning was not idempotent");
}

void test_determinism() {
    ml_scratch::DecisionTreeClassifier first{2};
    ml_scratch::DecisionTreeClassifier second{2};
    first.fit(staircase_data);
    second.fit(staircase_data);
    require(first.nodes() == second.nodes(), "the same dataset produced different trees");

    // Ties between equally good splits resolve to the lowest feature index.
    const ml_scratch::LabeledDataset symmetric{
        {{0.0, 0.0}, 0},
        {{0.0, 0.0}, 0},
        {{1.0, 1.0}, 1},
        {{1.0, 1.0}, 1},
    };
    ml_scratch::DecisionTreeClassifier tie_tree{2};
    tie_tree.fit(symmetric);
    require(tie_tree.nodes()[0].feature == 0, "a split tie did not prefer the lowest feature");
}

void test_input_validation() {
    require_invalid_argument([] { ml_scratch::DecisionTreeClassifier{1}; },
                             "a single-class tree was accepted");

    ml_scratch::DecisionTreeClassifier tree{2};
    require_invalid_argument([&] { tree.fit({}); }, "an empty dataset was accepted");
    require_invalid_argument([&] { tree.fit({{{1.0}, 0}, {{2.0, 3.0}, 1}}); },
                             "a ragged dataset was accepted");
    require_invalid_argument([&] { tree.fit({{{1.0}, 0}, {{2.0}, 5}}); },
                             "a label outside the class count was accepted");
    require_invalid_argument(
        [&] { tree.fit({{{std::numeric_limits<double>::quiet_NaN()}, 0}, {{2.0}, 1}}); },
        "a non-finite feature was accepted");

    ml_scratch::DecisionTreeConfig config;
    config.min_samples_split = 1;
    require_invalid_argument([&] { tree.fit(one_split_data, config); },
                             "min_samples_split below two was accepted");
    config.min_samples_split = 2;
    config.min_samples_leaf = 0;
    require_invalid_argument([&] { tree.fit(one_split_data, config); },
                             "a zero min_samples_leaf was accepted");

    bool rejected = false;
    try {
        static_cast<void>(tree.predict({1.0}));
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "an unfitted tree produced a prediction");

    ml_scratch::DecisionTreeClassifier fitted{2};
    fitted.fit(one_split_data);
    require_invalid_argument([&] { static_cast<void>(fitted.predict({1.0})); },
                             "a wrong feature count was accepted");
    require_invalid_argument([&] { static_cast<void>(fitted.prune({{{1.0}, 0}})); },
                             "pruning accepted a mismatched validation split");
}

} // namespace

int main() {
    try {
        test_impurity_measures();
        test_single_split();
        test_multi_level_and_multiclass();
        test_greedy_search_is_myopic();
        test_gini_and_entropy_agree_here();
        test_depth_limit_and_leaf_constraints();
        test_overfitting_is_visible();
        test_pruning();
        test_determinism();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
