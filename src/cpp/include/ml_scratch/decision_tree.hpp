#pragma once

#include "ml_scratch/dataset.hpp"

#include <cstddef>
#include <vector>

namespace ml_scratch {

enum class SplitCriterion {
    gini,
    entropy,
};

// Node impurity from per-class counts. Both measures are zero for a pure node and maximal for a
// uniform distribution; entropy is measured in bits.
[[nodiscard]] double gini_impurity(const std::vector<std::size_t>& counts);
[[nodiscard]] double entropy_impurity(const std::vector<std::size_t>& counts);
[[nodiscard]] double node_impurity(const std::vector<std::size_t>& counts,
                                   SplitCriterion criterion);

struct DecisionTreeConfig {
    SplitCriterion criterion{SplitCriterion::gini};
    // Zero means unlimited. A root-only tree has depth zero.
    std::size_t max_depth{0};
    std::size_t min_samples_split{2};
    std::size_t min_samples_leaf{1};
    // A split must beat this impurity decrease, measured against the node's own samples.
    double min_impurity_decrease{0.0};
};

struct DecisionTreeNode {
    bool leaf{true};
    // Meaningful only for an internal node: samples with feature <= threshold go left.
    std::size_t feature{0};
    double threshold{0.0};
    std::size_t left{0};
    std::size_t right{0};
    // Majority label of the training samples that reached this node, kept for every node so that
    // pruning can collapse any subtree into a leaf.
    std::size_t prediction{0};
    std::size_t sample_count{0};
    double impurity{0.0};
    std::vector<std::size_t> class_counts;

    bool operator==(const DecisionTreeNode&) const = default;
};

class DecisionTreeClassifier {
  public:
    explicit DecisionTreeClassifier(std::size_t class_count);

    void fit(const LabeledDataset& dataset, const DecisionTreeConfig& config = {});

    [[nodiscard]] std::size_t predict(const std::vector<double>& features) const;
    [[nodiscard]] std::vector<double>
    predict_distribution(const std::vector<double>& features) const;
    [[nodiscard]] double accuracy(const LabeledDataset& dataset) const;

    // Reduced-error pruning. Repeatedly collapses the internal node whose removal does not lower
    // accuracy on the validation split, preferring the one that removes the most nodes. Returns
    // how many internal nodes became leaves. The validation split must not be the training split.
    std::size_t prune(const LabeledDataset& validation);

    [[nodiscard]] std::size_t depth() const;
    [[nodiscard]] std::size_t leaf_count() const;
    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::size_t class_count() const noexcept { return class_count_; }
    [[nodiscard]] std::size_t feature_count() const noexcept { return feature_count_; }
    [[nodiscard]] bool fitted() const noexcept { return !nodes_.empty(); }
    [[nodiscard]] const std::vector<DecisionTreeNode>& nodes() const noexcept { return nodes_; }

  private:
    void require_fitted() const;
    [[nodiscard]] const DecisionTreeNode& leaf_for(const std::vector<double>& features) const;
    std::size_t build(const LabeledDataset& dataset, std::vector<std::size_t>& indices,
                      std::size_t begin, std::size_t end, std::size_t depth,
                      const DecisionTreeConfig& config);
    // Drops nodes that no longer belong to the tree so node_count() stays meaningful.
    void compact();

    std::size_t class_count_;
    std::size_t feature_count_{0};
    std::vector<DecisionTreeNode> nodes_;
};

} // namespace ml_scratch
