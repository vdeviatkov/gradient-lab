#include "ml_scratch/decision_tree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

std::size_t total_count(const std::vector<std::size_t>& counts) {
    if (counts.empty()) {
        throw std::invalid_argument("counts must not be empty");
    }
    const std::size_t total = std::accumulate(counts.begin(), counts.end(), std::size_t{0});
    if (total == 0) {
        throw std::invalid_argument("counts must contain at least one sample");
    }
    return total;
}

// Lowest class index wins a tie, so equal-count leaves predict deterministically.
std::size_t majority_class(const std::vector<std::size_t>& counts) {
    std::size_t best = 0;
    for (std::size_t label = 1; label < counts.size(); ++label) {
        if (counts[label] > counts[best]) {
            best = label;
        }
    }
    return best;
}

struct CandidateSplit {
    bool found{false};
    std::size_t feature{0};
    double threshold{0.0};
    std::size_t left_size{0};
    double impurity_decrease{0.0};
};

} // namespace

double gini_impurity(const std::vector<std::size_t>& counts) {
    const auto total = static_cast<double>(total_count(counts));
    double sum_of_squares = 0.0;
    for (const std::size_t count : counts) {
        const double probability = static_cast<double>(count) / total;
        sum_of_squares += probability * probability;
    }
    return std::max(1.0 - sum_of_squares, 0.0);
}

double entropy_impurity(const std::vector<std::size_t>& counts) {
    const auto total = static_cast<double>(total_count(counts));
    double entropy = 0.0;
    for (const std::size_t count : counts) {
        if (count == 0) {
            continue; // The limit of p log p as p approaches zero is zero.
        }
        const double probability = static_cast<double>(count) / total;
        entropy -= probability * std::log2(probability);
    }
    return std::max(entropy, 0.0);
}

double node_impurity(const std::vector<std::size_t>& counts, const SplitCriterion criterion) {
    return criterion == SplitCriterion::entropy ? entropy_impurity(counts) : gini_impurity(counts);
}

DecisionTreeClassifier::DecisionTreeClassifier(const std::size_t class_count)
    : class_count_(class_count) {
    if (class_count < 2) {
        throw std::invalid_argument("class_count must be at least two");
    }
}

void DecisionTreeClassifier::require_fitted() const {
    if (!fitted()) {
        throw std::logic_error("the model must be fitted first");
    }
}

std::size_t DecisionTreeClassifier::build(const LabeledDataset& dataset,
                                          std::vector<std::size_t>& indices,
                                          const std::size_t begin, const std::size_t end,
                                          const std::size_t depth,
                                          const DecisionTreeConfig& config) {
    std::vector<std::size_t> counts(class_count_, 0);
    for (std::size_t position = begin; position < end; ++position) {
        ++counts[dataset[indices[position]].label];
    }

    DecisionTreeNode node;
    node.sample_count = end - begin;
    node.impurity = node_impurity(counts, config.criterion);
    node.prediction = majority_class(counts);
    node.class_counts = std::move(counts);

    const std::size_t node_index = nodes_.size();
    nodes_.push_back(std::move(node));

    const bool depth_exhausted = config.max_depth != 0 && depth >= config.max_depth;
    if (depth_exhausted || nodes_[node_index].sample_count < config.min_samples_split ||
        nodes_[node_index].impurity <= 0.0) {
        return node_index;
    }

    const double parent_impurity = nodes_[node_index].impurity;
    const auto parent_size = static_cast<double>(end - begin);
    CandidateSplit best;

    for (std::size_t feature = 0; feature < feature_count_; ++feature) {
        // Sorting the range in place avoids a scratch copy per feature. The index tie-break keeps
        // the order, and therefore the chosen split, independent of the sort implementation.
        std::sort(indices.begin() + static_cast<std::ptrdiff_t>(begin),
                  indices.begin() + static_cast<std::ptrdiff_t>(end),
                  [&dataset, feature](const std::size_t left, const std::size_t right) {
                      const double left_value = dataset[left].features[feature];
                      const double right_value = dataset[right].features[feature];
                      if (left_value == right_value) {
                          return left < right;
                      }
                      return left_value < right_value;
                  });

        std::vector<std::size_t> left_counts(class_count_, 0);
        std::vector<std::size_t> right_counts = nodes_[node_index].class_counts;
        for (std::size_t position = begin; position + 1 < end; ++position) {
            const auto& sample = dataset[indices[position]];
            ++left_counts[sample.label];
            --right_counts[sample.label];

            const double value = sample.features[feature];
            const double next_value = dataset[indices[position + 1]].features[feature];
            if (value == next_value) {
                continue; // A split cannot separate two identical feature values.
            }

            const std::size_t left_size = position - begin + 1;
            const std::size_t right_size = end - begin - left_size;
            if (left_size < config.min_samples_leaf || right_size < config.min_samples_leaf) {
                continue;
            }

            const double decrease = parent_impurity -
                                    (static_cast<double>(left_size) / parent_size) *
                                        node_impurity(left_counts, config.criterion) -
                                    (static_cast<double>(right_size) / parent_size) *
                                        node_impurity(right_counts, config.criterion);
            if (best.found && decrease <= best.impurity_decrease) {
                continue;
            }

            double threshold = value + (next_value - value) / 2.0;
            if (!(threshold < next_value)) {
                threshold = value; // Guard against rounding up onto the next distinct value.
            }
            best = {true, feature, threshold, left_size, decrease};
        }
    }

    if (!best.found || best.impurity_decrease <= config.min_impurity_decrease) {
        return node_index;
    }

    std::sort(indices.begin() + static_cast<std::ptrdiff_t>(begin),
              indices.begin() + static_cast<std::ptrdiff_t>(end),
              [&dataset, &best](const std::size_t left, const std::size_t right) {
                  const double left_value = dataset[left].features[best.feature];
                  const double right_value = dataset[right].features[best.feature];
                  if (left_value == right_value) {
                      return left < right;
                  }
                  return left_value < right_value;
              });

    const std::size_t middle = begin + best.left_size;
    const std::size_t left_child = build(dataset, indices, begin, middle, depth + 1, config);
    const std::size_t right_child = build(dataset, indices, middle, end, depth + 1, config);

    nodes_[node_index].leaf = false;
    nodes_[node_index].feature = best.feature;
    nodes_[node_index].threshold = best.threshold;
    nodes_[node_index].left = left_child;
    nodes_[node_index].right = right_child;
    return node_index;
}

void DecisionTreeClassifier::fit(const LabeledDataset& dataset, const DecisionTreeConfig& config) {
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.class_count > class_count_) {
        throw std::invalid_argument("dataset contains a label outside the class count");
    }
    if (config.min_samples_split < 2) {
        throw std::invalid_argument("min_samples_split must be at least two");
    }
    if (config.min_samples_leaf == 0) {
        throw std::invalid_argument("min_samples_leaf must be positive");
    }
    if (!std::isfinite(config.min_impurity_decrease) || config.min_impurity_decrease < 0.0) {
        throw std::invalid_argument("min_impurity_decrease must be finite and non-negative");
    }

    feature_count_ = shape.feature_count;
    nodes_.clear();
    std::vector<std::size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), 0);
    static_cast<void>(build(dataset, indices, 0, dataset.size(), 0, config));
}

const DecisionTreeNode&
DecisionTreeClassifier::leaf_for(const std::vector<double>& features) const {
    require_fitted();
    if (features.size() != feature_count_) {
        throw std::invalid_argument("feature count does not match the fitted model");
    }
    for (const double feature : features) {
        if (!std::isfinite(feature)) {
            throw std::invalid_argument("features must be finite");
        }
    }

    std::size_t current = 0;
    while (!nodes_[current].leaf) {
        const DecisionTreeNode& node = nodes_[current];
        current = features[node.feature] <= node.threshold ? node.left : node.right;
    }
    return nodes_[current];
}

std::size_t DecisionTreeClassifier::predict(const std::vector<double>& features) const {
    return leaf_for(features).prediction;
}

std::vector<double>
DecisionTreeClassifier::predict_distribution(const std::vector<double>& features) const {
    const DecisionTreeNode& leaf = leaf_for(features);
    std::vector<double> distribution(class_count_, 0.0);
    for (std::size_t label = 0; label < class_count_; ++label) {
        distribution[label] =
            static_cast<double>(leaf.class_counts[label]) / static_cast<double>(leaf.sample_count);
    }
    return distribution;
}

double DecisionTreeClassifier::accuracy(const LabeledDataset& dataset) const {
    require_fitted();
    static_cast<void>(validate_labeled_dataset(dataset));

    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

void DecisionTreeClassifier::compact() {
    std::vector<DecisionTreeNode> compacted;
    compacted.reserve(nodes_.size());
    // Depth-first relabeling: push a node, then fix up its child indices once they are known.
    const auto copy_subtree = [this, &compacted](auto&& self,
                                                 const std::size_t index) -> std::size_t {
        const std::size_t new_index = compacted.size();
        compacted.push_back(nodes_[index]);
        if (!nodes_[index].leaf) {
            const std::size_t left = self(self, nodes_[index].left);
            const std::size_t right = self(self, nodes_[index].right);
            compacted[new_index].left = left;
            compacted[new_index].right = right;
        }
        return new_index;
    };
    static_cast<void>(copy_subtree(copy_subtree, 0));
    nodes_ = std::move(compacted);
}

std::size_t DecisionTreeClassifier::prune(const LabeledDataset& validation) {
    require_fitted();
    const DatasetShape shape = validate_labeled_dataset(validation);
    if (shape.feature_count != feature_count_) {
        throw std::invalid_argument("feature count does not match the fitted model");
    }
    if (shape.class_count > class_count_) {
        throw std::invalid_argument("dataset contains a label outside the class count");
    }

    std::size_t pruned = 0;
    for (;;) {
        const double current_accuracy = accuracy(validation);
        bool found = false;
        std::size_t best_index = 0;
        double best_accuracy = 0.0;
        std::size_t best_removed = 0;

        for (std::size_t index = 0; index < nodes_.size(); ++index) {
            if (nodes_[index].leaf) {
                continue;
            }

            nodes_[index].leaf = true;
            const double candidate_accuracy = accuracy(validation);
            nodes_[index].leaf = false;

            if (candidate_accuracy < current_accuracy) {
                continue;
            }
            // Prefer the largest accuracy, then the collapse that removes the most nodes.
            const std::size_t subtree_size = [this, index] {
                std::size_t total = 0;
                std::vector<std::size_t> stack{index};
                while (!stack.empty()) {
                    const std::size_t current = stack.back();
                    stack.pop_back();
                    ++total;
                    if (!nodes_[current].leaf) {
                        stack.push_back(nodes_[current].left);
                        stack.push_back(nodes_[current].right);
                    }
                }
                return total - 1; // The node itself survives as a leaf.
            }();

            if (!found || candidate_accuracy > best_accuracy ||
                (candidate_accuracy == best_accuracy && subtree_size > best_removed)) {
                found = true;
                best_index = index;
                best_accuracy = candidate_accuracy;
                best_removed = subtree_size;
            }
        }

        if (!found) {
            return pruned;
        }
        nodes_[best_index].leaf = true;
        ++pruned;
        compact();
    }
}

std::size_t DecisionTreeClassifier::depth() const {
    require_fitted();
    std::size_t deepest = 0;
    std::vector<std::pair<std::size_t, std::size_t>> stack{{0, 0}};
    while (!stack.empty()) {
        const auto [index, level] = stack.back();
        stack.pop_back();
        deepest = std::max(deepest, level);
        if (!nodes_[index].leaf) {
            stack.push_back({nodes_[index].left, level + 1});
            stack.push_back({nodes_[index].right, level + 1});
        }
    }
    return deepest;
}

std::size_t DecisionTreeClassifier::leaf_count() const {
    require_fitted();
    std::size_t leaves = 0;
    std::vector<std::size_t> stack{0};
    while (!stack.empty()) {
        const std::size_t index = stack.back();
        stack.pop_back();
        if (nodes_[index].leaf) {
            ++leaves;
            continue;
        }
        stack.push_back(nodes_[index].left);
        stack.push_back(nodes_[index].right);
    }
    return leaves;
}

} // namespace ml_scratch
