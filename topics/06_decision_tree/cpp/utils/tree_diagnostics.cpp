#include "ml_scratch/decision_tree.hpp"
#include <algorithm>
#include <utility>

namespace ml_scratch {

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
