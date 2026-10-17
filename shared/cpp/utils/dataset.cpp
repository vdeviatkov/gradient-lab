#include "ml_scratch/dataset.hpp"
#include <stdexcept>

namespace ml_scratch {

std::vector<std::size_t> class_counts(const LabeledDataset& dataset,
                                      const std::size_t class_count) {
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.class_count > class_count) {
        throw std::invalid_argument("dataset contains a label outside the class count");
    }

    std::vector<std::size_t> counts(class_count, 0);
    for (const auto& sample : dataset) {
        ++counts[sample.label];
    }
    return counts;
}

SupervisedDataset to_one_hot(const LabeledDataset& dataset, const std::size_t class_count) {
    const DatasetShape shape = validate_labeled_dataset(dataset);
    if (shape.class_count > class_count) {
        throw std::invalid_argument("dataset contains a label outside the class count");
    }
    if (class_count == 0) {
        throw std::invalid_argument("class_count must be positive");
    }

    SupervisedDataset encoded;
    encoded.reserve(dataset.size());
    for (const auto& sample : dataset) {
        std::vector<double> targets(class_count, 0.0);
        targets[sample.label] = 1.0;
        encoded.push_back({sample.features, std::move(targets)});
    }
    return encoded;
}

} // namespace ml_scratch
