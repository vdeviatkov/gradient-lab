#include "ml_scratch/softmax_regression.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {

double SoftmaxRegression::accuracy(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

ConfusionCounts SoftmaxRegression::confusion_matrix(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    ConfusionCounts confusion(class_count(), std::vector<std::size_t>(class_count(), 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict(sample.features)];
    }
    return confusion;
}

MulticlassMetrics SoftmaxRegression::evaluate(const LabeledDataset& dataset) const {
    return multiclass_metrics(confusion_matrix(dataset));
}

} // namespace ml_scratch
