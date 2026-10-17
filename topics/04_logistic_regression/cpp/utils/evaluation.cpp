#include "ml_scratch/logistic_regression.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {

ClassificationMetrics BinaryLogisticRegression::evaluate(const BinaryClassificationDataset& dataset,
                                                         const double threshold) const {
    validate_dataset(dataset);
    std::vector<int> targets;
    std::vector<int> predictions;
    targets.reserve(dataset.size());
    predictions.reserve(dataset.size());
    for (const auto& sample : dataset) {
        targets.push_back(sample.target);
        predictions.push_back(predict(sample.features, threshold));
    }
    return classification_metrics(targets, predictions);
}

} // namespace ml_scratch
