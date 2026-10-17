#include "ml_scratch/neural_network.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {

double FeedForwardNetwork::accuracy(const SupervisedDataset& dataset) const {
    validate_for_network(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        const auto expected = static_cast<std::size_t>(
            std::distance(sample.targets.begin(),
                          std::max_element(sample.targets.begin(), sample.targets.end())));
        if (predict_class(sample.features) == expected) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

double FeedForwardNetwork::accuracy(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict_class(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

std::vector<std::vector<std::size_t>>
FeedForwardNetwork::confusion_matrix(const LabeledDataset& dataset) const {
    static_cast<void>(validate_labeled(dataset));

    std::vector<std::vector<std::size_t>> confusion(output_size(),
                                                    std::vector<std::size_t>(output_size(), 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict_class(sample.features)];
    }
    return confusion;
}

} // namespace ml_scratch
