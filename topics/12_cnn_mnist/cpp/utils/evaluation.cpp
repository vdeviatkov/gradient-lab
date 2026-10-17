#include "ml_scratch/convolution.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {

double ConvolutionalNetwork::accuracy(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        if (predict_class(sample.features) == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.size());
}

std::vector<std::vector<std::size_t>>
ConvolutionalNetwork::confusion_matrix(const LabeledDataset& dataset) const {
    validate_dataset(dataset);
    const std::size_t classes = output_shape().size();
    std::vector<std::vector<std::size_t>> confusion(classes, std::vector<std::size_t>(classes, 0));
    for (const auto& sample : dataset) {
        ++confusion[sample.label][predict_class(sample.features)];
    }
    return confusion;
}

} // namespace ml_scratch
