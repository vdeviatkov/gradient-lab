#include "ml_scratch/linear_regression.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

void require_finite(const double value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

} // namespace

LinearRegression::LinearRegression(const std::size_t feature_count) : weights_(feature_count, 0.0) {
    if (feature_count == 0) {
        throw std::invalid_argument("feature_count must be positive");
    }
}

void LinearRegression::validate_dataset(const RegressionDataset& dataset) const {
    if (dataset.empty()) {
        throw std::invalid_argument("dataset must not be empty");
    }
    for (const auto& sample : dataset) {
        if (sample.features.size() != feature_count()) {
            throw std::invalid_argument("sample feature count does not match the model");
        }
        require_finite(sample.target, "targets must be finite");
        for (const double feature : sample.features) {
            require_finite(feature, "features must be finite");
        }
    }
}

double LinearRegression::predict(const std::vector<double>& features) const {
    if (features.size() != feature_count()) {
        throw std::invalid_argument("feature count does not match the model");
    }

    double prediction = bias_;
    for (std::size_t feature = 0; feature < feature_count(); ++feature) {
        require_finite(features[feature], "features must be finite");
        prediction += weights_[feature] * features[feature];
    }
    return prediction;
}

double LinearRegression::mean_squared_error(const RegressionDataset& dataset) const {
    validate_dataset(dataset);
    double squared_error_sum = 0.0;
    for (const auto& sample : dataset) {
        const double residual = predict(sample.features) - sample.target;
        squared_error_sum += residual * residual;
    }
    return squared_error_sum / static_cast<double>(dataset.size());
}

void LinearRegression::set_parameters(const std::vector<double>& weights, const double bias) {
    if (weights.size() != feature_count()) {
        throw std::invalid_argument("parameter feature count does not match the model");
    }
    require_finite(bias, "bias must be finite");
    for (const double weight : weights) {
        require_finite(weight, "weights must be finite");
    }
    weights_ = weights;
    bias_ = bias;
}

RegressionTrainingResult LinearRegression::fit(const RegressionDataset& dataset,
                                               const RegressionTrainingConfig& config) {
    validate_dataset(dataset);
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.max_epochs == 0) {
        throw std::invalid_argument("max_epochs must be positive");
    }
    if (!std::isfinite(config.target_mse) || config.target_mse < 0.0) {
        throw std::invalid_argument("target_mse must be finite and non-negative");
    }
    if (config.optimizer == RegressionOptimizer::mini_batch_gradient_descent &&
        config.batch_size == 0) {
        throw std::invalid_argument("batch_size must be positive for mini-batch training");
    }

    std::fill(weights_.begin(), weights_.end(), 0.0);
    bias_ = 0.0;
    std::vector<std::size_t> order(dataset.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    std::vector<double> history;
    history.reserve(config.max_epochs);

    const std::size_t effective_batch_size = [&]() {
        switch (config.optimizer) {
        case RegressionOptimizer::batch_gradient_descent:
            return dataset.size();
        case RegressionOptimizer::stochastic_gradient_descent:
            return std::size_t{1};
        case RegressionOptimizer::mini_batch_gradient_descent:
            return std::min(config.batch_size, dataset.size());
        }
        throw std::invalid_argument("unknown regression optimizer");
    }();

    for (std::size_t epoch = 1; epoch <= config.max_epochs; ++epoch) {
        if (config.shuffle && config.optimizer != RegressionOptimizer::batch_gradient_descent) {
            std::shuffle(order.begin(), order.end(), random_engine);
        }

        for (std::size_t begin = 0; begin < dataset.size(); begin += effective_batch_size) {
            const std::size_t end = std::min(begin + effective_batch_size, dataset.size());
            const double inverse_batch_size = 1.0 / static_cast<double>(end - begin);
            std::vector<double> weight_gradients(feature_count(), 0.0);
            double bias_gradient = 0.0;

            for (std::size_t position = begin; position < end; ++position) {
                const RegressionSample& sample = dataset[order[position]];
                const double residual = predict(sample.features) - sample.target;
                bias_gradient += residual;
                for (std::size_t feature = 0; feature < feature_count(); ++feature) {
                    weight_gradients[feature] += residual * sample.features[feature];
                }
            }

            bias_ -= config.learning_rate * bias_gradient * inverse_batch_size;
            for (std::size_t feature = 0; feature < feature_count(); ++feature) {
                weights_[feature] -=
                    config.learning_rate * weight_gradients[feature] * inverse_batch_size;
            }
        }

        const double mse = mean_squared_error(dataset);
        if (!std::isfinite(mse)) {
            throw std::runtime_error("training diverged to a non-finite loss");
        }
        history.push_back(mse);
        if (config.target_mse > 0.0 && mse <= config.target_mse) {
            return {epoch, true, std::move(history)};
        }
    }

    return {config.max_epochs, false, std::move(history)};
}

} // namespace ml_scratch
