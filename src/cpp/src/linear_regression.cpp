#include "ml_scratch/linear_regression.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

std::vector<double> solve_linear_system(std::vector<std::vector<double>> matrix,
                                        std::vector<double> right_hand_side) {
    const std::size_t size = matrix.size();
    for (std::size_t column = 0; column < size; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < size; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) {
                pivot = row;
            }
        }

        double row_scale = 0.0;
        for (const double value : matrix[pivot]) {
            row_scale = std::max(row_scale, std::abs(value));
        }
        const double pivot_tolerance =
            std::numeric_limits<double>::epsilon() * static_cast<double>(size) * row_scale;
        if (row_scale == 0.0 || std::abs(matrix[pivot][column]) <= pivot_tolerance) {
            throw std::invalid_argument(
                "closed-form solution requires linearly independent design columns");
        }

        std::swap(matrix[column], matrix[pivot]);
        std::swap(right_hand_side[column], right_hand_side[pivot]);

        for (std::size_t row = column + 1; row < size; ++row) {
            const double multiplier = matrix[row][column] / matrix[column][column];
            matrix[row][column] = 0.0;
            for (std::size_t next = column + 1; next < size; ++next) {
                matrix[row][next] -= multiplier * matrix[column][next];
            }
            right_hand_side[row] -= multiplier * right_hand_side[column];
        }
    }

    std::vector<double> solution(size, 0.0);
    for (std::size_t reverse = size; reverse > 0; --reverse) {
        const std::size_t row = reverse - 1;
        double value = right_hand_side[row];
        for (std::size_t column = row + 1; column < size; ++column) {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }
    return solution;
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

double LinearRegression::r_squared(const RegressionDataset& dataset) const {
    validate_dataset(dataset);
    const double target_sum = std::accumulate(
        dataset.begin(), dataset.end(), 0.0,
        [](const double sum, const RegressionSample& sample) { return sum + sample.target; });
    const double target_mean = target_sum / static_cast<double>(dataset.size());

    double residual_sum_of_squares = 0.0;
    double total_sum_of_squares = 0.0;
    for (const auto& sample : dataset) {
        const double residual = predict(sample.features) - sample.target;
        const double centered_target = sample.target - target_mean;
        residual_sum_of_squares += residual * residual;
        total_sum_of_squares += centered_target * centered_target;
    }
    if (total_sum_of_squares == 0.0) {
        throw std::invalid_argument("R-squared is undefined when all targets are equal");
    }
    return 1.0 - residual_sum_of_squares / total_sum_of_squares;
}

void LinearRegression::fit_closed_form(const RegressionDataset& dataset) {
    validate_dataset(dataset);
    const std::size_t parameter_count = feature_count() + 1;
    std::vector<std::vector<double>> gram_matrix(parameter_count,
                                                 std::vector<double>(parameter_count, 0.0));
    std::vector<double> target_products(parameter_count, 0.0);

    for (const auto& sample : dataset) {
        std::vector<double> augmented_features;
        augmented_features.reserve(parameter_count);
        augmented_features.push_back(1.0);
        augmented_features.insert(augmented_features.end(), sample.features.begin(),
                                  sample.features.end());

        for (std::size_t row = 0; row < parameter_count; ++row) {
            target_products[row] += augmented_features[row] * sample.target;
            for (std::size_t column = 0; column < parameter_count; ++column) {
                gram_matrix[row][column] += augmented_features[row] * augmented_features[column];
            }
        }
    }

    const std::vector<double> parameters =
        solve_linear_system(std::move(gram_matrix), std::move(target_products));
    bias_ = parameters.front();
    std::copy(parameters.begin() + 1, parameters.end(), weights_.begin());
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
