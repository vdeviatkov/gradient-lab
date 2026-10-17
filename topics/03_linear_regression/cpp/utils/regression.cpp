#include "ml_scratch/utils/regression.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

namespace {

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

void validate_regression_data(const LinearRegression& model, const RegressionDataset& dataset) {
    if (dataset.empty())
        throw std::invalid_argument("dataset must not be empty");
    for (const auto& sample : dataset) {
        if (sample.features.size() != model.feature_count())
            throw std::invalid_argument("sample feature count does not match the model");
        if (!std::isfinite(sample.target))
            throw std::invalid_argument("targets must be finite");
        for (double feature : sample.features) {
            if (!std::isfinite(feature))
                throw std::invalid_argument("features must be finite");
        }
    }
}

} // namespace

double r_squared(const LinearRegression& model, const RegressionDataset& dataset) {
    validate_regression_data(model, dataset);
    const double target_sum = std::accumulate(
        dataset.begin(), dataset.end(), 0.0,
        [](const double sum, const RegressionSample& sample) { return sum + sample.target; });
    const double target_mean = target_sum / static_cast<double>(dataset.size());

    double residual_sum_of_squares = 0.0;
    double total_sum_of_squares = 0.0;
    for (const auto& sample : dataset) {
        const double residual = model.predict(sample.features) - sample.target;
        const double centered_target = sample.target - target_mean;
        residual_sum_of_squares += residual * residual;
        total_sum_of_squares += centered_target * centered_target;
    }
    if (total_sum_of_squares == 0.0) {
        throw std::invalid_argument("R-squared is undefined when all targets are equal");
    }
    return 1.0 - residual_sum_of_squares / total_sum_of_squares;
}

void fit_closed_form(LinearRegression& model, const RegressionDataset& dataset) {
    validate_regression_data(model, dataset);
    const std::size_t parameter_count = model.feature_count() + 1;
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
    model.set_parameters({parameters.begin() + 1, parameters.end()}, parameters.front());
}

} // namespace ml_scratch
