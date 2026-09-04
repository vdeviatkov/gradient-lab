#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml_scratch {

struct RegressionSample {
    std::vector<double> features;
    double target;

    bool operator==(const RegressionSample&) const = default;
};

using RegressionDataset = std::vector<RegressionSample>;

enum class RegressionOptimizer {
    batch_gradient_descent,
    stochastic_gradient_descent,
    mini_batch_gradient_descent,
};

struct RegressionTrainingConfig {
    RegressionOptimizer optimizer{RegressionOptimizer::batch_gradient_descent};
    double learning_rate{0.01};
    std::size_t max_epochs{1'000};
    std::size_t batch_size{4};
    bool shuffle{true};
    std::uint32_t seed{0};
    double target_mse{0.0};
};

struct RegressionTrainingResult {
    std::size_t epochs;
    bool converged;
    std::vector<double> mse_per_epoch;

    bool operator==(const RegressionTrainingResult&) const = default;
};

class LinearRegression {
  public:
    explicit LinearRegression(std::size_t feature_count);

    [[nodiscard]] double predict(const std::vector<double>& features) const;
    [[nodiscard]] double mean_squared_error(const RegressionDataset& dataset) const;
    [[nodiscard]] double r_squared(const RegressionDataset& dataset) const;

    // Solves (X^T X) theta = X^T y with an explicit, pivoted Gaussian elimination.
    void fit_closed_form(const RegressionDataset& dataset);
    RegressionTrainingResult fit(const RegressionDataset& dataset,
                                 const RegressionTrainingConfig& config = {});

    [[nodiscard]] const std::vector<double>& weights() const noexcept { return weights_; }
    [[nodiscard]] double bias() const noexcept { return bias_; }
    [[nodiscard]] std::size_t feature_count() const noexcept { return weights_.size(); }

  private:
    void validate_dataset(const RegressionDataset& dataset) const;

    std::vector<double> weights_;
    double bias_{0.0};
};

} // namespace ml_scratch
