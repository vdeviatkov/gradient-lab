#include "ml_scratch/linear_regression.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

constexpr std::uint32_t seed = 20260903;

const ml_scratch::RegressionDataset training_data{
    {{-2.0}, -2.5}, {{-1.5}, -1.5}, {{-1.0}, -0.5}, {{-0.5}, 0.5}, {{0.0}, 1.5},
    {{0.5}, 2.5},   {{1.0}, 3.5},   {{1.5}, 4.5},   {{2.0}, 5.5},
};

const ml_scratch::RegressionDataset test_data{
    {{-2.5}, -3.5},
    {{0.25}, 2.0},
    {{3.0}, 7.5},
};

struct OptimizerExperiment {
    std::string_view name;
    ml_scratch::RegressionOptimizer optimizer;
    double learning_rate;
    std::size_t batch_size;
};

} // namespace

int main() {
    ml_scratch::LinearRegression reference{1};
    reference.fit_closed_form(training_data);

    std::cout << "C++ linear regression experiment (seed=" << seed << ")\n"
              << std::fixed << std::setprecision(8)
              << "closed form: weight=" << reference.weights()[0] << ", bias=" << reference.bias()
              << ", train MSE=" << reference.mean_squared_error(training_data)
              << ", test MSE=" << reference.mean_squared_error(test_data) << '\n';

    constexpr OptimizerExperiment experiments[]{
        {"batch GD", ml_scratch::RegressionOptimizer::batch_gradient_descent, 0.10, 9},
        {"SGD", ml_scratch::RegressionOptimizer::stochastic_gradient_descent, 0.03, 1},
        {"mini-batch GD", ml_scratch::RegressionOptimizer::mini_batch_gradient_descent, 0.05, 3},
    };

    for (const auto& experiment : experiments) {
        ml_scratch::RegressionTrainingConfig config;
        config.optimizer = experiment.optimizer;
        config.learning_rate = experiment.learning_rate;
        config.max_epochs = 2'000;
        config.batch_size = experiment.batch_size;
        config.seed = seed;
        config.target_mse = 1e-12;

        ml_scratch::LinearRegression model{1};
        const auto result = model.fit(training_data, config);
        const double coefficient_error = std::abs(model.weights()[0] - reference.weights()[0]);
        const double bias_error = std::abs(model.bias() - reference.bias());
        if (!result.converged || coefficient_error > 2e-6 || bias_error > 2e-6) {
            std::cerr << experiment.name << " failed to agree with the closed-form solution\n";
            return 1;
        }

        std::cout << experiment.name << ": epochs=" << result.epochs
                  << ", weight=" << model.weights()[0] << ", bias=" << model.bias()
                  << ", train MSE=" << model.mean_squared_error(training_data)
                  << ", test MSE=" << model.mean_squared_error(test_data) << '\n';
    }
    return 0;
}
