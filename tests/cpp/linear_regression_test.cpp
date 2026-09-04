#include "ml_scratch/linear_regression.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

void require_near(const double actual, const double expected, const double tolerance,
                  const std::string_view message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void require_invalid_argument(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

ml_scratch::RegressionDataset make_exact_dataset() {
    ml_scratch::RegressionDataset dataset;
    for (const double first : {-1.0, 0.0, 1.0}) {
        for (const double second : {-1.0, 0.0, 1.0}) {
            dataset.push_back({{first, second}, 1.0 + 2.0 * first - 3.0 * second});
        }
    }
    return dataset;
}

void require_exact_line(const ml_scratch::LinearRegression& model, const double tolerance) {
    require_near(model.bias(), 1.0, tolerance, "incorrect fitted bias");
    require_near(model.weights()[0], 2.0, tolerance, "incorrect first fitted weight");
    require_near(model.weights()[1], -3.0, tolerance, "incorrect second fitted weight");
}

void test_closed_form_solution_and_metrics() {
    const auto dataset = make_exact_dataset();
    ml_scratch::LinearRegression model{2};
    model.fit_closed_form(dataset);

    require_exact_line(model, 1e-12);
    require_near(model.predict({0.5, -0.5}), 3.5, 1e-12, "incorrect prediction");
    require_near(model.mean_squared_error(dataset), 0.0, 1e-24, "non-zero exact-fit MSE");
    require_near(model.r_squared(dataset), 1.0, 1e-12, "incorrect exact-fit R-squared");
}

void test_gradient_optimizers_agree_with_closed_form() {
    const auto dataset = make_exact_dataset();
    for (const auto optimizer : {ml_scratch::RegressionOptimizer::batch_gradient_descent,
                                 ml_scratch::RegressionOptimizer::stochastic_gradient_descent,
                                 ml_scratch::RegressionOptimizer::mini_batch_gradient_descent}) {
        ml_scratch::LinearRegression model{2};
        ml_scratch::RegressionTrainingConfig config;
        config.optimizer = optimizer;
        config.learning_rate =
            optimizer == ml_scratch::RegressionOptimizer::batch_gradient_descent ? 0.2 : 0.05;
        config.max_epochs = 2'000;
        config.batch_size = 3;
        config.seed = 31415;
        config.target_mse = 1e-12;

        const auto result = model.fit(dataset, config);
        require(result.converged, "gradient optimizer did not reach its target MSE");
        require(result.epochs == result.mse_per_epoch.size(), "loss history has wrong size");
        require(result.mse_per_epoch.back() <= config.target_mse,
                "reported convergence above target MSE");
        require_exact_line(model, 2e-6);
    }
}

void test_seed_reproducibility() {
    const auto dataset = make_exact_dataset();
    ml_scratch::RegressionTrainingConfig config;
    config.optimizer = ml_scratch::RegressionOptimizer::mini_batch_gradient_descent;
    config.learning_rate = 0.05;
    config.max_epochs = 25;
    config.batch_size = 2;
    config.seed = 2718;

    ml_scratch::LinearRegression first{2};
    ml_scratch::LinearRegression second{2};
    const auto first_result = first.fit(dataset, config);
    const auto second_result = second.fit(dataset, config);

    require(first_result == second_result, "same seed produced different loss histories");
    require(first.weights() == second.weights(), "same seed produced different weights");
    require(first.bias() == second.bias(), "same seed produced different biases");
}

void test_input_validation() {
    require_invalid_argument([] { const ml_scratch::LinearRegression model{0}; },
                             "zero feature count was accepted");

    ml_scratch::LinearRegression model{2};
    require_invalid_argument([&] { model.fit_closed_form({}); }, "empty dataset was accepted");
    require_invalid_argument([&] { static_cast<void>(model.predict({1.0})); },
                             "wrong prediction shape was accepted");
    require_invalid_argument([&] { model.fit_closed_form({{{1.0}, 2.0}}); },
                             "wrong training shape was accepted");
    require_invalid_argument(
        [&] { model.fit_closed_form({{{1.0, 2.0}, std::numeric_limits<double>::infinity()}}); },
        "non-finite target was accepted");
    require_invalid_argument([&] { model.fit_closed_form({{{1.0, 2.0}, 1.0}, {{2.0, 4.0}, 2.0}}); },
                             "singular design matrix was accepted");

    const auto dataset = make_exact_dataset();
    ml_scratch::RegressionTrainingConfig config;
    config.learning_rate = 0.0;
    require_invalid_argument([&] { model.fit(dataset, config); },
                             "zero learning rate was accepted");
    config.learning_rate = 0.1;
    config.max_epochs = 0;
    require_invalid_argument([&] { model.fit(dataset, config); }, "zero epoch budget was accepted");
    config.max_epochs = 10;
    config.optimizer = ml_scratch::RegressionOptimizer::mini_batch_gradient_descent;
    config.batch_size = 0;
    require_invalid_argument([&] { model.fit(dataset, config); },
                             "zero mini-batch size was accepted");

    const ml_scratch::RegressionDataset constant_targets{{{0.0, 0.0}, 1.0}, {{1.0, 1.0}, 1.0}};
    require_invalid_argument([&] { static_cast<void>(model.r_squared(constant_targets)); },
                             "R-squared accepted constant targets");
}

} // namespace

int main() {
    try {
        test_closed_form_solution_and_metrics();
        test_gradient_optimizers_agree_with_closed_form();
        test_seed_reproducibility();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
