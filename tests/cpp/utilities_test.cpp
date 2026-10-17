#include "ml_scratch/utils/gradient_check.hpp"
#include "ml_scratch/utils/regression.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("utility contract failed");
}

template <typename Function> void require_invalid(Function function) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected);
}

void test_regression_reference() {
    using namespace ml_scratch;
    const RegressionDataset data{{{0.0}, 1.0}, {{1.0}, 2.0}, {{2.0}, 2.0}};
    LinearRegression model{1};
    fit_closed_form(model, data);
    require(std::abs(model.weights()[0] - 0.5) < 1e-12);
    require(std::abs(model.bias() - 7.0 / 6.0) < 1e-12);
    require(std::abs(model.mean_squared_error(data) - 1.0 / 18.0) < 1e-12);
    require(std::abs(r_squared(model, data) - 0.75) < 1e-12);
    const auto weights = model.weights();
    const auto bias = model.bias();
    require_invalid([&] { fit_closed_form(model, {{{1.0}, 2.0}, {{1.0}, 3.0}}); });
    require(model.weights() == weights && model.bias() == bias);
    model.set_parameters({0.0}, 5.0 / 3.0);
    require(std::abs(r_squared(model, data)) < 1e-12);
    model.set_parameters({0.0}, 0.0);
    require(r_squared(model, data) < 0.0); // Worse than the target-mean baseline is allowed.
}

void test_numerical_gradients_leave_models_unchanged() {
    using namespace ml_scratch;
    FeedForwardNetwork dense{{{1, 1, Activation::identity}}, Loss::mean_squared_error, 17};
    const SupervisedDataset data{{{0.2}, {0.7}}, {{-0.3}, {0.1}}};
    const auto dense_parameters = dense.parameters();
    require(check_gradient(dense.gradient(data), numerical_gradient(dense, data)).passed);
    require(dense.parameters() == dense_parameters);
    CharRnn rnn{2, 2, 17};
    const TokenSequence ids{0, 1, 0};
    const CharRnn::State state{0.2, -0.1};
    const auto rnn_parameters = rnn.parameters();
    require(check_gradient(rnn.gradient(ids, state), numerical_gradient(rnn, ids, state)).passed);
    require(rnn.parameters() == rnn_parameters && state == CharRnn::State({0.2, -0.1}));
    CharTransformer transformer{
        {.vocabulary_size = 2, .context_length = 2, .model_size = 2, .heads = 1, .layers = 1}, 17};
    const auto transformer_parameters = transformer.parameters();
    require(check_gradient(transformer.gradient(ids), numerical_gradient(transformer, ids)).passed);
    require(transformer.parameters() == transformer_parameters);
    for (double epsilon : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN()}) {
        require_invalid([&] { static_cast<void>(numerical_gradient(dense, data, epsilon)); });
        require_invalid([&] { static_cast<void>(numerical_gradient(rnn, ids, state, epsilon)); });
        require_invalid([&] { static_cast<void>(numerical_gradient(transformer, ids, epsilon)); });
    }
    require(dense.parameters() == dense_parameters && rnn.parameters() == rnn_parameters &&
            transformer.parameters() == transformer_parameters);
}

void test_batch_check_preserves_state() {
    using namespace ml_scratch;
    FeedForwardNetwork network{{{1, 2, Activation::hyperbolic_tangent, Normalization::batch},
                                {2, 2, Activation::identity}},
                               Loss::softmax_cross_entropy,
                               17};
    const LabeledDataset data{{{-1.0}, 0}, {{0.3}, 1}, {{1.0}, 1}};
    const auto parameters = network.parameters();
    const auto statistics = network.running_statistics();
    require(check_batch_gradient(network, data).passed);
    require(network.parameters() == parameters && network.running_statistics() == statistics);
}
} // namespace

int main() {
    try {
        test_regression_reference();
        test_numerical_gradients_leave_models_unchanged();
        test_batch_check_preserves_state();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
