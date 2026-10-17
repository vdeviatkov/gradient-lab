#include "ml_scratch/logic_gates.hpp"
#include "ml_scratch/perceptron.hpp"
#include "ml_scratch/xor_mlp.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

void test_datasets() {
    const auto and_data = ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::and_gate);
    const auto or_data = ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::or_gate);
    const auto xor_data = ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::xor_gate);

    constexpr std::array<int, 4> and_targets{0, 0, 0, 1};
    constexpr std::array<int, 4> or_targets{0, 1, 1, 1};
    constexpr std::array<int, 4> xor_targets{0, 1, 1, 0};
    for (std::size_t index = 0; index < and_data.size(); ++index) {
        require(and_data[index].target == and_targets[index], "incorrect AND truth table");
        require(or_data[index].target == or_targets[index], "incorrect OR truth table");
        require(xor_data[index].target == xor_targets[index], "incorrect XOR truth table");
    }
}

void test_perceptron_logic_gates() {
    for (const auto gate : {ml_scratch::LogicGate::and_gate, ml_scratch::LogicGate::or_gate}) {
        const auto dataset = ml_scratch::logic_gate_dataset(gate);
        ml_scratch::BinaryPerceptron model{0.1, 7};
        const auto result = model.fit(dataset, 100);
        require(result.converged, "linearly separable gate did not converge");
        require(result.mistakes_per_epoch.back() == 0, "final epoch contained mistakes");
        require(model.accuracy(dataset) == 1.0, "gate predictions are incorrect");
    }

    const auto xor_data = ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::xor_gate);
    ml_scratch::BinaryPerceptron xor_model{0.1, 7};
    const auto xor_result = xor_model.fit(xor_data, 100);
    require(!xor_result.converged, "single-layer perceptron cannot converge on XOR");
    require(xor_model.accuracy(xor_data) < 1.0, "single linear boundary cannot classify XOR");
}

void test_perceptron_reproducibility() {
    const auto dataset = ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::or_gate);
    ml_scratch::BinaryPerceptron first{0.1, 23};
    ml_scratch::BinaryPerceptron second{0.1, 23};

    const auto first_result = first.fit(dataset);
    const auto second_result = second.fit(dataset);
    require(first_result == second_result, "perceptron histories differ for the same seed");
    require(first.weights() == second.weights(), "perceptron weights differ for the same seed");
    require(first.bias() == second.bias(), "perceptron biases differ for the same seed");
}

void test_xor_mlp() {
    const auto dataset = ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::xor_gate);
    ml_scratch::XorMlp model{4, 0.5, 7};
    const auto result = model.fit(dataset, 10'000, 0.02);

    require(result.converged, "two-layer MLP did not converge on XOR");
    require(model.accuracy(dataset) == 1.0, "two-layer MLP predictions are incorrect");
    require(result.loss_per_epoch.back() <= 0.02, "two-layer MLP did not reach target loss");
    for (const auto& [features, target] : dataset) {
        require(model.predict(features) == target, "two-layer MLP missed an XOR example");
    }
}

void test_xor_mlp_reproducibility() {
    const auto dataset = ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::xor_gate);
    ml_scratch::XorMlp first{4, 0.5, 19};
    ml_scratch::XorMlp second{4, 0.5, 19};

    const auto first_result = first.fit(dataset, 10'000, 0.02);
    const auto second_result = second.fit(dataset, 10'000, 0.02);
    require(first_result == second_result, "MLP training histories differ for the same seed");
    require(first.parameters() == second.parameters(), "MLP parameters differ for the same seed");
}

void test_invalid_configuration() {
    bool rejected_hidden_units = false;
    try {
        const ml_scratch::XorMlp invalid_model{0, 0.5, 7};
    } catch (const std::invalid_argument&) {
        rejected_hidden_units = true;
    }
    require(rejected_hidden_units, "zero hidden units were accepted");

    bool rejected_learning_rate = false;
    try {
        const ml_scratch::BinaryPerceptron invalid_model{0.0, 7};
    } catch (const std::invalid_argument&) {
        rejected_learning_rate = true;
    }
    require(rejected_learning_rate, "zero learning rate was accepted");
}

} // namespace

int main() {
    try {
        test_datasets();
        test_perceptron_logic_gates();
        test_perceptron_reproducibility();
        test_xor_mlp();
        test_xor_mlp_reproducibility();
        test_invalid_configuration();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
