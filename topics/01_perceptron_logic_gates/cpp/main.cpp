#include "ml_scratch/logic_gates.hpp"
#include "ml_scratch/perceptron.hpp"

#include <array>
#include <cstdint>
#include <iostream>

int main() {
    constexpr std::uint32_t seed = 7;
    constexpr std::size_t max_epochs = 100;
    constexpr std::array gates{ml_scratch::LogicGate::and_gate, ml_scratch::LogicGate::or_gate,
                               ml_scratch::LogicGate::xor_gate};

    std::cout << "C++ perceptron logic-gate experiment (seed=" << seed << ")\n";
    for (const ml_scratch::LogicGate gate : gates) {
        const ml_scratch::LogicDataset dataset = ml_scratch::logic_gate_dataset(gate);
        ml_scratch::BinaryPerceptron model{0.1, seed};
        const ml_scratch::PerceptronTrainingResult result = model.fit(dataset, max_epochs);

        if (gate == ml_scratch::LogicGate::xor_gate) {
            if (result.converged) {
                std::cerr << "XOR was incorrectly reported as linearly separable\n";
                return 1;
            }
            std::cout << "XOR: not linearly separable; single-layer perceptron did not converge in "
                      << result.epochs << " epochs\n";
        } else {
            if (!result.converged || model.accuracy(dataset) != 1.0) {
                std::cerr << ml_scratch::logic_gate_name(gate) << " failed to converge\n";
                return 1;
            }
            std::cout << ml_scratch::logic_gate_name(gate) << ": learned in " << result.epochs
                      << " epochs\n";
        }

        std::cout << "  predictions/targets:";
        for (const auto& [features, target] : dataset) {
            std::cout << ' ' << model.predict(features) << '/' << target;
        }
        std::cout << '\n';
    }
    return 0;
}
