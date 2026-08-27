#include "ml_scratch/logic_gates.hpp"
#include "ml_scratch/xor_mlp.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>

int main() {
    constexpr std::uint32_t seed = 7;
    constexpr std::size_t max_epochs = 10'000;
    constexpr double target_loss = 0.02;

    const ml_scratch::LogicDataset dataset =
        ml_scratch::logic_gate_dataset(ml_scratch::LogicGate::xor_gate);
    ml_scratch::XorMlp model{4, 0.5, seed};
    const ml_scratch::XorMlpTrainingResult result = model.fit(dataset, max_epochs, target_loss);

    if (!result.converged || model.accuracy(dataset) != 1.0) {
        std::cerr << "XOR MLP failed to meet its convergence criteria\n";
        return 1;
    }

    std::cout << "C++ two-layer XOR MLP experiment (seed=" << seed << ")\n"
              << "XOR: learned in " << result.epochs << " epochs"
              << ", binary cross-entropy=" << std::fixed << std::setprecision(6)
              << result.loss_per_epoch.back() << '\n';
    for (const auto& [features, target] : dataset) {
        std::cout << "  (" << features[0] << ", " << features[1] << ") -> "
                  << model.predict(features) << " (p=" << model.predict_probability(features)
                  << ", target=" << target << ")\n";
    }
    return 0;
}
