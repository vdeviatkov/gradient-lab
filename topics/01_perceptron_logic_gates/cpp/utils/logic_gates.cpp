#include "ml_scratch/logic_gates.hpp"

#include <stdexcept>

namespace ml_scratch {

LogicDataset logic_gate_dataset(const LogicGate gate) {
    constexpr std::array<LogicInput, 4> inputs{{
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0},
    }};

    std::array<int, 4> targets{};
    switch (gate) {
    case LogicGate::and_gate:
        targets = {0, 0, 0, 1};
        break;
    case LogicGate::or_gate:
        targets = {0, 1, 1, 1};
        break;
    case LogicGate::xor_gate:
        targets = {0, 1, 1, 0};
        break;
    default:
        throw std::invalid_argument("unknown logic gate");
    }

    LogicDataset dataset{};
    for (std::size_t index = 0; index < dataset.size(); ++index) {
        dataset[index] = {inputs[index], targets[index]};
    }
    return dataset;
}

std::string_view logic_gate_name(const LogicGate gate) {
    switch (gate) {
    case LogicGate::and_gate:
        return "AND";
    case LogicGate::or_gate:
        return "OR";
    case LogicGate::xor_gate:
        return "XOR";
    default:
        throw std::invalid_argument("unknown logic gate");
    }
}

} // namespace ml_scratch
