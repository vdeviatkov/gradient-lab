#pragma once

#include <array>
#include <string_view>

namespace ml_scratch {

using LogicInput = std::array<double, 2>;

struct LogicExample {
    LogicInput features;
    int target;

    bool operator==(const LogicExample&) const = default;
};

using LogicDataset = std::array<LogicExample, 4>;

enum class LogicGate { and_gate, or_gate, xor_gate };

[[nodiscard]] LogicDataset logic_gate_dataset(LogicGate gate);
[[nodiscard]] std::string_view logic_gate_name(LogicGate gate);

} // namespace ml_scratch
