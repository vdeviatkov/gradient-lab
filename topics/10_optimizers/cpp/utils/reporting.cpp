#include "ml_scratch/optimizer.hpp"

namespace ml_scratch {

std::string_view optimizer_name(const OptimizerKind kind) {
    switch (kind) {
    case OptimizerKind::gradient_descent:
        return "gradient descent";
    case OptimizerKind::momentum:
        return "momentum";
    case OptimizerKind::nesterov_momentum:
        return "Nesterov momentum";
    case OptimizerKind::rmsprop:
        return "RMSProp";
    case OptimizerKind::adam:
        return "Adam";
    }
    return "unknown";
}

} // namespace ml_scratch
