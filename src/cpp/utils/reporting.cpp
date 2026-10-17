#include "ml_scratch/gan.hpp"
#include "ml_scratch/optimizer.hpp"
#include "ml_scratch/rnn.hpp"

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

const char* generator_loss_name(const GeneratorLoss loss) {
    switch (loss) {
    case GeneratorLoss::non_saturating:
        return "non-saturating";
    case GeneratorLoss::minimax:
        return "minimax";
    }
    return "";
}

const char* recurrent_cell_name(const RecurrentCell cell) {
    switch (cell) {
    case RecurrentCell::elman:
        return "elman";
    case RecurrentCell::lstm:
        return "lstm";
    case RecurrentCell::gru:
        return "gru";
    }
    return "";
}

} // namespace ml_scratch
