#include "ml_scratch/gan.hpp"

namespace ml_scratch {

const char* generator_loss_name(const GeneratorLoss loss) {
    switch (loss) {
    case GeneratorLoss::non_saturating:
        return "non-saturating";
    case GeneratorLoss::minimax:
        return "minimax";
    }
    return "";
}

} // namespace ml_scratch
