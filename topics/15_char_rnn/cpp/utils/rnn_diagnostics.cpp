#include "ml_scratch/rnn.hpp"

namespace ml_scratch {

std::vector<double> CharRnn::gradient_reach(const TokenSequence& ids) const {
    validate_ids(ids);
    const std::size_t steps = ids.size() - 1;
    const State initial = initial_state();
    Workspace workspace;
    State running = initial;
    forward(ids, 0, steps, running, workspace);
    std::vector<double> flat(parameters_.size(), 0.0);
    std::vector<double> reach;
    backward(ids, 0, steps, initial, 1.0, true, flat, workspace, &reach);
    return reach;
}

} // namespace ml_scratch
