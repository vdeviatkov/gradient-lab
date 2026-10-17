#include "ml_scratch/transformer.hpp"
#include "transformer_workspace.hpp"

namespace ml_scratch {

std::vector<std::vector<std::vector<std::vector<double>>>>
CharTransformer::attention_weights(const TokenSequence& ids) const {
    validate_ids(ids, true);
    Workspace workspace;
    const std::size_t steps = ids.size();
    forward(ids, steps, workspace, false);
    std::vector<std::vector<std::vector<std::vector<double>>>> weights(config_.layers);
    for (std::size_t layer = 0; layer < config_.layers; ++layer) {
        weights[layer].resize(config_.heads);
        for (std::size_t head = 0; head < config_.heads; ++head) {
            weights[layer][head].assign(steps, std::vector<double>(steps, 0.0));
            const double* attention =
                workspace.blocks[layer].attention.data() + head * steps * steps;
            for (std::size_t i = 0; i < steps; ++i) {
                for (std::size_t j = 0; j <= i; ++j) {
                    weights[layer][head][i][j] = attention[i * steps + j];
                }
            }
        }
    }
    return weights;
}

} // namespace ml_scratch
