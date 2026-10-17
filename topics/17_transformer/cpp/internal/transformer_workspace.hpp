#pragma once

#include "ml_scratch/transformer.hpp"

namespace ml_scratch {

struct CharTransformer::Workspace {
    // Per block, [position][dimension] unless noted.
    struct Block {
        std::vector<double> residual;        // the stream entering the block
        std::vector<double> norm1;           // standardized values
        std::vector<double> norm1_inverse;   // [position]
        std::vector<double> attention_input; // gain * norm1 + bias
        std::vector<double> query;
        std::vector<double> key;
        std::vector<double> value;
        std::vector<double> attention;       // [head][query][key]
        std::vector<double> context;         // concatenated heads
        std::vector<double> after_attention; // residual + context W_o^T
        std::vector<double> norm2;
        std::vector<double> norm2_inverse;
        std::vector<double> feed_forward_input;
        std::vector<double> hidden_pre; // [position][feed_forward]
        std::vector<double> hidden;     // relu of the above
    };
    std::vector<Block> blocks;
    std::vector<double> final_residual;
    std::vector<double> final_norm;
    std::vector<double> final_inverse;
    std::vector<double> final_output;
    std::vector<double> logits;        // [position][id]
    std::vector<double> probabilities; // [position][id]
    // Backward scratch.
    std::vector<double> d_residual;
    std::vector<double> d_after_attention;
    std::vector<double> d_input;
    std::vector<double> d_hidden;
    std::vector<double> d_context;
    std::vector<double> d_query;
    std::vector<double> d_key;
    std::vector<double> d_value;
    std::vector<double> d_scores;
    std::vector<double> scratch;
};

} // namespace ml_scratch
