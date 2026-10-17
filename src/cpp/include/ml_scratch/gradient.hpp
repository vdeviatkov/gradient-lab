#pragma once

#include <vector>

namespace ml_scratch {

// Scales `gradient` down to Euclidean norm `max_norm` if it is larger, and returns the norm it
// had before. A gradient at or below the threshold is left exactly as it was; an infinite
// threshold only measures.
double clip_gradient(std::vector<double>& gradient, double max_norm);

} // namespace ml_scratch
