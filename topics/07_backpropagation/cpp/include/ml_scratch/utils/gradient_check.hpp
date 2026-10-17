#pragma once

#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/rnn.hpp"
#include "ml_scratch/transformer.hpp"

namespace ml_scratch {

struct GradientCheckResult {
    // The standard gradient-check ratio, ||a - n|| / (||a|| + ||n||) in the Euclidean norm.
    // Normalizing by the whole gradient's magnitude is what makes it robust: a per-element ratio
    // divides by a single entry, so any parameter whose gradient is near zero reports a large
    // relative error no matter how well the two gradients agree there. This drives `passed`.
    //
    // The ratio has one degeneracy of its own: when the entire gradient vanishes, as it does at a
    // minimum, there is nothing left to normalize against and round-off alone can push it past
    // any tolerance. Check gradients at initialization or mid-training, and read the absolute
    // figures below when the gradient itself is near zero.
    double relative_error;
    // Largest elementwise |analytic - numerical|, and the largest per-element relative error with
    // a floor on the denominator. Both are diagnostics: the elementwise figures locate a bug,
    // while the norm ratio decides whether there is one.
    double max_absolute_error;
    double max_elementwise_relative_error;
    // Index into the flattened parameter vector holding the largest absolute disagreement.
    std::size_t worst_parameter;
    double analytic_value;
    double numerical_value;
    bool passed;

    bool operator==(const GradientCheckResult&) const = default;
};

// Central differences perturb a copy, leaving the original model unchanged.
[[nodiscard]] std::vector<double> numerical_gradient(const FeedForwardNetwork& network,
                                                     const SupervisedDataset& dataset,
                                                     double epsilon = 1e-5);
[[nodiscard]] std::vector<double> numerical_gradient(const CharRnn& network,
                                                     const TokenSequence& ids,
                                                     const CharRnn::State& state = {},
                                                     double epsilon = 1e-5);
[[nodiscard]] std::vector<double>
numerical_gradient(const CharTransformer& network, const TokenSequence& ids, double epsilon = 1e-5);

// Compares analytic and numerical gradients. The verdict uses the norm ratio described in
// GradientCheckResult; the elementwise figures are reported alongside it for diagnosis.
[[nodiscard]] GradientCheckResult check_gradient(const std::vector<double>& analytic,
                                                 const std::vector<double>& numerical,
                                                 double tolerance = 1e-7);

// Convenience overload that computes both gradients for the network first.
[[nodiscard]] GradientCheckResult check_gradient(const FeedForwardNetwork& network,
                                                 const SupervisedDataset& dataset,
                                                 double epsilon = 1e-5, double tolerance = 1e-7);

// Verifies the training-mode gradient over one batch, which is the only way to check a network
// that normalizes across the batch: its loss is a function of the whole batch jointly, so the
// numerical difference has to perturb that same joint function.
[[nodiscard]] GradientCheckResult check_batch_gradient(FeedForwardNetwork& network,
                                                       const LabeledDataset& batch,
                                                       double epsilon = 1e-5,
                                                       double tolerance = 1e-7);

} // namespace ml_scratch
