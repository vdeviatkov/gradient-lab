#pragma once

#include "ml_scratch/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml_scratch {

enum class Activation {
    identity,
    sigmoid,
    hyperbolic_tangent,
    // Not differentiable at zero. Gradient checking can report a spurious mismatch for a
    // pre-activation that lands on the kink; see docs/mathematics/backpropagation.md.
    rectified_linear,
};

enum class Loss {
    // Mean over outputs and samples of (prediction - target)^2.
    mean_squared_error,
    // Sigmoid applied inside the loss, so the final layer must produce raw logits.
    binary_cross_entropy,
    // Softmax applied inside the loss, so the final layer must produce raw logits.
    softmax_cross_entropy,
};

struct DenseLayer {
    std::size_t input_size;
    std::size_t output_size;
    Activation activation;

    bool operator==(const DenseLayer&) const = default;
};

struct NetworkTrainingConfig {
    double learning_rate{0.1};
    std::size_t max_epochs{1'000};
    // Zero means one full-batch update per epoch.
    std::size_t batch_size{0};
    bool shuffle{true};
    std::uint32_t seed{0};
    double target_loss{0.0};
};

struct NetworkTrainingResult {
    std::size_t epochs;
    bool converged;
    std::vector<double> loss_per_epoch;

    bool operator==(const NetworkTrainingResult&) const = default;
};

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

// A fully connected feed-forward network with explicitly derived gradients.
//
// Parameters are exposed as one flat vector so that a caller — and the numerical gradient check —
// can treat the whole network as a single function of a parameter vector. The layout is layer by
// layer, and within a layer the weight matrix row by row (all weights feeding output unit 0, then
// unit 1, and so on) followed by that layer's biases.
class FeedForwardNetwork {
  public:
    FeedForwardNetwork(std::vector<DenseLayer> layers, Loss loss, std::uint32_t seed = 0);

    // Raw final-layer values. For the cross-entropy losses these are logits, not probabilities.
    [[nodiscard]] std::vector<double> forward(const std::vector<double>& input) const;
    // Applies the loss function's output transform: sigmoid, softmax, or identity.
    [[nodiscard]] std::vector<double> predict(const std::vector<double>& input) const;
    // Index of the largest predicted score. Ties resolve to the lowest index.
    [[nodiscard]] std::size_t predict_class(const std::vector<double>& input) const;

    [[nodiscard]] double loss(const SupervisedDataset& dataset) const;
    [[nodiscard]] double accuracy(const SupervisedDataset& dataset) const;

    // Gradient of the dataset loss with respect to every parameter, in the flat layout above.
    [[nodiscard]] std::vector<double> gradient(const SupervisedDataset& dataset) const;
    // The same gradient estimated by central differences, used only to verify `gradient`.
    // The default step sits near the measured optimum for doubles: larger steps are dominated by
    // the O(epsilon^2) truncation error, smaller ones by cancellation in (L+ - L-).
    [[nodiscard]] std::vector<double> numerical_gradient(const SupervisedDataset& dataset,
                                                         double epsilon = 1e-5) const;

    NetworkTrainingResult fit(const SupervisedDataset& dataset,
                              const NetworkTrainingConfig& config = {});

    [[nodiscard]] std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] const std::vector<DenseLayer>& layers() const noexcept { return layers_; }
    [[nodiscard]] Loss loss_function() const noexcept { return loss_; }
    [[nodiscard]] std::size_t input_size() const noexcept { return layers_.front().input_size; }
    [[nodiscard]] std::size_t output_size() const noexcept { return layers_.back().output_size; }

  private:
    struct LayerParameters {
        // weights[output][input]
        FeatureMatrix weights;
        std::vector<double> biases;
    };

    struct ForwardCache {
        // Pre-activations and activations for each layer; activations[0] is the input.
        std::vector<std::vector<double>> pre_activations;
        std::vector<std::vector<double>> activations;
    };

    [[nodiscard]] ForwardCache forward_cache(const std::vector<double>& input) const;
    [[nodiscard]] double sample_loss(const std::vector<double>& outputs,
                                     const std::vector<double>& targets) const;
    // dL/dz for the final layer, which the loss and its output transform determine together.
    [[nodiscard]] std::vector<double> output_delta(const std::vector<double>& pre_activations,
                                                   const std::vector<double>& outputs,
                                                   const std::vector<double>& targets) const;
    void validate_for_network(const SupervisedDataset& dataset) const;

    std::vector<DenseLayer> layers_;
    Loss loss_;
    std::vector<LayerParameters> parameters_;
    std::size_t parameter_count_{0};
};

// Compares analytic and numerical gradients. The verdict uses the norm ratio described in
// GradientCheckResult; the elementwise figures are reported alongside it for diagnosis.
[[nodiscard]] GradientCheckResult check_gradient(const std::vector<double>& analytic,
                                                 const std::vector<double>& numerical,
                                                 double tolerance = 1e-7);

// Convenience overload that computes both gradients for the network first.
[[nodiscard]] GradientCheckResult check_gradient(const FeedForwardNetwork& network,
                                                 const SupervisedDataset& dataset,
                                                 double epsilon = 1e-5, double tolerance = 1e-7);

} // namespace ml_scratch
