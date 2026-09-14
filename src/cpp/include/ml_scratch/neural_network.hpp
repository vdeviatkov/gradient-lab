#pragma once

#include "ml_scratch/dataset.hpp"
#include "ml_scratch/optimizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
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
    // The update rule. Its default is plain gradient descent, so the learning rate below is the
    // only knob unless a different kind is chosen.
    OptimizerConfig optimizer{};
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

    // Classification overloads that take class indices instead of one-hot targets. They compute
    // exactly the same thing as the SupervisedDataset versions applied to to_one_hot(dataset), but
    // never materialize the target vectors, which is what makes a 60000 x 784 dataset practical.
    // They require softmax_cross_entropy and one output per class.
    [[nodiscard]] double loss(const LabeledDataset& dataset) const;
    [[nodiscard]] double accuracy(const LabeledDataset& dataset) const;
    // confusion[actual][predicted]
    [[nodiscard]] std::vector<std::vector<std::size_t>>
    confusion_matrix(const LabeledDataset& dataset) const;
    [[nodiscard]] std::vector<double> gradient(const LabeledDataset& dataset) const;

    // Gradient of the dataset loss with respect to every parameter, in the flat layout above.
    [[nodiscard]] std::vector<double> gradient(const SupervisedDataset& dataset) const;
    // The same gradient estimated by central differences, used only to verify `gradient`.
    // The default step sits near the measured optimum for doubles: larger steps are dominated by
    // the O(epsilon^2) truncation error, smaller ones by cancellation in (L+ - L-).
    [[nodiscard]] std::vector<double> numerical_gradient(const SupervisedDataset& dataset,
                                                         double epsilon = 1e-5) const;

    NetworkTrainingResult fit(const SupervisedDataset& dataset,
                              const NetworkTrainingConfig& config = {});
    NetworkTrainingResult fit(const LabeledDataset& dataset,
                              const NetworkTrainingConfig& config = {});

    [[nodiscard]] std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);

    // Text checkpoints holding the architecture, the loss, and every parameter. Values are written
    // with 17 significant digits, which round-trips an IEEE-754 double exactly.
    void save(const std::string& path) const;
    [[nodiscard]] static FeedForwardNetwork load(const std::string& path);
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

    // Reusable scratch space, so a training epoch does not allocate once per sample.
    struct Workspace {
        ForwardCache cache;
        std::vector<double> delta;
        std::vector<double> previous_delta;
    };

    [[nodiscard]] ForwardCache forward_cache(const std::vector<double>& input) const;
    void forward_into(const std::vector<double>& input, ForwardCache& cache) const;
    // One sample's backward pass, accumulated into `flat`. `targets` may be null, in which case
    // `label` selects the one-hot target under softmax_cross_entropy.
    void accumulate_gradient(const std::vector<double>& features,
                             const std::vector<double>* targets, std::size_t label, double scale,
                             std::vector<double>& flat, Workspace& workspace) const;
    // Subtracts an already-scaled update from the parameters, walking the same flat layout.
    void subtract_update(const std::vector<double>& update);
    void require_classification() const;
    [[nodiscard]] std::size_t validate_labeled(const LabeledDataset& dataset) const;
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
    // Offset of each layer's block inside the flat parameter vector.
    std::vector<std::size_t> layer_offsets_;
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
