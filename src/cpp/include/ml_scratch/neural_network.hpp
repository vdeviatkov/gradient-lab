#pragma once

#include "ml_scratch/dataset.hpp"
#include "ml_scratch/optimizer.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <random>
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
    // max(v, slope * v): a rectifier whose negative side keeps a small fixed slope, so no unit can
    // stop passing gradient by landing in the flat region. A GAN's discriminator uses it because
    // the generator learns only through the gradient the discriminator passes back.
    leaky_rectified_linear,
};

// The negative-side slope of leaky_rectified_linear, the value the DCGAN paper uses.
inline constexpr double leaky_rectified_linear_slope = 0.2;

enum class Loss {
    // Mean over outputs and samples of (prediction - target)^2.
    mean_squared_error,
    // Sigmoid applied inside the loss, so the final layer must produce raw logits.
    binary_cross_entropy,
    // Softmax applied inside the loss, so the final layer must produce raw logits.
    softmax_cross_entropy,
};

enum class Normalization {
    none,
    // Standardizes each unit across the samples of a mini-batch, then rescales with a learned
    // gamma and beta. Training and inference differ: training uses the batch's own statistics
    // while inference uses a running average collected during training.
    batch,
    // Standardizes each sample across its own units. It involves no batch statistics, so training
    // and inference compute exactly the same function and the batch size is irrelevant.
    layer,
};

enum class Initialization {
    // He scaling for a rectified-linear layer, Glorot for every other activation. The default.
    automatic,
    glorot_uniform,
    he_uniform,
    // Uniform over [-scale, scale], with the scale supplied by InitializationConfig.
    fixed_uniform,
    // Every weight zero. Included so an experiment can measure why symmetry has to be broken.
    zeros,
};

struct InitializationConfig {
    Initialization kind{Initialization::automatic};
    // Used by fixed_uniform only.
    double scale{0.01};

    bool operator==(const InitializationConfig&) const = default;
};

struct DenseLayer {
    std::size_t input_size;
    std::size_t output_size;
    Activation activation;
    // Applied to the pre-activation, before the activation function.
    Normalization normalization{Normalization::none};
    // Probability of dropping each output during training, using inverted dropout so that no
    // rescaling is needed at inference. Zero disables it.
    double dropout_rate{0.0};
    // An identity skip connection: the layer's input is added to its pre-activation, so it
    // computes g(Wx + b + x) instead of g(Wx + b). Requires input_size == output_size, since
    // the two terms must have the same shape to be added. Addition before the activation matches
    // the placement in the original residual network.
    bool residual{false};

    bool operator==(const DenseLayer&) const = default;
};

// Penalties on the weights. Biases and normalization parameters are never penalized: they shift a
// response rather than scale it, so shrinking them does not reduce model complexity.
struct RegularizationConfig {
    double l1{0.0};
    double l2{0.0};

    bool operator==(const RegularizationConfig&) const = default;
};

// Stops training once validation loss has failed to improve for `patience` consecutive epochs, and
// restores the parameters from the best epoch. Requires a validation split; zero disables it.
struct EarlyStoppingConfig {
    std::size_t patience{0};
    // An epoch counts as an improvement only if it beats the best loss by more than this.
    double min_improvement{0.0};

    bool operator==(const EarlyStoppingConfig&) const = default;
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
    RegularizationConfig regularization{};
    EarlyStoppingConfig early_stopping{};
    // Decay of the running statistics batch normalization keeps for inference.
    double normalization_momentum{0.9};
};

struct NetworkTrainingResult {
    std::size_t epochs;
    bool converged;
    // The objective actually minimized: the data loss plus any weight penalty.
    std::vector<double> loss_per_epoch;
    // Populated only when a validation split was supplied.
    std::vector<double> validation_loss_per_epoch;
    // One-based epoch with the lowest validation loss, or zero without a validation split.
    std::size_t best_epoch{0};
    bool stopped_early{false};

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
    FeedForwardNetwork(std::vector<DenseLayer> layers, Loss loss, std::uint32_t seed = 0,
                       InitializationConfig initialization = {});

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
    // Overloads that measure validation loss each epoch and enable early stopping.
    NetworkTrainingResult fit(const SupervisedDataset& dataset, const SupervisedDataset& validation,
                              const NetworkTrainingConfig& config);
    NetworkTrainingResult fit(const LabeledDataset& dataset, const LabeledDataset& validation,
                              const NetworkTrainingConfig& config);

    // The weight penalty alone, for the given coefficients.
    [[nodiscard]] double weight_penalty(const RegularizationConfig& regularization) const;

    // Training-mode loss and gradient over a batch treated as one unit, using the batch's own
    // statistics wherever a layer normalizes across the batch. These exist so that gradient
    // checking can verify the path batch normalization actually trains through: `loss` and
    // `gradient` above run in inference mode, where the running statistics are constants and the
    // samples are independent. Neither updates the running statistics. Dropout never fires here,
    // since a stochastic mask would make the check meaningless.
    [[nodiscard]] double batch_training_loss(const LabeledDataset& batch);
    [[nodiscard]] std::vector<double> batch_training_gradient(const LabeledDataset& batch);
    [[nodiscard]] bool uses_batch_normalization() const noexcept;
    [[nodiscard]] bool uses_dropout() const noexcept;

    [[nodiscard]] std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);
    // Subtracts an already-scaled update from the parameters, walking the same flat layout. What
    // an Optimizer's compute_update produces goes straight in here, without a round trip through
    // parameters() and set_parameters().
    void subtract_update(const std::vector<double>& update);
    // Batch normalization's running mean and variance. They are state rather than parameters: no
    // gradient touches them, but inference depends on them, so checkpoints carry them.
    [[nodiscard]] std::vector<double> running_statistics() const;
    void set_running_statistics(const std::vector<double>& values);

    // Text checkpoints holding the architecture, the loss, and every parameter. Values are written
    // with 17 significant digits, which round-trips an IEEE-754 double exactly.
    void save(const std::string& path) const;
    [[nodiscard]] static FeedForwardNetwork load(const std::string& path);
    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] const std::vector<DenseLayer>& layers() const noexcept { return layers_; }
    [[nodiscard]] Loss loss_function() const noexcept { return loss_; }
    [[nodiscard]] std::size_t input_size() const noexcept { return layers_.front().input_size; }
    [[nodiscard]] std::size_t output_size() const noexcept { return layers_.back().output_size; }

    // The chain through one layer is s = Wa + b, n = standardize(s), y = scale*n + shift,
    // a' = g(y), out = a' * dropout_factor. Every intermediate is kept because the backward pass
    // needs it; without normalization or dropout, y = s and out = a'.
    struct ForwardCache {
        // activations[0] is the input; activations[l + 1] is layer l's output after dropout.
        std::vector<std::vector<double>> activations;
        std::vector<std::vector<double>> pre_activations;   // s
        std::vector<std::vector<double>> normalized;        // n
        std::vector<std::vector<double>> activation_inputs; // y
        std::vector<std::vector<double>> activation_values; // g(y), before dropout
        std::vector<std::vector<double>> dropout_factors;   // empty when the layer keeps everything
        std::vector<double> inverse_deviation;              // 1/sqrt(var + eps), layer norm only
    };

    // Reusable scratch space, so a training epoch does not allocate once per sample. A caller
    // driving the passes below keeps one per network and reuses it across samples.
    struct Workspace {
        ForwardCache cache;
        std::vector<double> delta;
        std::vector<double> previous_delta;
        std::vector<double> normalized;
    };

    // The sample-at-a-time passes, exposed so that a network can be one stage of a larger
    // computation: a generator is trained through a discriminator by running the discriminator's
    // backward pass to its input and feeding that gradient into the generator's. Both backward
    // passes consume the intermediates the most recent `forward` into the same workspace left
    // behind, so a forward must precede each of them. Batch normalization is applied in inference
    // mode and dropout never fires here.
    //
    // Runs the forward pass and returns the raw final-layer output, which stays valid until the
    // workspace is used again.
    const std::vector<double>& forward(const std::vector<double>& input,
                                       Workspace& workspace) const;
    // Backpropagates an externally supplied dL/d(output) — the gradient of some loss this network
    // knows nothing about, taken with respect to its final activations. Adds scale * dL/dtheta to
    // `flat` when it is non-null, and writes dL/d(input) to `input_gradient` when that is.
    void backpropagate(const std::vector<double>& output_gradient, double scale,
                       std::vector<double>* flat, Workspace& workspace,
                       std::vector<double>* input_gradient = nullptr) const;
    // The same, for the network's own loss against `targets`: the output delta is the one the
    // loss and its output transform define together, exactly as `gradient` computes it.
    void backpropagate_loss(const std::vector<double>& targets, double scale,
                            std::vector<double>* flat, Workspace& workspace,
                            std::vector<double>* input_gradient = nullptr) const;
    // Loss of one raw output against its targets under this network's loss function.
    [[nodiscard]] double sample_loss(const std::vector<double>& outputs,
                                     const std::vector<double>& targets) const;

    // The checkpoint format written to and read from an arbitrary stream, so that a model made of
    // several networks can keep them in one file. `description` names the stream in error messages.
    void save(std::ostream& stream, const std::string& description = "stream") const;
    [[nodiscard]] static FeedForwardNetwork load(std::istream& stream,
                                                 const std::string& description = "stream");

  private:
    struct LayerParameters {
        // weights[output][input]
        FeatureMatrix weights;
        std::vector<double> biases;
        // Normalization scale and shift, empty when the layer is not normalized.
        std::vector<double> scale;
        std::vector<double> shift;
        // Batch normalization's inference statistics, empty otherwise.
        std::vector<double> running_mean;
        std::vector<double> running_variance;
    };

    // Batch normalization couples the samples of a mini-batch, so its forward and backward passes
    // have to see the whole batch at once. Every other feature is per-sample, which is why the
    // sample-at-a-time path is kept for networks without it.
    struct BatchWorkspace {
        // [layer][sample][unit]
        std::vector<FeatureMatrix> pre_activations;
        std::vector<FeatureMatrix> normalized;
        std::vector<FeatureMatrix> activations;
        std::vector<FeatureMatrix> deltas;
        std::vector<std::vector<double>> batch_mean;
        std::vector<std::vector<double>> batch_inverse_deviation;
    };

    [[nodiscard]] ForwardCache forward_cache(const std::vector<double>& input) const;
    // `training` selects batch normalization's statistics source and whether dropout fires.
    // `dropout_engine` may be null when no layer drops units.
    void forward_into(const std::vector<double>& input, ForwardCache& cache, bool training = false,
                      std::mt19937* dropout_engine = nullptr) const;
    // One sample's backward pass, accumulated into `flat`. `targets` may be null, in which case
    // `label` selects the one-hot target under softmax_cross_entropy.
    void accumulate_gradient(const std::vector<double>& features,
                             const std::vector<double>* targets, std::size_t label, double scale,
                             std::vector<double>& flat, Workspace& workspace, bool training = false,
                             std::mt19937* dropout_engine = nullptr) const;
    void require_classification() const;
    void add_penalty_gradient(const RegularizationConfig& regularization,
                              std::vector<double>& flat) const;
    // Batch-at-a-time forward and backward, used only when a layer normalizes across the batch.
    // Returns the mean loss over the batch. `update_running` controls whether batch
    // normalization's inference statistics absorb this batch.
    double accumulate_batch_gradient(const FeatureMatrix& inputs,
                                     const std::vector<const std::vector<double>*>& targets,
                                     const std::vector<std::size_t>& labels, double scale,
                                     std::vector<double>& flat, BatchWorkspace& workspace,
                                     bool training, std::mt19937* dropout_engine,
                                     bool update_running = true);
    [[nodiscard]] std::size_t normalization_offset(std::size_t layer) const;
    [[nodiscard]] std::size_t validate_labeled(const LabeledDataset& dataset) const;
    // The backward pass shared by every sample-at-a-time gradient: walks the cached forward pass
    // in `workspace` from the output delta already in `workspace.delta` down to the first layer.
    // `flat` may be null to skip the parameter gradient, and `input_gradient` may be null to stop
    // at the first layer.
    void backward_from_delta(double scale, std::vector<double>* flat, Workspace& workspace,
                             std::vector<double>* input_gradient) const;
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
    // Decay used when updating batch normalization's running statistics.
    double normalization_momentum_{0.9};
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

// Verifies the training-mode gradient over one batch, which is the only way to check a network
// that normalizes across the batch: its loss is a function of the whole batch jointly, so the
// numerical difference has to perturb that same joint function.
[[nodiscard]] GradientCheckResult check_batch_gradient(FeedForwardNetwork& network,
                                                       const LabeledDataset& batch,
                                                       double epsilon = 1e-5,
                                                       double tolerance = 1e-7);

} // namespace ml_scratch
