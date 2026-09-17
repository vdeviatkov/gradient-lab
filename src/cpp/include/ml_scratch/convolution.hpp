#pragma once

#include "ml_scratch/dataset.hpp"
#include "ml_scratch/neural_network.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ml_scratch {

// Activations flow through the network as flat vectors with an explicit shape, laid out
// channel-major: index(c, y, x) = (c * height + y) * width + x.
struct TensorShape {
    std::size_t channels{1};
    std::size_t height{1};
    std::size_t width{1};

    [[nodiscard]] std::size_t size() const noexcept { return channels * height * width; }
    [[nodiscard]] std::size_t index(const std::size_t channel, const std::size_t row,
                                    const std::size_t column) const noexcept {
        return (channel * height + row) * width + column;
    }

    bool operator==(const TensorShape&) const = default;
};

enum class ConvLayerKind {
    convolution,
    max_pooling,
    average_pooling,
    // Reinterprets a tensor as a vector. It holds no parameters and changes no values.
    flatten,
    dense,
};

struct ConvLayerSpec {
    ConvLayerKind kind{ConvLayerKind::flatten};
    Activation activation{Activation::identity};
    // Convolution.
    std::size_t filters{0};
    std::size_t kernel_size{0};
    std::size_t stride{1};
    std::size_t padding{0};
    // Pooling.
    std::size_t pool_size{0};
    std::size_t pool_stride{0};
    // Dense.
    std::size_t units{0};
    // An identity skip: the layer's input is added to its pre-activation. Requires the input and
    // output shapes to match, so the filter count must equal the input channel count and the
    // stride and padding must preserve the spatial size.
    bool residual{false};

    bool operator==(const ConvLayerSpec&) const = default;
};

// Readable constructors for a layer stack.
[[nodiscard]] ConvLayerSpec convolution(std::size_t filters, std::size_t kernel_size,
                                        Activation activation, std::size_t stride = 1,
                                        std::size_t padding = 0);
// A convolution that adds its input to its pre-activation. The shape must be preserved, which for
// an odd kernel means stride 1 and padding (kernel_size - 1) / 2.
[[nodiscard]] ConvLayerSpec residual_convolution(std::size_t filters, std::size_t kernel_size,
                                                 Activation activation);
// A stride of zero means "same as the window", the usual non-overlapping pooling.
[[nodiscard]] ConvLayerSpec max_pooling(std::size_t pool_size, std::size_t stride = 0);
[[nodiscard]] ConvLayerSpec average_pooling(std::size_t pool_size, std::size_t stride = 0);
[[nodiscard]] ConvLayerSpec flatten();
[[nodiscard]] ConvLayerSpec dense(std::size_t units, Activation activation);

// A convolutional classifier with explicitly derived gradients.
//
// It is a separate class from FeedForwardNetwork rather than an extension of it because its
// activations are shaped rather than flat and its layers are of several kinds. It shares that
// network's activations, losses, optimizers, and flat-parameter convention, so the same gradient
// checker verifies both.
//
// Parameters flatten layer by layer: a convolution contributes its kernels in
// [filter][input channel][row][column] order followed by one bias per filter, a dense layer its
// weight rows followed by its biases, and pooling and flatten contribute nothing.
class ConvolutionalNetwork {
  public:
    ConvolutionalNetwork(TensorShape input_shape, std::vector<ConvLayerSpec> layers, Loss loss,
                         std::uint32_t seed = 0);

    // Raw final-layer values; logits under the cross-entropy losses.
    [[nodiscard]] std::vector<double> forward(const std::vector<double>& input) const;
    [[nodiscard]] std::vector<double> predict(const std::vector<double>& input) const;
    [[nodiscard]] std::size_t predict_class(const std::vector<double>& input) const;

    [[nodiscard]] double loss(const LabeledDataset& dataset) const;
    [[nodiscard]] double accuracy(const LabeledDataset& dataset) const;
    [[nodiscard]] std::vector<std::vector<std::size_t>>
    confusion_matrix(const LabeledDataset& dataset) const;
    [[nodiscard]] std::vector<double> gradient(const LabeledDataset& dataset) const;

    NetworkTrainingResult fit(const LabeledDataset& dataset,
                              const NetworkTrainingConfig& config = {});
    NetworkTrainingResult fit(const LabeledDataset& dataset, const LabeledDataset& validation,
                              const NetworkTrainingConfig& config);

    [[nodiscard]] std::vector<double> parameters() const;
    void set_parameters(const std::vector<double>& values);
    void save(const std::string& path) const;
    [[nodiscard]] static ConvolutionalNetwork load(const std::string& path);

    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }
    [[nodiscard]] const std::vector<ConvLayerSpec>& layers() const noexcept { return layers_; }
    [[nodiscard]] TensorShape input_shape() const noexcept { return shapes_.front(); }
    [[nodiscard]] TensorShape output_shape() const noexcept { return shapes_.back(); }
    // shapes()[i] is the shape entering layer i; shapes().back() is the network's output.
    [[nodiscard]] const std::vector<TensorShape>& shapes() const noexcept { return shapes_; }

  private:
    struct LayerParameters {
        // Convolution kernels, flattened as [filter][channel][row][column]; empty for the rest.
        std::vector<double> weights;
        std::vector<double> biases;
    };

    struct Cache {
        // activations[0] is the input; activations[i + 1] is layer i's output.
        std::vector<std::vector<double>> activations;
        std::vector<std::vector<double>> pre_activations;
        // For max pooling, the input index each output value came from.
        std::vector<std::vector<std::size_t>> argmax;
    };

    void forward_into(const std::vector<double>& input, Cache& cache) const;
    void accumulate_gradient(const std::vector<double>& features, std::size_t label, double scale,
                             std::vector<double>& flat, Cache& cache, std::vector<double>& delta,
                             std::vector<double>& previous) const;
    void subtract_update(const std::vector<double>& update);
    void validate_dataset(const LabeledDataset& dataset) const;

    std::vector<ConvLayerSpec> layers_;
    Loss loss_;
    // shapes_[i] is the input shape of layer i, so shapes_ holds layers_.size() + 1 entries.
    std::vector<TensorShape> shapes_;
    std::vector<LayerParameters> parameters_;
    std::vector<std::size_t> layer_offsets_;
    std::size_t parameter_count_{0};
};

} // namespace ml_scratch
