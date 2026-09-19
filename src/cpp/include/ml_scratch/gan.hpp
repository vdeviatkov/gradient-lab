#pragma once

#include "ml_scratch/dataset.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/optimizer.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ml_scratch {

enum class GeneratorLoss {
    // Minimizes -log D(G(z)). Its gradient is largest exactly when the discriminator rejects the
    // sample, which is when the generator most needs to move. The default, and what the original
    // paper recommends in practice.
    non_saturating,
    // Minimizes log(1 - D(G(z))), the generator's half of the minimax game as written. Its gradient
    // is proportional to D(G(z)), so it vanishes once the discriminator becomes confident — the
    // saturation the other loss is named for avoiding.
    minimax,
};

[[nodiscard]] const char* generator_loss_name(GeneratorLoss loss);

struct GanTrainingConfig {
    // Both networks share the update rule and learning rate. Adam with beta1 = 0.5 is the DCGAN
    // setting: a lower first-moment decay reacts faster to an opponent that keeps moving the
    // objective, where the usual 0.9 averages over a target that no longer exists.
    OptimizerConfig optimizer{.kind = OptimizerKind::adam, .beta1 = 0.5};
    double learning_rate{0.0002};
    std::size_t epochs{1};
    std::size_t batch_size{64};
    bool shuffle{true};
    std::uint32_t seed{0};
    GeneratorLoss generator_loss{GeneratorLoss::non_saturating};
    // The target the discriminator is trained toward on real samples. One is the plain objective;
    // a value just below it is one-sided label smoothing, which stops the discriminator from
    // driving its real-sample logits to infinity.
    double real_target{1.0};
};

struct GanTrainingResult {
    std::size_t epochs{0};
    // Mean over the epoch's batches of each player's objective, measured on the batch the update
    // was computed from.
    std::vector<double> discriminator_loss_per_epoch;
    std::vector<double> generator_loss_per_epoch;
    // Mean D(x) over real samples and mean D(G(z)) over fakes, as the discriminator saw them when
    // its update was computed. At the game's equilibrium both are 1/2.
    std::vector<double> real_score_per_epoch;
    std::vector<double> fake_score_per_epoch;
    // Euclidean norm of the generator's batch gradient, averaged over the epoch. This is what the
    // minimax loss lets collapse.
    std::vector<double> generator_gradient_norm_per_epoch;

    bool operator==(const GanTrainingResult&) const = default;
};

// A conditional generative adversarial network of two fully connected networks.
//
// The generator maps a noise vector concatenated with a one-hot class label to a sample, and the
// discriminator maps a sample concatenated with the same label to one logit, the log-odds that the
// pair came from the data rather than the generator. Conditioning both on the label is what lets a
// caller ask for a particular class. Neither network has a loss of its own in the usual sense: the
// discriminator is trained with binary cross-entropy against real-or-fake targets, and the
// generator is trained through the discriminator, by backpropagating the discriminator's verdict
// to its input and continuing into the generator that produced it.
class ConditionalGan {
  public:
    // `generator` must take noise_size + class_count inputs; `discriminator` must take
    // generator output size + class_count inputs and produce one linear output, since the loss
    // applies its own sigmoid. Both are trained sample at a time, so neither may use batch
    // normalization, and dropout is refused because it never fires on that path.
    ConditionalGan(std::size_t noise_size, std::size_t class_count,
                   std::vector<DenseLayer> generator, std::vector<DenseLayer> discriminator,
                   std::uint32_t seed = 0);

    // A vector of noise_size standard normal deviates, drawn from `engine` by Box-Muller. The
    // engine is the caller's so that a fixed noise set can be reused to watch one sample evolve.
    [[nodiscard]] std::vector<double> sample_noise(std::mt19937& engine) const;
    [[nodiscard]] std::vector<double> generate(const std::vector<double>& noise,
                                               std::size_t label) const;
    // Probability the discriminator assigns to (sample, label) having come from the data.
    [[nodiscard]] double discriminate(const std::vector<double>& sample, std::size_t label) const;

    // The generator's objective on the given noise and labels, averaged, and its gradient with
    // respect to the generator's parameters. The discriminator is treated as a fixed function.
    [[nodiscard]] double generator_loss(const FeatureMatrix& noise,
                                        const std::vector<std::size_t>& labels,
                                        GeneratorLoss loss = GeneratorLoss::non_saturating) const;
    [[nodiscard]] std::vector<double>
    generator_gradient(const FeatureMatrix& noise, const std::vector<std::size_t>& labels,
                       GeneratorLoss loss = GeneratorLoss::non_saturating) const;
    // The discriminator's objective on a set of real samples and a set of fakes generated from
    // the given noise and labels: mean cross-entropy toward real_target on the former plus mean
    // cross-entropy toward zero on the latter, and its gradient with respect to the
    // discriminator's parameters.
    [[nodiscard]] double discriminator_loss(const LabeledDataset& real, const FeatureMatrix& noise,
                                            const std::vector<std::size_t>& fake_labels,
                                            double real_target = 1.0) const;
    [[nodiscard]] std::vector<double>
    discriminator_gradient(const LabeledDataset& real, const FeatureMatrix& noise,
                           const std::vector<std::size_t>& fake_labels,
                           double real_target = 1.0) const;

    // Alternating mini-batch training: for each batch of real samples, one discriminator update
    // on the batch and an equal number of fakes, then one generator update on fresh noise for
    // the same labels. Labels are drawn from the real batch so the generator sees the data's
    // class balance.
    GanTrainingResult fit(const LabeledDataset& real, const GanTrainingConfig& config = {});

    // Both networks in one text checkpoint, using the feed-forward format twice.
    void save(const std::string& path) const;
    [[nodiscard]] static ConditionalGan load(const std::string& path);

    [[nodiscard]] const FeedForwardNetwork& generator() const noexcept { return generator_; }
    [[nodiscard]] FeedForwardNetwork& generator() noexcept { return generator_; }
    [[nodiscard]] const FeedForwardNetwork& discriminator() const noexcept {
        return discriminator_;
    }
    [[nodiscard]] FeedForwardNetwork& discriminator() noexcept { return discriminator_; }
    [[nodiscard]] std::size_t noise_size() const noexcept { return noise_size_; }
    [[nodiscard]] std::size_t class_count() const noexcept { return class_count_; }
    [[nodiscard]] std::size_t sample_size() const noexcept { return generator_.output_size(); }

  private:
    ConditionalGan(std::size_t noise_size, std::size_t class_count, FeedForwardNetwork generator,
                   FeedForwardNetwork discriminator);

    // Scratch for one training step, kept across steps so an epoch allocates nothing per sample.
    struct Workspace {
        FeedForwardNetwork::Workspace generator;
        FeedForwardNetwork::Workspace discriminator;
        std::vector<double> generator_input;     // [noise; one-hot label]
        std::vector<double> discriminator_input; // [sample; one-hot label]
        std::vector<double> input_gradient;      // dL/d(discriminator input)
        std::vector<double> output_gradient;     // dL/d(generator output)
    };

    void validate_label(std::size_t label) const;
    void validate_noise(const std::vector<double>& noise) const;
    void validate_sample(const std::vector<double>& sample) const;
    // Writes [noise; one-hot(label)] into the generator input buffer, and [sample; one-hot(label)]
    // into the discriminator's.
    void assemble_generator_input(const std::vector<double>& noise, std::size_t label,
                                  std::vector<double>& buffer) const;
    void assemble_discriminator_input(const std::vector<double>& sample, std::size_t label,
                                      std::vector<double>& buffer) const;
    // Forward pass through the generator into the workspace, and through the discriminator on
    // what it produced. Returns the discriminator's logit.
    double forward_fake(const std::vector<double>& noise, std::size_t label,
                        Workspace& workspace) const;
    // Adds scale * dL_G/dtheta_G for one sample to `flat` and returns the sample's loss. Must
    // follow forward_fake on the same workspace.
    double accumulate_generator_gradient(double logit, GeneratorLoss loss, double scale,
                                         std::vector<double>* flat, Workspace& workspace) const;

    std::size_t noise_size_;
    std::size_t class_count_;
    FeedForwardNetwork generator_;
    FeedForwardNetwork discriminator_;
};

} // namespace ml_scratch
