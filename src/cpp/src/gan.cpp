#include "ml_scratch/gan.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

double sigmoid(const double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + std::exp(-value));
    }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

// log(1 + e^v) without overflow. Written in terms of the logit z, the discriminator's
// cross-entropy toward target t is softplus(z) - t z.
double softplus(const double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double binary_cross_entropy(const double logit, const double target) {
    return softplus(logit) - target * logit;
}

// A standard normal deviate by Box-Muller, from two raw engine draws. Uniform values are taken
// from the engine directly rather than through a standard distribution, whose implementation is
// not specified bit-for-bit across libraries.
double standard_normal(std::mt19937& engine) {
    constexpr double range = 4294967296.0; // std::mt19937::max() + 1
    // The first draw is shifted into (0, 1] so the logarithm is finite.
    const double first = (static_cast<double>(engine()) + 1.0) / range;
    const double second = static_cast<double>(engine()) / range;
    constexpr double two_pi = 6.283185307179586;
    return std::sqrt(-2.0 * std::log(first)) * std::cos(two_pi * second);
}

void validate_training_config(const GanTrainingConfig& config) {
    if (!std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("learning_rate must be finite and positive");
    }
    if (config.epochs == 0) {
        throw std::invalid_argument("epochs must be positive");
    }
    if (config.batch_size == 0) {
        throw std::invalid_argument("batch_size must be positive");
    }
    if (!std::isfinite(config.real_target) || config.real_target <= 0.0 ||
        config.real_target > 1.0) {
        throw std::invalid_argument("real_target must be finite and in the interval (0, 1]");
    }
}

void require_sample_at_a_time(const std::vector<DenseLayer>& layers, const char* name) {
    for (const DenseLayer& layer : layers) {
        if (layer.normalization == Normalization::batch) {
            throw std::invalid_argument(std::string{name} +
                                        " cannot use batch normalization: adversarial training "
                                        "runs the networks sample at a time");
        }
        if (layer.dropout_rate > 0.0) {
            throw std::invalid_argument(std::string{name} +
                                        " cannot use dropout: it never fires on the "
                                        "sample-at-a-time path, so it would silently be a no-op");
        }
    }
}

} // namespace

const char* generator_loss_name(const GeneratorLoss loss) {
    switch (loss) {
    case GeneratorLoss::non_saturating:
        return "non-saturating";
    case GeneratorLoss::minimax:
        return "minimax";
    }
    return "";
}

ConditionalGan::ConditionalGan(const std::size_t noise_size, const std::size_t class_count,
                               std::vector<DenseLayer> generator,
                               std::vector<DenseLayer> discriminator, const std::uint32_t seed)
    : ConditionalGan{noise_size, class_count,
                     [&] {
                         if (noise_size == 0) {
                             throw std::invalid_argument("noise_size must be positive");
                         }
                         if (class_count == 0) {
                             throw std::invalid_argument("class_count must be positive");
                         }
                         if (generator.empty() || discriminator.empty()) {
                             throw std::invalid_argument(
                                 "both networks need at least one layer");
                         }
                         require_sample_at_a_time(generator, "the generator");
                         require_sample_at_a_time(discriminator, "the discriminator");
                         // The generator's own loss is never evaluated; mean squared error is
                         // the one that allows any output activation.
                         return FeedForwardNetwork{std::move(generator), Loss::mean_squared_error,
                                                   seed};
                     }(),
                     // A distinct seed keeps the two networks from starting as scaled copies of
                     // each other's first layer.
                     FeedForwardNetwork{std::move(discriminator), Loss::binary_cross_entropy,
                                        seed + 1}} {}

ConditionalGan::ConditionalGan(const std::size_t noise_size, const std::size_t class_count,
                               FeedForwardNetwork generator, FeedForwardNetwork discriminator)
    : noise_size_(noise_size), class_count_(class_count), generator_(std::move(generator)),
      discriminator_(std::move(discriminator)) {
    if (generator_.input_size() != noise_size_ + class_count_) {
        throw std::invalid_argument(
            "the generator must take noise_size + class_count inputs: the noise vector with the "
            "one-hot label appended");
    }
    if (discriminator_.input_size() != generator_.output_size() + class_count_) {
        throw std::invalid_argument(
            "the discriminator must take a generated sample with the one-hot label appended");
    }
    if (discriminator_.output_size() != 1) {
        throw std::invalid_argument("the discriminator must produce a single logit");
    }
    if (discriminator_.loss_function() != Loss::binary_cross_entropy) {
        throw std::invalid_argument("the discriminator must be trained with binary cross-entropy");
    }
}

void ConditionalGan::validate_label(const std::size_t label) const {
    if (label >= class_count_) {
        throw std::invalid_argument("label is outside the class count");
    }
}

void ConditionalGan::validate_noise(const std::vector<double>& noise) const {
    if (noise.size() != noise_size_) {
        throw std::invalid_argument("noise size does not match the generator");
    }
    for (const double value : noise) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("noise must be finite");
        }
    }
}

void ConditionalGan::validate_sample(const std::vector<double>& sample) const {
    if (sample.size() != sample_size()) {
        throw std::invalid_argument("sample size does not match the generator's output");
    }
    for (const double value : sample) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("samples must be finite");
        }
    }
}

void ConditionalGan::assemble_generator_input(const std::vector<double>& noise,
                                              const std::size_t label,
                                              std::vector<double>& buffer) const {
    buffer.assign(noise_size_ + class_count_, 0.0);
    std::copy(noise.begin(), noise.end(), buffer.begin());
    buffer[noise_size_ + label] = 1.0;
}

void ConditionalGan::assemble_discriminator_input(const std::vector<double>& sample,
                                                  const std::size_t label,
                                                  std::vector<double>& buffer) const {
    buffer.assign(sample_size() + class_count_, 0.0);
    std::copy(sample.begin(), sample.end(), buffer.begin());
    buffer[sample_size() + label] = 1.0;
}

std::vector<double> ConditionalGan::sample_noise(std::mt19937& engine) const {
    std::vector<double> noise(noise_size_);
    for (double& value : noise) {
        value = standard_normal(engine);
    }
    return noise;
}

std::vector<double> ConditionalGan::generate(const std::vector<double>& noise,
                                             const std::size_t label) const {
    validate_noise(noise);
    validate_label(label);
    std::vector<double> input;
    assemble_generator_input(noise, label, input);
    return generator_.forward(input);
}

double ConditionalGan::discriminate(const std::vector<double>& sample,
                                    const std::size_t label) const {
    validate_sample(sample);
    validate_label(label);
    std::vector<double> input;
    assemble_discriminator_input(sample, label, input);
    return sigmoid(discriminator_.forward(input).front());
}

double ConditionalGan::forward_fake(const std::vector<double>& noise, const std::size_t label,
                                    Workspace& workspace) const {
    assemble_generator_input(noise, label, workspace.generator_input);
    const std::vector<double>& sample =
        generator_.forward(workspace.generator_input, workspace.generator);
    assemble_discriminator_input(sample, label, workspace.discriminator_input);
    return discriminator_.forward(workspace.discriminator_input, workspace.discriminator).front();
}

double ConditionalGan::accumulate_generator_gradient(const double logit, const GeneratorLoss loss,
                                                     const double scale, std::vector<double>* flat,
                                                     Workspace& workspace) const {
    // Both objectives are functions of the discriminator's logit z alone, so each is one scalar
    // dL/dz to push back through the discriminator. With p = sigmoid(z):
    //   non-saturating  L = -log p = softplus(-z),   dL/dz = p - 1
    //   minimax         L = log(1 - p) = -softplus(z), dL/dz = -p
    // The non-saturating gradient is largest when p is near zero, which is when the discriminator
    // rejects the fake; the minimax gradient is largest when p is near one, which is when the
    // generator has already won, and vanishes exactly when it is losing.
    const double probability = sigmoid(logit);
    double sample_loss = 0.0;
    double logit_gradient = 0.0;
    switch (loss) {
    case GeneratorLoss::non_saturating:
        sample_loss = softplus(-logit);
        logit_gradient = probability - 1.0;
        break;
    case GeneratorLoss::minimax:
        sample_loss = -softplus(logit);
        logit_gradient = -probability;
        break;
    }

    // The discriminator's parameters are left alone here — a null gradient buffer — and only its
    // input gradient is wanted. The label half of that gradient is dropped: the label was given,
    // not generated, so nothing upstream can move it.
    workspace.output_gradient.assign(1, logit_gradient);
    discriminator_.backpropagate(workspace.output_gradient, 1.0, nullptr, workspace.discriminator,
                                 &workspace.input_gradient);
    workspace.input_gradient.resize(sample_size());
    generator_.backpropagate(workspace.input_gradient, scale, flat, workspace.generator);
    return sample_loss;
}

double ConditionalGan::generator_loss(const FeatureMatrix& noise,
                                      const std::vector<std::size_t>& labels,
                                      const GeneratorLoss loss) const {
    if (noise.empty() || noise.size() != labels.size()) {
        throw std::invalid_argument("noise and labels must be non-empty and equal in number");
    }
    Workspace workspace;
    double total = 0.0;
    for (std::size_t index = 0; index < noise.size(); ++index) {
        validate_noise(noise[index]);
        validate_label(labels[index]);
        const double logit = forward_fake(noise[index], labels[index], workspace);
        total += loss == GeneratorLoss::non_saturating ? softplus(-logit) : -softplus(logit);
    }
    return total / static_cast<double>(noise.size());
}

std::vector<double> ConditionalGan::generator_gradient(const FeatureMatrix& noise,
                                                       const std::vector<std::size_t>& labels,
                                                       const GeneratorLoss loss) const {
    if (noise.empty() || noise.size() != labels.size()) {
        throw std::invalid_argument("noise and labels must be non-empty and equal in number");
    }
    Workspace workspace;
    std::vector<double> flat(generator_.parameter_count(), 0.0);
    const double scale = 1.0 / static_cast<double>(noise.size());
    for (std::size_t index = 0; index < noise.size(); ++index) {
        validate_noise(noise[index]);
        validate_label(labels[index]);
        const double logit = forward_fake(noise[index], labels[index], workspace);
        accumulate_generator_gradient(logit, loss, scale, &flat, workspace);
    }
    return flat;
}

double ConditionalGan::discriminator_loss(const LabeledDataset& real, const FeatureMatrix& noise,
                                          const std::vector<std::size_t>& fake_labels,
                                          const double real_target) const {
    if (real.empty() || noise.empty() || noise.size() != fake_labels.size()) {
        throw std::invalid_argument("real samples, noise, and fake labels must all be supplied");
    }
    Workspace workspace;
    double real_total = 0.0;
    for (const LabeledSample& sample : real) {
        validate_sample(sample.features);
        validate_label(sample.label);
        assemble_discriminator_input(sample.features, sample.label, workspace.discriminator_input);
        const double logit =
            discriminator_.forward(workspace.discriminator_input, workspace.discriminator).front();
        real_total += binary_cross_entropy(logit, real_target);
    }
    double fake_total = 0.0;
    for (std::size_t index = 0; index < noise.size(); ++index) {
        validate_noise(noise[index]);
        validate_label(fake_labels[index]);
        fake_total += binary_cross_entropy(forward_fake(noise[index], fake_labels[index], workspace),
                                           0.0);
    }
    return real_total / static_cast<double>(real.size()) +
           fake_total / static_cast<double>(noise.size());
}

std::vector<double> ConditionalGan::discriminator_gradient(
    const LabeledDataset& real, const FeatureMatrix& noise,
    const std::vector<std::size_t>& fake_labels, const double real_target) const {
    if (real.empty() || noise.empty() || noise.size() != fake_labels.size()) {
        throw std::invalid_argument("real samples, noise, and fake labels must all be supplied");
    }
    Workspace workspace;
    std::vector<double> flat(discriminator_.parameter_count(), 0.0);
    const std::vector<double> real_targets{real_target};
    const std::vector<double> fake_targets{0.0};

    const double real_scale = 1.0 / static_cast<double>(real.size());
    for (const LabeledSample& sample : real) {
        validate_sample(sample.features);
        validate_label(sample.label);
        assemble_discriminator_input(sample.features, sample.label, workspace.discriminator_input);
        discriminator_.forward(workspace.discriminator_input, workspace.discriminator);
        discriminator_.backpropagate_loss(real_targets, real_scale, &flat, workspace.discriminator);
    }
    const double fake_scale = 1.0 / static_cast<double>(noise.size());
    for (std::size_t index = 0; index < noise.size(); ++index) {
        validate_noise(noise[index]);
        validate_label(fake_labels[index]);
        forward_fake(noise[index], fake_labels[index], workspace);
        discriminator_.backpropagate_loss(fake_targets, fake_scale, &flat, workspace.discriminator);
    }
    return flat;
}

GanTrainingResult ConditionalGan::fit(const LabeledDataset& real, const GanTrainingConfig& config) {
    validate_training_config(config);
    const DatasetShape shape = validate_labeled_dataset(real);
    if (shape.feature_count != sample_size()) {
        throw std::invalid_argument("feature count does not match the generator's output size");
    }
    if (shape.class_count > class_count_) {
        throw std::invalid_argument("the dataset has more classes than the model");
    }

    std::vector<std::size_t> order(real.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 random_engine{config.seed};
    const std::size_t batch_size = std::min(config.batch_size, real.size());

    Optimizer discriminator_optimizer{config.optimizer, discriminator_.parameter_count()};
    Optimizer generator_optimizer{config.optimizer, generator_.parameter_count()};
    std::vector<double> discriminator_gradient_buffer(discriminator_.parameter_count(), 0.0);
    std::vector<double> generator_gradient_buffer(generator_.parameter_count(), 0.0);
    std::vector<double> update;
    Workspace workspace;
    // One noise vector per batch position, drawn fresh for every batch and for each player.
    FeatureMatrix noise(batch_size, std::vector<double>(noise_size_, 0.0));
    const std::vector<double> real_targets{config.real_target};
    const std::vector<double> fake_targets{0.0};

    GanTrainingResult result;
    for (std::size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        if (config.shuffle && batch_size < real.size()) {
            std::shuffle(order.begin(), order.end(), random_engine);
        }

        double discriminator_loss_total = 0.0;
        double generator_loss_total = 0.0;
        double real_score_total = 0.0;
        double fake_score_total = 0.0;
        double gradient_norm_total = 0.0;
        std::size_t batches = 0;

        for (std::size_t begin = 0; begin < real.size(); begin += batch_size) {
            const std::size_t end = std::min(begin + batch_size, real.size());
            const std::size_t count = end - begin;
            const double scale = 1.0 / static_cast<double>(count);

            // ---- Discriminator step: real samples toward real_target, fakes toward zero. ----
            // The fakes come from the generator as it currently stands and are treated as
            // constants: the gradient buffer is the discriminator's alone, and nothing flows
            // back into the generator here.
            std::fill(discriminator_gradient_buffer.begin(), discriminator_gradient_buffer.end(),
                      0.0);
            double batch_discriminator_loss = 0.0;
            for (std::size_t position = 0; position < count; ++position) {
                const LabeledSample& sample = real[order[begin + position]];
                assemble_discriminator_input(sample.features, sample.label,
                                             workspace.discriminator_input);
                const double logit =
                    discriminator_.forward(workspace.discriminator_input, workspace.discriminator)
                        .front();
                batch_discriminator_loss += scale * binary_cross_entropy(logit, config.real_target);
                real_score_total += scale * sigmoid(logit);
                discriminator_.backpropagate_loss(real_targets, scale,
                                                  &discriminator_gradient_buffer,
                                                  workspace.discriminator);

                for (double& value : noise[position]) {
                    value = standard_normal(random_engine);
                }
                const double fake_logit = forward_fake(noise[position], sample.label, workspace);
                batch_discriminator_loss += scale * binary_cross_entropy(fake_logit, 0.0);
                fake_score_total += scale * sigmoid(fake_logit);
                discriminator_.backpropagate_loss(fake_targets, scale,
                                                  &discriminator_gradient_buffer,
                                                  workspace.discriminator);
            }
            discriminator_optimizer.compute_update(discriminator_gradient_buffer,
                                                   config.learning_rate, update);
            discriminator_.subtract_update(update);

            // ---- Generator step: fresh noise, the same labels, through the updated critic. ----
            std::fill(generator_gradient_buffer.begin(), generator_gradient_buffer.end(), 0.0);
            double batch_generator_loss = 0.0;
            for (std::size_t position = 0; position < count; ++position) {
                const std::size_t label = real[order[begin + position]].label;
                for (double& value : noise[position]) {
                    value = standard_normal(random_engine);
                }
                const double logit = forward_fake(noise[position], label, workspace);
                batch_generator_loss +=
                    scale * accumulate_generator_gradient(logit, config.generator_loss, scale,
                                                          &generator_gradient_buffer, workspace);
            }
            double squared_norm = 0.0;
            for (const double value : generator_gradient_buffer) {
                squared_norm += value * value;
            }
            generator_optimizer.compute_update(generator_gradient_buffer, config.learning_rate,
                                               update);
            generator_.subtract_update(update);

            if (!std::isfinite(batch_discriminator_loss) || !std::isfinite(batch_generator_loss)) {
                throw std::runtime_error("adversarial training diverged to a non-finite loss");
            }
            discriminator_loss_total += batch_discriminator_loss;
            generator_loss_total += batch_generator_loss;
            gradient_norm_total += std::sqrt(squared_norm);
            ++batches;
        }

        const auto batch_count = static_cast<double>(batches);
        result.epochs = epoch;
        result.discriminator_loss_per_epoch.push_back(discriminator_loss_total / batch_count);
        result.generator_loss_per_epoch.push_back(generator_loss_total / batch_count);
        result.real_score_per_epoch.push_back(real_score_total / batch_count);
        result.fake_score_per_epoch.push_back(fake_score_total / batch_count);
        result.generator_gradient_norm_per_epoch.push_back(gradient_norm_total / batch_count);
    }
    return result;
}

void ConditionalGan::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    stream << "ml_scratch_conditional_gan 1\n"
           << "noise " << noise_size_ << '\n'
           << "classes " << class_count_ << '\n';
    generator_.save(stream, path);
    discriminator_.save(stream, path);
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

ConditionalGan ConditionalGan::load(const std::string& path) {
    std::ifstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint: " + path);
    }
    const auto expect = [&stream, &path](const std::string& keyword) {
        std::string token;
        if (!(stream >> token) || token != keyword) {
            throw std::runtime_error("malformed checkpoint " + path + ": expected " + keyword);
        }
    };
    const auto read_size = [&stream, &path] {
        long long value = 0;
        if (!(stream >> value) || value < 0) {
            throw std::runtime_error("malformed checkpoint " + path + ": expected a size");
        }
        return static_cast<std::size_t>(value);
    };

    expect("ml_scratch_conditional_gan");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("noise");
    const std::size_t noise_size = read_size();
    expect("classes");
    const std::size_t class_count = read_size();
    if (noise_size == 0 || class_count == 0) {
        throw std::runtime_error("checkpoint declares an empty noise or class size: " + path);
    }
    FeedForwardNetwork generator = FeedForwardNetwork::load(stream, path);
    FeedForwardNetwork discriminator = FeedForwardNetwork::load(stream, path);
    // The private constructor re-checks that the two networks fit together.
    return ConditionalGan{noise_size, class_count, std::move(generator), std::move(discriminator)};
}

} // namespace ml_scratch
