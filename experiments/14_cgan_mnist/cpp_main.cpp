#include "ml_scratch/gan.hpp"
#include "ml_scratch/mnist.hpp"
#include "ml_scratch/neural_network.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260917;
constexpr std::size_t class_count = 10;
constexpr std::size_t train_count = 10'000;
constexpr std::size_t held_out_count = 2'000;

// The judge: a classifier trained on the same real images, used only to score generated ones.
constexpr std::size_t judge_hidden = 128;
constexpr std::size_t judge_epochs = 5;

// The generator and discriminator.
constexpr std::size_t noise_size = 64;
constexpr std::size_t hidden = 256;
constexpr std::size_t gan_epochs = 30;
constexpr std::size_t minimax_epochs = 20;
// Five times the DCGAN rate: at 0.0002 the generator had not begun to use the label after 20
// epochs of 10000 images, which is a fraction of the schedule the rate was chosen for.
constexpr double gan_learning_rate = 0.001;

// Evaluation draws this many samples per class from every generator, with the same noise.
constexpr std::size_t per_class = 100;

std::string data_directory(const char* variable, const std::string& fallback) {
    if (const char* override_path = std::getenv(variable)) {
        return override_path;
    }
    return fallback;
}

double euclidean_distance(const std::vector<double>& left, const std::vector<double>& right) {
    double total = 0.0;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        total += difference * difference;
    }
    return std::sqrt(total);
}

// Samples grouped by class: images[c] holds every image generated, or drawn, for class c.
using ClassGroups = std::vector<ml_scratch::FeatureMatrix>;

ClassGroups group_by_class(const ml_scratch::LabeledDataset& dataset, const std::size_t limit) {
    ClassGroups groups(class_count);
    for (const auto& sample : dataset) {
        if (groups[sample.label].size() < limit) {
            groups[sample.label].push_back(sample.features);
        }
    }
    return groups;
}

// Mean pairwise Euclidean distance between the images of one class. A generator that has
// collapsed onto one image per class scores zero here no matter how convincing that image is.
double within_class_spread(const ml_scratch::FeatureMatrix& images) {
    double total = 0.0;
    std::size_t pairs = 0;
    for (std::size_t first = 0; first < images.size(); ++first) {
        for (std::size_t second = first + 1; second < images.size(); ++second) {
            total += euclidean_distance(images[first], images[second]);
            ++pairs;
        }
    }
    return pairs == 0 ? 0.0 : total / static_cast<double>(pairs);
}

// Mean over `from` of the distance to the nearest image in `to`, class by class.
double mean_nearest_distance(const ClassGroups& from, const ClassGroups& to) {
    double total = 0.0;
    std::size_t count = 0;
    for (std::size_t label = 0; label < class_count; ++label) {
        for (const auto& image : from[label]) {
            double nearest = std::numeric_limits<double>::infinity();
            for (const auto& candidate : to[label]) {
                nearest = std::min(nearest, euclidean_distance(image, candidate));
            }
            total += nearest;
            ++count;
        }
    }
    return total / static_cast<double>(count);
}

struct Evaluation {
    // Fraction of samples the judge assigns to the class they were generated for.
    double class_consistency;
    // Within-class spread as a fraction of the real held-out images' spread, averaged over
    // classes. One means as varied as the data; zero means one image per class.
    double diversity_ratio;
    // Number of classes with at least half their samples judged correctly.
    std::size_t classes_covered;
    // Mean distance from each real held-out image to its nearest generated image of the same
    // class: how closely the generated set covers the data. Lower is better.
    double coverage_distance;
    // Mean distance from each generated image to its nearest real training image of the same
    // class. A generator that memorized the training set scores near zero; the real held-out
    // images' own figure is the yardstick for a sample that is new but on the data manifold.
    double novelty_distance;
};

Evaluation evaluate(const ClassGroups& generated, const ClassGroups& training,
                    const ClassGroups& held_out, const ml_scratch::FeedForwardNetwork& judge) {
    std::size_t consistent = 0;
    std::size_t total = 0;
    std::size_t covered = 0;
    double ratio_total = 0.0;
    for (std::size_t label = 0; label < class_count; ++label) {
        std::size_t class_consistent = 0;
        for (const auto& image : generated[label]) {
            class_consistent += judge.predict_class(image) == label ? 1 : 0;
        }
        consistent += class_consistent;
        total += generated[label].size();
        if (2 * class_consistent >= generated[label].size()) {
            ++covered;
        }
        ratio_total += within_class_spread(generated[label]) / within_class_spread(held_out[label]);
    }
    return {static_cast<double>(consistent) / static_cast<double>(total),
            ratio_total / static_cast<double>(class_count), covered,
            mean_nearest_distance(held_out, generated), mean_nearest_distance(generated, training)};
}

ClassGroups generate_groups(const ml_scratch::ConditionalGan& gan, const std::uint32_t noise_seed) {
    ClassGroups groups(class_count);
    for (std::size_t label = 0; label < class_count; ++label) {
        // The same noise for every class, so the label is the only thing that differs.
        std::mt19937 engine{noise_seed};
        for (std::size_t index = 0; index < per_class; ++index) {
            groups[label].push_back(gan.generate(gan.sample_noise(engine), label));
        }
    }
    return groups;
}

// Baseline one: the training mean image of each class, repeated. It is as recognizable as a
// class average can be and has no diversity at all.
ClassGroups class_mean_baseline(const ClassGroups& training) {
    ClassGroups groups(class_count);
    for (std::size_t label = 0; label < class_count; ++label) {
        const std::size_t pixels = training[label].front().size();
        std::vector<double> mean(pixels, 0.0);
        for (const auto& image : training[label]) {
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                mean[pixel] += image[pixel] / static_cast<double>(training[label].size());
            }
        }
        groups[label].assign(per_class, mean);
    }
    return groups;
}

// Baseline two: every pixel drawn independently from its class's per-pixel mean and standard
// deviation, clipped to [0, 1]. It has as much per-pixel variety as the data, but no model of
// how pixels go together.
ClassGroups independent_pixel_baseline(const ClassGroups& training, const std::uint32_t noise_seed) {
    ClassGroups groups(class_count);
    std::mt19937 engine{noise_seed};
    constexpr double range = 4294967296.0;
    const auto normal = [&engine] {
        const double first = (static_cast<double>(engine()) + 1.0) / range;
        const double second = static_cast<double>(engine()) / range;
        return std::sqrt(-2.0 * std::log(first)) * std::cos(6.283185307179586 * second);
    };
    for (std::size_t label = 0; label < class_count; ++label) {
        const std::size_t pixels = training[label].front().size();
        const auto count = static_cast<double>(training[label].size());
        std::vector<double> mean(pixels, 0.0);
        std::vector<double> deviation(pixels, 0.0);
        for (const auto& image : training[label]) {
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                mean[pixel] += image[pixel] / count;
            }
        }
        for (const auto& image : training[label]) {
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                const double difference = image[pixel] - mean[pixel];
                deviation[pixel] += difference * difference / count;
            }
        }
        for (std::size_t index = 0; index < per_class; ++index) {
            std::vector<double> image(pixels);
            for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
                image[pixel] = std::clamp(mean[pixel] + std::sqrt(deviation[pixel]) * normal(),
                                          0.0, 1.0);
            }
            groups[label].push_back(std::move(image));
        }
    }
    return groups;
}

// ---- Output: an ASCII strip and a PNG grid, written without any image library. ----

void print_ascii_digits(const ClassGroups& generated, const std::size_t rows,
                        const std::size_t columns) {
    // One generated sample per class, side by side, every other row and column so that ten
    // digits fit in a terminal.
    constexpr std::string_view ramp = " .:-=+*#%@";
    for (std::size_t row = 0; row < rows; row += 2) {
        std::cout << "  ";
        for (std::size_t label = 0; label < class_count; ++label) {
            const auto& image = generated[label].front();
            for (std::size_t column = 0; column < columns; column += 2) {
                const double value = std::clamp(image[row * columns + column], 0.0, 1.0);
                const auto shade = static_cast<std::size_t>(value * (ramp.size() - 1) + 0.5);
                std::cout << ramp[shade];
            }
            std::cout << ' ';
        }
        std::cout << '\n';
    }
}

std::uint32_t crc32(const std::vector<std::uint8_t>& bytes, const std::uint32_t initial = 0) {
    std::uint32_t crc = ~initial;
    for (const std::uint8_t byte : bytes) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

void append_big_endian(std::vector<std::uint8_t>& bytes, const std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value >> 24));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8));
    bytes.push_back(static_cast<std::uint8_t>(value));
}

void append_chunk(std::vector<std::uint8_t>& file, const std::string_view type,
                  const std::vector<std::uint8_t>& payload) {
    append_big_endian(file, static_cast<std::uint32_t>(payload.size()));
    std::vector<std::uint8_t> checked(type.begin(), type.end());
    checked.insert(checked.end(), payload.begin(), payload.end());
    file.insert(file.end(), checked.begin(), checked.end());
    append_big_endian(file, crc32(checked));
}

// An 8-bit grayscale PNG whose image data is a zlib stream of stored (uncompressed) deflate
// blocks, which is the one form of deflate that needs no compressor: each block is a header, a
// length, its complement, and the bytes.
bool write_png(const std::string& path, const std::vector<std::uint8_t>& pixels,
               const std::size_t width, const std::size_t height) {
    std::vector<std::uint8_t> raw;
    raw.reserve((width + 1) * height);
    for (std::size_t row = 0; row < height; ++row) {
        raw.push_back(0); // filter type: none
        raw.insert(raw.end(), pixels.begin() + static_cast<std::ptrdiff_t>(row * width),
                   pixels.begin() + static_cast<std::ptrdiff_t>((row + 1) * width));
    }

    std::vector<std::uint8_t> zlib{0x78, 0x01};
    constexpr std::size_t block_limit = 65535;
    for (std::size_t begin = 0; begin < raw.size(); begin += block_limit) {
        const std::size_t length = std::min(block_limit, raw.size() - begin);
        const bool last = begin + length == raw.size();
        zlib.push_back(last ? 1 : 0);
        zlib.push_back(static_cast<std::uint8_t>(length & 0xFF));
        zlib.push_back(static_cast<std::uint8_t>(length >> 8));
        zlib.push_back(static_cast<std::uint8_t>(~length & 0xFF));
        zlib.push_back(static_cast<std::uint8_t>((~length >> 8) & 0xFF));
        zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(begin),
                    raw.begin() + static_cast<std::ptrdiff_t>(begin + length));
    }
    std::uint32_t adler_a = 1;
    std::uint32_t adler_b = 0;
    for (const std::uint8_t byte : raw) {
        adler_a = (adler_a + byte) % 65521;
        adler_b = (adler_b + adler_a) % 65521;
    }
    append_big_endian(zlib, (adler_b << 16) | adler_a);

    std::vector<std::uint8_t> file{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    std::vector<std::uint8_t> header;
    append_big_endian(header, static_cast<std::uint32_t>(width));
    append_big_endian(header, static_cast<std::uint32_t>(height));
    header.insert(header.end(), {8, 0, 0, 0, 0}); // 8-bit grayscale, no interlace
    append_chunk(file, "IHDR", header);
    append_chunk(file, "IDAT", zlib);
    append_chunk(file, "IEND", {});

    std::ofstream stream{path, std::ios::binary};
    if (!stream) {
        return false;
    }
    stream.write(reinterpret_cast<const char*>(file.data()),
                 static_cast<std::streamsize>(file.size()));
    return static_cast<bool>(stream);
}

// A grid with one row per class and `columns` samples across, separated by a one-pixel gutter.
bool write_sample_grid(const std::string& path, const ClassGroups& generated,
                       const std::size_t rows, const std::size_t columns,
                       const std::size_t samples) {
    const std::size_t width = samples * (columns + 1) + 1;
    const std::size_t height = class_count * (rows + 1) + 1;
    std::vector<std::uint8_t> pixels(width * height, 64);
    for (std::size_t label = 0; label < class_count; ++label) {
        for (std::size_t sample = 0; sample < samples; ++sample) {
            const auto& image = generated[label][sample];
            for (std::size_t row = 0; row < rows; ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    const double value = std::clamp(image[row * columns + column], 0.0, 1.0);
                    const std::size_t x = 1 + sample * (columns + 1) + column;
                    const std::size_t y = 1 + label * (rows + 1) + row;
                    pixels[y * width + x] = static_cast<std::uint8_t>(value * 255.0 + 0.5);
                }
            }
        }
    }
    return write_png(path, pixels, width, height);
}

void print_evaluation_header() {
    std::cout << "  model                     consistency   diversity   covered   coverage   "
                 "novelty\n";
}

void print_evaluation(const std::string& label, const Evaluation& evaluation) {
    std::string padded = label;
    padded.resize(24, ' ');
    std::cout << "  " << padded << std::setw(14) << evaluation.class_consistency << std::setw(12)
              << evaluation.diversity_ratio << std::setw(10) << evaluation.classes_covered
              << std::setw(11) << evaluation.coverage_distance << std::setw(10)
              << evaluation.novelty_distance << '\n';
}

ml_scratch::ConditionalGan build_gan(const std::size_t pixels) {
    return ml_scratch::ConditionalGan{
        noise_size,
        class_count,
        {{noise_size + class_count, hidden, ml_scratch::Activation::leaky_rectified_linear},
         {hidden, pixels, ml_scratch::Activation::sigmoid}},
        {{pixels + class_count, hidden, ml_scratch::Activation::leaky_rectified_linear},
         {hidden, 1, ml_scratch::Activation::identity}},
        seed};
}

struct EpochRow {
    double discriminator_loss;
    double generator_loss;
    double real_score;
    double fake_score;
    double gradient_norm;
    double consistency;
    double diversity;
};

void print_epoch_header() {
    std::cout << "  epoch   D loss   G loss   D(real)   D(fake)   |grad G|   consistency   "
                 "diversity   seconds\n";
}

void print_epoch(const std::size_t epoch, const EpochRow& row, const double seconds) {
    std::cout << "  " << std::setw(5) << epoch << std::setw(9) << row.discriminator_loss
              << std::setw(9) << row.generator_loss << std::setw(10) << row.real_score
              << std::setw(10) << row.fake_score << std::setw(11) << std::scientific
              << std::setprecision(2) << row.gradient_norm << std::defaultfloat << std::fixed
              << std::setprecision(4) << std::setw(14) << row.consistency << std::setw(12)
              << row.diversity << std::setw(10) << std::setprecision(1) << seconds
              << std::setprecision(4) << '\n';
}

} // namespace

int main() {
    const std::string mnist = data_directory("ML_SCRATCH_MNIST_DIR", "data/mnist");
    ml_scratch::LabeledDataset training;
    ml_scratch::LabeledDataset held_out;
    std::size_t rows = 0;
    std::size_t columns = 0;
    try {
        const auto data = ml_scratch::load_mnist(mnist + "/train-images-idx3-ubyte",
                                                 mnist + "/train-labels-idx1-ubyte",
                                                 train_count + held_out_count);
        training.assign(data.samples.begin(),
                        data.samples.begin() + static_cast<std::ptrdiff_t>(train_count));
        held_out.assign(data.samples.begin() + static_cast<std::ptrdiff_t>(train_count),
                        data.samples.end());
        rows = data.rows;
        columns = data.columns;
    } catch (const std::exception& error) {
        std::cerr << "could not load MNIST from " << mnist << ": " << error.what() << "\n\n"
                  << "Download it first: ./scripts/download_mnist.sh\n";
        return 1;
    }
    const std::size_t pixels = rows * columns;
    const ClassGroups training_groups = group_by_class(training, train_count);
    const ClassGroups held_out_groups = group_by_class(held_out, per_class);

    std::cout << std::fixed << std::setprecision(4)
              << "C++ conditional GAN on MNIST (seed=" << seed << ")\n\n"
              << "  " << training.size() << " training images for both the GAN and the judge, "
              << held_out.size() << " held out\n"
              << "  generator " << noise_size << "+" << class_count << " -> " << hidden
              << " leaky ReLU -> " << pixels << " sigmoid; discriminator " << pixels << "+"
              << class_count << " -> " << hidden << " leaky ReLU -> 1 logit\n"
              << "  Adam at " << gan_learning_rate << " with beta1 = 0.5, batch 64, "
              << gan_epochs << " epochs, non-saturating generator loss\n\n";

    // ---------- The judge ----------
    // A classifier that never sees a generated image while training. Its held-out accuracy is
    // reported so the scores below can be read against how good a judge it is.
    std::cout << "judge: " << pixels << " -> " << judge_hidden << " ReLU -> " << class_count
              << ", Adam at 0.001, batch 32, " << judge_epochs << " epochs\n";
    ml_scratch::FeedForwardNetwork judge{
        {{pixels, judge_hidden, ml_scratch::Activation::rectified_linear},
         {judge_hidden, class_count, ml_scratch::Activation::identity}},
        ml_scratch::Loss::softmax_cross_entropy,
        seed};
    {
        ml_scratch::NetworkTrainingConfig config;
        config.optimizer.kind = ml_scratch::OptimizerKind::adam;
        config.learning_rate = 0.001;
        config.batch_size = 32;
        config.max_epochs = judge_epochs;
        config.seed = seed;
        const auto start = std::chrono::steady_clock::now();
        static_cast<void>(judge.fit(training, config));
        const auto finish = std::chrono::steady_clock::now();
        std::cout << "  held-out accuracy " << judge.accuracy(held_out) << " ("
                  << std::setprecision(1) << std::chrono::duration<double>(finish - start).count()
                  << " s)\n\n"
                  << std::setprecision(4);
    }
    const double judge_accuracy = judge.accuracy(held_out);

    // ---------- Baselines and the real data under the same metrics ----------
    const std::uint32_t evaluation_noise_seed = seed + 7;
    std::cout << "baselines under the same metrics\n"
              << "  consistency: fraction the judge assigns to the requested class; diversity: "
                 "within-class spread\n"
              << "  relative to real held-out images; covered: classes at least half consistent; "
                 "coverage:\n"
              << "  mean distance from a real held-out image to its nearest sample of that class; "
                 "novelty: mean\n"
              << "  distance from a sample to its nearest real training image of that class\n\n";
    print_evaluation_header();
    const Evaluation real_evaluation =
        evaluate(held_out_groups, training_groups, held_out_groups, judge);
    print_evaluation("real held-out images", real_evaluation);
    const Evaluation mean_evaluation =
        evaluate(class_mean_baseline(training_groups), training_groups, held_out_groups, judge);
    print_evaluation("class-mean baseline", mean_evaluation);
    const Evaluation pixel_evaluation =
        evaluate(independent_pixel_baseline(training_groups, evaluation_noise_seed),
                 training_groups, held_out_groups, judge);
    print_evaluation("independent-pixel baseline", pixel_evaluation);
    std::cout << '\n';

    // ---------- Part one: adversarial training with the non-saturating loss ----------
    const auto train_and_track = [&](ml_scratch::ConditionalGan& gan, const std::size_t epochs,
                                     const ml_scratch::GeneratorLoss loss) {
        ml_scratch::GanTrainingConfig config;
        config.learning_rate = gan_learning_rate;
        config.epochs = 1;
        config.batch_size = 64;
        config.generator_loss = loss;
        std::vector<EpochRow> history;
        print_epoch_header();
        for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
            // One epoch at a time, so the evaluation can be run between epochs; the seed advances
            // so each epoch shuffles and draws noise differently.
            config.seed = seed + static_cast<std::uint32_t>(epoch);
            const auto start = std::chrono::steady_clock::now();
            const auto result = gan.fit(training, config);
            const auto finish = std::chrono::steady_clock::now();
            const Evaluation evaluation = evaluate(generate_groups(gan, evaluation_noise_seed),
                                                   training_groups, held_out_groups, judge);
            history.push_back({result.discriminator_loss_per_epoch.front(),
                               result.generator_loss_per_epoch.front(),
                               result.real_score_per_epoch.front(),
                               result.fake_score_per_epoch.front(),
                               result.generator_gradient_norm_per_epoch.front(),
                               evaluation.class_consistency, evaluation.diversity_ratio});
            print_epoch(epoch, history.back(),
                        std::chrono::duration<double>(finish - start).count());
        }
        return history;
    };

    std::cout << "part one: the non-saturating generator loss\n";
    ml_scratch::ConditionalGan gan = build_gan(pixels);
    const Evaluation untrained =
        evaluate(generate_groups(gan, evaluation_noise_seed), training_groups, held_out_groups,
                 judge);
    const auto history = train_and_track(gan, gan_epochs, ml_scratch::GeneratorLoss::non_saturating);
    std::cout << '\n';

    const ClassGroups generated = generate_groups(gan, evaluation_noise_seed);
    const Evaluation trained = evaluate(generated, training_groups, held_out_groups, judge);
    print_evaluation_header();
    print_evaluation("untrained generator", untrained);
    print_evaluation("trained generator", trained);
    std::cout << "\n  one generated sample per class, 0 to 9, from the trained generator\n\n";
    print_ascii_digits(generated, rows, columns);

    const std::string figure =
        data_directory("ML_SCRATCH_FIGURE_PATH", "results/figures/14_cgan_mnist_samples.png");
    if (write_sample_grid(figure, generated, rows, columns, 8)) {
        std::cout << "\n  wrote " << figure << ": one row per class, eight samples across\n";
    } else {
        std::cout << "\n  could not write " << figure << " (is the directory present?)\n";
    }
    const std::string checkpoint =
        data_directory("ML_SCRATCH_CGAN_CHECKPOINT", "data/cgan_mnist.checkpoint");
    try {
        gan.save(checkpoint);
        const auto restored = ml_scratch::ConditionalGan::load(checkpoint);
        std::mt19937 engine{1};
        const auto noise = gan.sample_noise(engine);
        std::cout << "  checkpoint " << checkpoint << " round-trips: "
                  << (restored.generate(noise, 3) == gan.generate(noise, 3) ? "yes" : "NO") << "\n";
    } catch (const std::exception& error) {
        std::cout << "  checkpoint not written: " << error.what() << '\n';
    }

    // ---------- Part two: the minimax loss, from the same initialization ----------
    std::cout << "\npart two: the minimax generator loss, same networks and schedule, "
              << minimax_epochs << " epochs\n"
              << "  the generator minimizes log(1 - D(G(z))) instead of -log D(G(z)); its gradient "
                 "is proportional\n"
              << "  to D(G(z)), so it should fade exactly when the discriminator is winning\n";
    ml_scratch::ConditionalGan minimax_gan = build_gan(pixels);
    const auto minimax_history =
        train_and_track(minimax_gan, minimax_epochs, ml_scratch::GeneratorLoss::minimax);
    const Evaluation minimax_evaluation = evaluate(
        generate_groups(minimax_gan, evaluation_noise_seed), training_groups, held_out_groups, judge);

    // ---------- Summary and criteria ----------
    double best_diversity = 0.0;
    double worst_diversity = std::numeric_limits<double>::infinity();
    for (const EpochRow& row : history) {
        best_diversity = std::max(best_diversity, row.diversity);
        worst_diversity = std::min(worst_diversity, row.diversity);
    }
    const EpochRow& first = history.front();
    const EpochRow& last = history.back();
    const EpochRow& minimax_last = minimax_history.back();
    const EpochRow& same_epoch = history[minimax_epochs - 1];
    // The generator's gradient norm averaged over the epochs both runs share.
    double mean_gradient = 0.0;
    double minimax_mean_gradient = 0.0;
    for (std::size_t epoch = 0; epoch < minimax_epochs; ++epoch) {
        mean_gradient += history[epoch].gradient_norm / static_cast<double>(minimax_epochs);
        minimax_mean_gradient +=
            minimax_history[epoch].gradient_norm / static_cast<double>(minimax_epochs);
    }

    std::cout << "\nsummary\n"
              << "  judge held-out accuracy " << judge_accuracy << "; chance consistency "
              << 1.0 / static_cast<double>(class_count) << '\n'
              << "  trained generator: consistency " << trained.class_consistency
              << " (untrained " << untrained.class_consistency << "), diversity "
              << trained.diversity_ratio << ", " << trained.classes_covered << "/" << class_count
              << " classes covered\n"
              << "  coverage distance " << trained.coverage_distance << " against "
              << mean_evaluation.coverage_distance << " for the class mean and "
              << pixel_evaluation.coverage_distance << " for independent pixels\n"
              << "  novelty distance " << trained.novelty_distance << " against "
              << real_evaluation.novelty_distance << " for real held-out images\n"
              << "  diversity ranged from " << worst_diversity << " to " << best_diversity
              << " across epochs\n"
              << "  minimax over " << minimax_epochs << " epochs: mean |grad G| "
              << std::scientific << std::setprecision(2) << minimax_mean_gradient << " against "
              << mean_gradient << " non-saturating" << std::defaultfloat << std::fixed
              << std::setprecision(4) << "; at epoch " << minimax_epochs << " D(fake) "
              << minimax_last.fake_score << " against " << same_epoch.fake_score
              << ", consistency " << minimax_evaluation.class_consistency << " against "
              << same_epoch.consistency << '\n';

    // The judge has to be competent for its verdicts to mean anything.
    const bool judge_is_competent = judge_accuracy > 0.9;
    // The roadmap's requirement: class consistency, diversity, and coverage measured by a
    // separately trained classifier, against a baseline.
    const bool learned_the_classes =
        trained.class_consistency > 0.8 && trained.class_consistency > untrained.class_consistency;
    const bool every_class_covered = trained.classes_covered == class_count;
    const bool did_not_collapse = trained.diversity_ratio > 0.5;
    // The independent-pixel sampler has the data's per-pixel variety; the generator has to reach
    // the real images more closely than it does. The class mean is reported but not required: it
    // is the L2-optimal single image, and nearest-neighbour distance rewards exactly that.
    const bool beat_the_baseline = trained.coverage_distance < pixel_evaluation.coverage_distance;
    // Samples are not copies of training images: as far from their nearest training image as a
    // genuinely new image is, to within a margin.
    const bool did_not_memorize = trained.novelty_distance > 0.5 * real_evaluation.novelty_distance;
    // The instability the minimax loss is known for: a gradient that fades while the
    // discriminator is winning, and a generator that has learned less by the same epoch.
    const bool minimax_saturated = minimax_mean_gradient < 0.5 * mean_gradient &&
                                   minimax_evaluation.class_consistency < same_epoch.consistency;
    const bool discriminator_never_won_outright = last.fake_score > 0.05 && first.fake_score > 0.05;

    if (!judge_is_competent || !learned_the_classes || !every_class_covered || !did_not_collapse ||
        !beat_the_baseline || !did_not_memorize || !minimax_saturated ||
        !discriminator_never_won_outright) {
        std::cerr << "experiment failed its criteria: judge=" << judge_accuracy
                  << ", consistency=" << trained.class_consistency
                  << ", diversity=" << trained.diversity_ratio
                  << ", covered=" << trained.classes_covered
                  << ", coverage=" << trained.coverage_distance
                  << ", novelty=" << trained.novelty_distance
                  << ", minimax mean |grad G|=" << minimax_mean_gradient
                  << ", D(fake)=" << last.fake_score << '\n';
        return 1;
    }
    return 0;
}
