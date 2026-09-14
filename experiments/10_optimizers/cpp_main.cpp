#include "ml_scratch/mnist.hpp"
#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/optimizer.hpp"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260912;
constexpr std::size_t class_count = 10;
constexpr std::size_t hidden_units = 64;
constexpr std::size_t train_size = 10'000;
constexpr std::size_t validation_size = 2'000;
constexpr std::size_t epochs = 20;

std::string mnist_directory() {
    if (const char* override_path = std::getenv("ML_SCRATCH_MNIST_DIR")) {
        return override_path;
    }
    return "data/mnist";
}

struct Run {
    std::string label;
    std::size_t updates;
    double final_loss;
    double best_loss;
    double validation_accuracy;
    double seconds;
};

// Every run starts from the same parameters, so differences come from the update rule or the batch
// size alone and never from initialization.
Run train(const std::string& label, const ml_scratch::LabeledDataset& train_data,
          const ml_scratch::LabeledDataset& validation_data, const std::size_t feature_count,
          const ml_scratch::OptimizerConfig& optimizer, const double learning_rate,
          const std::size_t batch_size) {
    ml_scratch::NetworkTrainingConfig config;
    config.optimizer = optimizer;
    config.learning_rate = learning_rate;
    config.batch_size = batch_size;
    config.max_epochs = epochs;
    config.seed = seed;

    ml_scratch::FeedForwardNetwork network{
        {{feature_count, hidden_units, ml_scratch::Activation::rectified_linear},
         {hidden_units, class_count, ml_scratch::Activation::identity}},
        ml_scratch::Loss::softmax_cross_entropy,
        seed};

    const auto start = std::chrono::steady_clock::now();
    const auto result = network.fit(train_data, config);
    const auto finish = std::chrono::steady_clock::now();

    const std::size_t per_epoch =
        batch_size == 0 ? 1 : (train_data.size() + batch_size - 1) / batch_size;
    double best = result.loss_per_epoch.front();
    for (const double value : result.loss_per_epoch) {
        best = std::min(best, value);
    }
    return {label,
            per_epoch * epochs,
            result.loss_per_epoch.back(),
            best,
            network.accuracy(validation_data),
            std::chrono::duration<double>(finish - start).count()};
}

void print_header() {
    std::cout << "  configuration              updates   final loss    best loss   validation   "
                 "seconds\n";
}

void print_run(const Run& run) {
    std::string label = run.label;
    label.resize(26, ' ');
    std::cout << "  " << label << std::setw(9) << run.updates << std::setw(13) << std::fixed
              << std::setprecision(4) << run.final_loss << std::setw(13) << run.best_loss
              << std::setw(13) << run.validation_accuracy << std::setw(10) << std::setprecision(1)
              << run.seconds << std::setprecision(4) << '\n';
}

} // namespace

int main() {
    const std::string directory = mnist_directory();
    ml_scratch::MnistSplit data;
    try {
        data = ml_scratch::load_mnist(directory + "/train-images-idx3-ubyte",
                                      directory + "/train-labels-idx1-ubyte",
                                      train_size + validation_size);
    } catch (const std::exception& error) {
        std::cerr << "could not load MNIST from " << directory << ": " << error.what() << "\n\n"
                  << "Download it first (the data is not committed):\n"
                  << "    ./scripts/download_mnist.sh\n"
                  << "or point ML_SCRATCH_MNIST_DIR at a directory holding the uncompressed IDX "
                     "files.\n";
        return 1;
    }

    const ml_scratch::LabeledDataset train_data{
        data.samples.begin(), data.samples.begin() + static_cast<std::ptrdiff_t>(train_size)};
    const ml_scratch::LabeledDataset validation_data{
        data.samples.begin() + static_cast<std::ptrdiff_t>(train_size), data.samples.end()};
    const std::size_t feature_count = data.rows * data.columns;

    std::cout << std::fixed << std::setprecision(4)
              << "C++ optimizer comparison on MNIST (seed=" << seed << ")\n"
              << "network " << feature_count << " -> " << hidden_units << " ReLU -> " << class_count
              << ", softmax cross-entropy; every run starts from the same initial parameters\n"
              << "subset: train=" << train_data.size() << ", validation=" << validation_data.size()
              << "; budget=" << epochs << " epochs for every run\n"
              << "a fixed epoch budget means equal passes over the data, not equal numbers of "
                 "parameter updates\n\n";

    // Axis one: how much data goes into each update, with the update rule held at plain descent.
    // Each batch size is also given a rate that suits it, because the two are coupled: a batch of
    // one sees a gradient with far more noise than a batch of ten thousand, so the same rate does
    // not mean the same thing.
    std::cout << "batch size, plain gradient descent\n";
    print_header();
    const ml_scratch::OptimizerConfig plain{};
    const Run full_batch_slow =
        train("full batch, lr 0.10", train_data, validation_data, feature_count, plain, 0.10, 0);
    print_run(full_batch_slow);
    const Run full_batch =
        train("full batch, lr 1.00", train_data, validation_data, feature_count, plain, 1.00, 0);
    print_run(full_batch);
    const Run mini_batch = train("mini-batch 64, lr 0.10", train_data, validation_data,
                                 feature_count, plain, 0.10, 64);
    print_run(mini_batch);
    const Run stochastic =
        train("stochastic 1, lr 0.10", train_data, validation_data, feature_count, plain, 0.10, 1);
    print_run(stochastic);
    const Run stochastic_slow =
        train("stochastic 1, lr 0.01", train_data, validation_data, feature_count, plain, 0.01, 1);
    print_run(stochastic_slow);

    // Axis two: the update rule, with the batch size held at 64.
    std::cout << "\nupdate rule at batch size 64\n"
              << "  adaptive rules divide out the gradient scale, so they are run at the rate that "
                 "suits them\n";
    print_header();
    std::vector<Run> rule_runs;
    struct RuleCandidate {
        ml_scratch::OptimizerKind kind;
        double learning_rate;
    };
    for (const RuleCandidate& candidate :
         {RuleCandidate{ml_scratch::OptimizerKind::gradient_descent, 0.10},
          RuleCandidate{ml_scratch::OptimizerKind::momentum, 0.10},
          RuleCandidate{ml_scratch::OptimizerKind::nesterov_momentum, 0.10},
          RuleCandidate{ml_scratch::OptimizerKind::rmsprop, 0.001},
          RuleCandidate{ml_scratch::OptimizerKind::adam, 0.001}}) {
        ml_scratch::OptimizerConfig optimizer;
        optimizer.kind = candidate.kind;
        std::string label{ml_scratch::optimizer_name(candidate.kind)};
        label += ", lr " + std::to_string(candidate.learning_rate).substr(0, 5);
        rule_runs.push_back(train(label, train_data, validation_data, feature_count, optimizer,
                                  candidate.learning_rate, 64));
        print_run(rule_runs.back());
    }

    // Sensitivity is the practical reason adaptive rules are popular: they tolerate a learning rate
    // chosen without a search.
    std::cout << "\nsensitivity to the learning rate (batch 64, validation accuracy after "
              << epochs << " epochs)\n"
              << "  rule                   lr 0.001     lr 0.010     lr 0.100     lr 1.000   "
                 "spread\n";
    // Recorded for the two rates that Adam is usually run at, which is where the claim that
    // adaptive rules need less tuning can actually be checked.
    double plain_low_window = 0.0;
    double adam_low_window = 0.0;
    for (const ml_scratch::OptimizerKind kind :
         {ml_scratch::OptimizerKind::gradient_descent, ml_scratch::OptimizerKind::momentum,
          ml_scratch::OptimizerKind::adam}) {
        ml_scratch::OptimizerConfig optimizer;
        optimizer.kind = kind;
        std::string label{ml_scratch::optimizer_name(kind)};
        label.resize(22, ' ');
        std::cout << "  " << label;

        double lowest = 1.0;
        double highest = 0.0;
        std::vector<double> accuracies;
        for (const double rate : {0.001, 0.010, 0.100, 1.000}) {
            const Run run =
                train("sweep", train_data, validation_data, feature_count, optimizer, rate, 64);
            std::cout << std::setw(13) << run.validation_accuracy;
            accuracies.push_back(run.validation_accuracy);
            lowest = std::min(lowest, run.validation_accuracy);
            highest = std::max(highest, run.validation_accuracy);
        }
        std::cout << std::setw(9) << highest - lowest << '\n';
        const double low_window = std::abs(accuracies[0] - accuracies[1]);
        if (kind == ml_scratch::OptimizerKind::gradient_descent) {
            plain_low_window = low_window;
        } else if (kind == ml_scratch::OptimizerKind::adam) {
            adam_low_window = low_window;
        }
    }

    const Run& plain_rule = rule_runs.front();
    const Run& adam_rule = rule_runs.back();
    const Run* best_rule = &rule_runs.front();
    for (const Run& run : rule_runs) {
        if (run.validation_accuracy > best_rule->validation_accuracy) {
            best_rule = &run;
        }
    }

    // The better of the two full-batch rates, so the comparison against mini-batches is not an
    // artifact of a badly chosen rate.
    const Run& best_full_batch =
        full_batch.final_loss < full_batch_slow.final_loss ? full_batch : full_batch_slow;

    std::cout << "\nsummary\n"
              << "  full batch made only " << best_full_batch.updates << " updates in " << epochs
              << " epochs and reached loss " << best_full_batch.final_loss
              << " at its better rate; mini-batches made " << mini_batch.updates << " and reached "
              << mini_batch.final_loss << ", " << best_full_batch.final_loss / mini_batch.final_loss
              << "x lower\n"
              << "  raising the full-batch rate from 0.10 to 1.00 did not recover the missing "
                 "updates: the loss went from "
              << full_batch_slow.final_loss << " to " << full_batch.final_loss
              << ", because the stable step size is bounded by curvature rather than by patience\n"
              << "  batch size and learning rate are coupled: at batch 1 a rate of 0.10 reached "
              << stochastic.final_loss << " while 0.01 reached " << stochastic_slow.final_loss
              << '\n'
              << "  the best update rule after " << epochs << " epochs was " << best_rule->label
              << " at validation accuracy " << best_rule->validation_accuracy << '\n'
              << "  over the 0.001 to 0.010 window Adam's validation accuracy moved by "
              << adam_low_window << " and plain descent's by " << plain_low_window << '\n';

    // Mini-batches must win by a wide margin at the BETTER of the two full-batch rates, so the
    // gap cannot be dismissed as a badly tuned baseline.
    const bool mini_beat_full = best_full_batch.final_loss > 5.0 * mini_batch.final_loss;
    const bool stochastic_rate_coupling = stochastic_slow.final_loss < stochastic.final_loss;
    const bool momentum_beat_plain = rule_runs[1].best_loss < plain_rule.best_loss;
    const bool adaptive_rules_ran = adam_rule.validation_accuracy > 0.90;
    // The measured claim is narrow on purpose: within the decade Adam is normally run in, its
    // result barely moves, while plain descent's does. It is not that Adam tolerates any rate.
    const bool adam_flat_in_its_window = adam_low_window < plain_low_window;

    if (!mini_beat_full || !stochastic_rate_coupling || !momentum_beat_plain ||
        !adaptive_rules_ran || !adam_flat_in_its_window) {
        std::cerr << "experiment failed its criteria: mini-batch loss=" << mini_batch.final_loss
                  << " vs full batch=" << best_full_batch.final_loss
                  << ", momentum best loss=" << rule_runs[1].best_loss
                  << " vs plain=" << plain_rule.best_loss << ", Adam window=" << adam_low_window
                  << " vs plain=" << plain_low_window << '\n';
        return 1;
    }
    return 0;
}
