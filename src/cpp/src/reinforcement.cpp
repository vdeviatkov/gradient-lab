#include "ml_scratch/reinforcement.hpp"

#include "ml_scratch/rnn.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

namespace ml_scratch {
namespace {

std::uint64_t next_random(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

double uniform(std::uint64_t& state) noexcept {
    // 53 bits is a double's mantissa, so this is uniform over the representable values in [0, 1).
    return static_cast<double>(next_random(state) >> 11) / 9007199254740992.0;
}

// The greedy action over a span of action values, with ties going to the lowest index.
std::size_t argmax(const double* values, const std::size_t count) noexcept {
    std::size_t best = 0;
    for (std::size_t index = 1; index < count; ++index) {
        if (values[index] > values[best]) {
            best = index;
        }
    }
    return best;
}

} // namespace

double ExplorationSchedule::epsilon(const std::size_t taken) const noexcept {
    if (steps == 0 || taken >= steps) {
        return end;
    }
    const double progress = static_cast<double>(taken) / static_cast<double>(steps);
    return start + (end - start) * progress;
}

// ---------------------------------------------------------------------------------------------
// Tabular Q-learning
// ---------------------------------------------------------------------------------------------

TabularQLearning::TabularQLearning() : TabularQLearning(Config{}) {}

TabularQLearning::TabularQLearning(Config config)
    : config_(config),
      table_(SnakeGame::state_count * snake_action_count, 0.0),
      visited_(SnakeGame::state_count, false),
      random_state_(config.seed) {
    if (!std::isfinite(config_.learning_rate) || config_.learning_rate <= 0.0 ||
        config_.learning_rate > 1.0) {
        throw std::invalid_argument("the learning rate must be in (0, 1]");
    }
    if (!std::isfinite(config_.discount) || config_.discount < 0.0 || config_.discount > 1.0) {
        throw std::invalid_argument("the discount must be in [0, 1]");
    }
}

double TabularQLearning::value(const std::size_t state, const std::size_t action) const {
    if (state >= SnakeGame::state_count || action >= snake_action_count) {
        throw std::invalid_argument("state or action is out of range");
    }
    return table_[state * snake_action_count + action];
}

std::size_t TabularQLearning::act(const std::size_t state) const {
    if (state >= SnakeGame::state_count) {
        throw std::invalid_argument("state is out of range");
    }
    return argmax(table_.data() + state * snake_action_count, snake_action_count);
}

double TabularQLearning::epsilon() const noexcept {
    return config_.exploration.epsilon(selections_);
}

std::size_t TabularQLearning::explore(const std::size_t state) {
    const double threshold = epsilon();
    ++selections_;
    if (uniform(random_state_) < threshold) {
        return static_cast<std::size_t>(next_random(random_state_) % snake_action_count);
    }
    return act(state);
}

double TabularQLearning::learn(const Transition& transition) {
    if (transition.state >= SnakeGame::state_count ||
        transition.next_state >= SnakeGame::state_count ||
        transition.action >= snake_action_count) {
        throw std::invalid_argument("state or action is out of range");
    }
    // A terminal state has no future, so its value is zero rather than whatever the table
    // happens to hold for the observation the dead snake would have seen.
    double target = transition.reward;
    if (!transition.died) {
        const double* next = table_.data() + transition.next_state * snake_action_count;
        target += config_.discount * next[argmax(next, snake_action_count)];
    }
    double& entry = table_[transition.state * snake_action_count + transition.action];
    const double error = target - entry;
    entry += config_.learning_rate * error;
    visited_[transition.state] = true;
    return error;
}

std::size_t TabularQLearning::visited_states() const {
    return static_cast<std::size_t>(std::count(visited_.begin(), visited_.end(), true));
}

void TabularQLearning::save(const std::string& path) const {
    std::ofstream stream{path};
    if (!stream) {
        throw std::runtime_error("cannot open checkpoint for writing: " + path);
    }
    stream << "ml_scratch_tabular_q 1\n"
           << "states " << SnakeGame::state_count << '\n'
           << "actions " << snake_action_count << '\n'
           << "selections " << selections_ << '\n'
           << std::setprecision(17);
    for (std::size_t state = 0; state < SnakeGame::state_count; ++state) {
        for (std::size_t action = 0; action < snake_action_count; ++action) {
            stream << table_[state * snake_action_count + action] << ' ';
        }
        stream << (visited_[state] ? 1 : 0) << '\n';
    }
    if (!stream) {
        throw std::runtime_error("failed while writing checkpoint: " + path);
    }
}

TabularQLearning TabularQLearning::load(const std::string& path) {
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
    expect("ml_scratch_tabular_q");
    if (read_size() != 1) {
        throw std::runtime_error("unsupported checkpoint version in " + path);
    }
    expect("states");
    if (read_size() != SnakeGame::state_count) {
        throw std::runtime_error("checkpoint state count does not match: " + path);
    }
    expect("actions");
    if (read_size() != snake_action_count) {
        throw std::runtime_error("checkpoint action count does not match: " + path);
    }
    expect("selections");
    const std::size_t selections = read_size();

    TabularQLearning agent{};
    for (std::size_t state = 0; state < SnakeGame::state_count; ++state) {
        for (std::size_t action = 0; action < snake_action_count; ++action) {
            if (!(stream >> agent.table_[state * snake_action_count + action])) {
                throw std::runtime_error("truncated checkpoint: " + path);
            }
        }
        int visited = 0;
        if (!(stream >> visited)) {
            throw std::runtime_error("truncated checkpoint: " + path);
        }
        agent.visited_[state] = visited != 0;
    }
    agent.selections_ = selections;
    return agent;
}

// ---------------------------------------------------------------------------------------------
// Replay
// ---------------------------------------------------------------------------------------------

ReplayBuffer::ReplayBuffer(const std::size_t capacity, const std::uint64_t seed)
    : capacity_(capacity), random_state_(seed) {
    if (capacity == 0) {
        throw std::invalid_argument("the replay capacity must be positive");
    }
    buffer_.reserve(capacity);
}

void ReplayBuffer::push(Transition transition) {
    if (buffer_.size() < capacity_) {
        buffer_.push_back(std::move(transition));
        return;
    }
    // Once full the oldest transition is overwritten, so the buffer holds a moving window of the
    // agent's recent experience rather than its whole history.
    buffer_[cursor_] = std::move(transition);
    cursor_ = (cursor_ + 1) % capacity_;
}

std::vector<const Transition*> ReplayBuffer::sample(const std::size_t count) {
    if (buffer_.empty()) {
        throw std::logic_error("the replay buffer is empty");
    }
    std::vector<const Transition*> batch;
    batch.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        batch.push_back(&buffer_[static_cast<std::size_t>(next_random(random_state_) %
                                                          buffer_.size())]);
    }
    return batch;
}

// ---------------------------------------------------------------------------------------------
// Deep Q-learning
// ---------------------------------------------------------------------------------------------

std::vector<DenseLayer> DeepQLearning::layers(const Config& config) {
    std::vector<DenseLayer> layers;
    std::size_t input = SnakeGame::observation_size;
    for (const std::size_t width : config.hidden) {
        layers.push_back({input, width, Activation::rectified_linear});
        input = width;
    }
    // The output is one value per action, and a value is any real number, so the final layer is
    // linear and the loss is squared error rather than a cross-entropy.
    layers.push_back({input, snake_action_count, Activation::identity});
    return layers;
}

DeepQLearning::DeepQLearning() : DeepQLearning(Config{}) {}

DeepQLearning::DeepQLearning(Config config)
    : config_(std::move(config)),
      network_(layers(config_), Loss::mean_squared_error, config_.seed),
      target_(network_),
      optimizer_(OptimizerConfig{.kind = OptimizerKind::adam}, network_.parameter_count()),
      replay_(config_.replay_capacity, config_.seed + 1u),
      random_state_(config_.seed + 2u) {
    if (config_.hidden.empty()) {
        throw std::invalid_argument("the network needs at least one hidden layer");
    }
    if (!std::isfinite(config_.learning_rate) || config_.learning_rate <= 0.0) {
        throw std::invalid_argument("the learning rate must be positive");
    }
    if (!std::isfinite(config_.discount) || config_.discount < 0.0 || config_.discount > 1.0) {
        throw std::invalid_argument("the discount must be in [0, 1]");
    }
    if (config_.batch_size == 0) {
        throw std::invalid_argument("the batch size must be positive");
    }
    if (!std::isfinite(config_.clip_norm) || config_.clip_norm < 0.0) {
        throw std::invalid_argument("clip_norm must be finite and non-negative");
    }
    // A warmup longer than the buffer can hold would never be reached, and the agent would
    // collect experience forever without learning from any of it.
    if (config_.warmup > config_.replay_capacity) {
        throw std::invalid_argument("the warmup cannot exceed the replay capacity");
    }
}

std::vector<double> DeepQLearning::values(const std::vector<double>& features) const {
    return network_.forward(features);
}

std::size_t DeepQLearning::act(const std::vector<double>& features) const {
    const std::vector<double> q = values(features);
    return argmax(q.data(), q.size());
}

double DeepQLearning::epsilon() const noexcept {
    return config_.exploration.epsilon(selections_);
}

std::size_t DeepQLearning::explore(const std::vector<double>& features) {
    const double threshold = epsilon();
    ++selections_;
    if (uniform(random_state_) < threshold) {
        return static_cast<std::size_t>(next_random(random_state_) % snake_action_count);
    }
    return act(features);
}

double DeepQLearning::observe(Transition transition) {
    replay_.push(std::move(transition));
    if (replay_.size() < config_.warmup) {
        return 0.0;
    }

    const std::vector<const Transition*> batch = replay_.sample(config_.batch_size);
    std::vector<double> gradient(network_.parameter_count(), 0.0);
    std::vector<double> output_gradient(snake_action_count, 0.0);
    FeedForwardNetwork::Workspace workspace;
    const double scale = 1.0 / static_cast<double>(batch.size());
    double squared_error = 0.0;

    for (const Transition* sample : batch) {
        // The target is computed from the target network, which lags the learner; with
        // target_update zero the two are the same object and the target moves with every step,
        // which is the instability the ablation measures.
        double target = sample->reward;
        if (!sample->died) {
            const std::vector<double> next = target_.forward(sample->next_features);
            target += config_.discount * next[argmax(next.data(), next.size())];
        }
        const std::vector<double>& predicted = network_.forward(sample->features, workspace);
        const double error = predicted[sample->action] - target;
        squared_error += scale * error * error;
        // Only the taken action has a target, so every other output's gradient is zero: the
        // update says nothing about actions this transition did not try.
        std::fill(output_gradient.begin(), output_gradient.end(), 0.0);
        output_gradient[sample->action] = 2.0 * error;
        network_.backpropagate(output_gradient, scale, &gradient, workspace);
    }

    if (config_.clip_norm > 0.0) {
        static_cast<void>(clip_gradient(gradient, config_.clip_norm));
    }
    std::vector<double> update;
    optimizer_.compute_update(gradient, config_.learning_rate, update);
    network_.subtract_update(update);
    ++updates_;
    if (config_.target_update > 0 && updates_ % config_.target_update == 0) {
        target_.set_parameters(network_.parameters());
    } else if (config_.target_update == 0) {
        target_.set_parameters(network_.parameters());
    }
    return squared_error;
}

void DeepQLearning::save(const std::string& path) const { network_.save(path); }

DeepQLearning DeepQLearning::load(const std::string& path, Config config) {
    FeedForwardNetwork restored = FeedForwardNetwork::load(path);
    DeepQLearning agent{std::move(config)};
    if (restored.parameter_count() != agent.network_.parameter_count()) {
        throw std::runtime_error("checkpoint does not match the configured architecture: " + path);
    }
    agent.network_.set_parameters(restored.parameters());
    agent.target_.set_parameters(restored.parameters());
    return agent;
}

} // namespace ml_scratch
