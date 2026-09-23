#pragma once

#include "ml_scratch/neural_network.hpp"
#include "ml_scratch/optimizer.hpp"
#include "ml_scratch/snake.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ml_scratch {

// One transition, the unit both learners consume. `next_features` and `next_state` describe the
// state the action led to; `died` says whether that state is terminal, which is the flag that
// decides whether its value is bootstrapped or taken to be zero. A truncated episode is NOT
// terminal: the agent was cut off, not killed, so its future value still counts.
struct Transition {
    std::vector<double> features;
    std::size_t state;
    std::size_t action;
    double reward;
    std::vector<double> next_features;
    std::size_t next_state;
    bool died;
};

// Linear decay from `start` to `end` over `steps` selections, then constant. The schedule is
// counted in action selections rather than episodes, so an agent that dies early does not race
// through its exploration.
struct ExplorationSchedule {
    double start{1.0};
    double end{0.05};
    std::size_t steps{50'000};

    [[nodiscard]] double epsilon(std::size_t taken) const noexcept;

    bool operator==(const ExplorationSchedule&) const = default;
};

struct EpisodeStatistics {
    double mean_return;
    double mean_score;
    double best_score;
    double mean_steps;
    // Fraction of episodes that ended in a wall or the snake's own body rather than the step
    // limit. An agent that learns to circle forever has a low death rate and a low score.
    double death_rate;
};

// Plays `episodes` greedy episodes — no exploration at all — from consecutive seeds starting at
// `seed`, so every agent is evaluated on exactly the same food sequences. This is the only
// function the experiment uses to compare agents: an epsilon-greedy score mixes the policy with
// its exploration noise and is not what the learned policy does.
template <typename Policy>
[[nodiscard]] EpisodeStatistics evaluate(Policy&& policy, const SnakeConfig& config,
                                         std::size_t episodes, std::uint64_t seed);

// Q(s, a) for the 2048 observation bit patterns, learned by the one-step update
//   Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a)).
class TabularQLearning {
  public:
    struct Config {
        double learning_rate{0.1};
        double discount{0.95};
        ExplorationSchedule exploration{};
        std::uint32_t seed{0};

        bool operator==(const Config&) const = default;
    };

    TabularQLearning();
    explicit TabularQLearning(Config config);

    // The greedy action, with ties broken by the lowest index so a fresh table is not silently
    // biased by whatever order the actions happen to be tried in.
    [[nodiscard]] std::size_t act(std::size_t state) const;
    // The same, exploring with the current epsilon and counting the selection.
    [[nodiscard]] std::size_t explore(std::size_t state);
    // Applies one update and returns the temporal-difference error it corrected.
    double learn(const Transition& transition);

    [[nodiscard]] double value(std::size_t state, std::size_t action) const;
    [[nodiscard]] double epsilon() const noexcept;
    [[nodiscard]] std::size_t selections() const noexcept { return selections_; }
    // How many of the 2048 states have ever been updated: how much of the table the agent's own
    // behaviour actually reached.
    [[nodiscard]] std::size_t visited_states() const;
    [[nodiscard]] const std::vector<double>& table() const noexcept { return table_; }

    void save(const std::string& path) const;
    [[nodiscard]] static TabularQLearning load(const std::string& path);

  private:
    Config config_;
    std::vector<double> table_;
    std::vector<bool> visited_;
    std::uint64_t random_state_;
    std::size_t selections_{0};
};

// A fixed-capacity circular buffer of transitions, sampled uniformly with replacement. Replay
// is what lets a network be trained on this problem at all: consecutive steps of one episode are
// almost the same state, and a network fitted on them in order forgets everything else.
class ReplayBuffer {
  public:
    ReplayBuffer(std::size_t capacity, std::uint64_t seed);

    void push(Transition transition);
    [[nodiscard]] std::vector<const Transition*> sample(std::size_t count);
    [[nodiscard]] std::size_t size() const noexcept { return buffer_.size(); }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

  private:
    std::size_t capacity_;
    std::vector<Transition> buffer_;
    std::size_t cursor_{0};
    std::uint64_t random_state_;
};

// Q-learning with a network in place of the table: the same update, with
//   target = r + gamma * max_a' Q_target(s', a')
// regressed onto Q(s, a) by squared error. Two things make that stable enough to train, and the
// experiment measures both: a replay buffer, and a target network whose parameters are a delayed
// copy of the learner's, so the value being regressed onto does not move with every update.
class DeepQLearning {
  public:
    struct Config {
        std::vector<std::size_t> hidden{64, 64};
        double learning_rate{0.001};
        double discount{0.95};
        ExplorationSchedule exploration{};
        std::size_t replay_capacity{20'000};
        // Transitions per update, drawn uniformly from the buffer. One means no batching.
        std::size_t batch_size{32};
        // Transitions to collect before any learning starts, so the first batches are not drawn
        // from a handful of near-identical states. It cannot exceed the replay capacity, which a
        // buffer the size of one batch — the no-replay ablation — makes easy to trip over.
        std::size_t warmup{500};
        // Updates between copies of the learner's parameters into the target network. Zero makes
        // the target network the learner itself, which is the ablation.
        std::size_t target_update{200};
        // Global-norm clipping of each update's gradient; zero disables it.
        double clip_norm{10.0};
        std::uint32_t seed{0};
    };

    DeepQLearning();
    explicit DeepQLearning(Config config);

    [[nodiscard]] std::vector<double> values(const std::vector<double>& features) const;
    [[nodiscard]] std::size_t act(const std::vector<double>& features) const;
    [[nodiscard]] std::size_t explore(const std::vector<double>& features);
    // Stores the transition and, once past the warmup, applies one batched update. Returns the
    // mean squared temporal-difference error of that batch, or zero when none was applied.
    double observe(Transition transition);

    [[nodiscard]] double epsilon() const noexcept;
    [[nodiscard]] std::size_t selections() const noexcept { return selections_; }
    [[nodiscard]] std::size_t updates() const noexcept { return updates_; }
    [[nodiscard]] const FeedForwardNetwork& network() const noexcept { return network_; }
    [[nodiscard]] const ReplayBuffer& replay() const noexcept { return replay_; }

    void save(const std::string& path) const;
    [[nodiscard]] static DeepQLearning load(const std::string& path, Config config);

  private:
    [[nodiscard]] static std::vector<DenseLayer> layers(const Config& config);

    Config config_;
    FeedForwardNetwork network_;
    FeedForwardNetwork target_;
    Optimizer optimizer_;
    ReplayBuffer replay_;
    std::uint64_t random_state_;
    std::size_t selections_{0};
    std::size_t updates_{0};
};

// ---------------------------------------------------------------------------------------------

template <typename Policy>
EpisodeStatistics evaluate(Policy&& policy, const SnakeConfig& config, const std::size_t episodes,
                           const std::uint64_t seed) {
    double return_total = 0.0;
    double score_total = 0.0;
    double step_total = 0.0;
    double best = 0.0;
    std::size_t deaths = 0;
    for (std::size_t episode = 0; episode < episodes; ++episode) {
        SnakeGame game{config, seed + episode};
        double episode_return = 0.0;
        SnakeStepResult result{};
        while (!game.finished()) {
            result = game.step(static_cast<SnakeAction>(policy(game)));
            episode_return += result.reward;
        }
        return_total += episode_return;
        score_total += static_cast<double>(game.score());
        step_total += static_cast<double>(game.steps());
        best = std::max(best, static_cast<double>(game.score()));
        deaths += result.died ? 1 : 0;
    }
    const auto count = static_cast<double>(episodes);
    return {return_total / count, score_total / count, best, step_total / count,
            static_cast<double>(deaths) / count};
}

} // namespace ml_scratch
