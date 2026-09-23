#include "ml_scratch/reinforcement.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

void require_near(const double actual, const double expected, const double tolerance,
                  const std::string_view message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void require_invalid_argument(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

ml_scratch::Transition transition(const std::size_t state, const std::size_t action,
                                  const double reward, const std::size_t next,
                                  const bool died) {
    std::vector<double> features(ml_scratch::SnakeGame::observation_size, 0.0);
    std::vector<double> next_features(ml_scratch::SnakeGame::observation_size, 0.0);
    // A distinguishable feature vector per state, so a network sees different inputs.
    features[state % features.size()] = 1.0;
    next_features[next % next_features.size()] = 1.0;
    return {features, state, action, reward, next_features, next, died};
}

void test_exploration_schedule() {
    const ml_scratch::ExplorationSchedule schedule{1.0, 0.1, 100};
    require_near(schedule.epsilon(0), 1.0, 1e-15, "the schedule starts at its start");
    require_near(schedule.epsilon(50), 0.55, 1e-15, "it decays linearly");
    require_near(schedule.epsilon(100), 0.1, 1e-15, "it reaches its end on time");
    require_near(schedule.epsilon(10'000), 0.1, 1e-15, "and stays there");
    const ml_scratch::ExplorationSchedule immediate{1.0, 0.2, 0};
    require_near(immediate.epsilon(0), 0.2, 1e-15, "zero steps means no exploration decay");
}

void test_tabular_update_by_hand() {
    ml_scratch::TabularQLearning::Config config;
    config.learning_rate = 0.5;
    config.discount = 0.9;
    config.exploration = {0.0, 0.0, 0}; // greedy, so `explore` is `act`
    ml_scratch::TabularQLearning agent{config};

    require(agent.act(0) == 0, "a fresh table breaks ties toward the lowest action");
    require(agent.value(0, 0) == 0.0 && agent.visited_states() == 0, "a fresh table is empty");

    // A terminal transition: the target is the reward alone.
    double error = agent.learn(transition(7, 1, -1.0, 9, true));
    require_near(error, -1.0, 1e-15, "the error is the target minus the old value");
    require_near(agent.value(7, 1), -0.5, 1e-15, "half the error is applied at alpha 0.5");
    require(agent.visited_states() == 1, "the visited count follows the updates");
    require(agent.act(7) == 0, "the punished action is no longer greedy");

    // A non-terminal transition bootstraps the best value of the next state.
    agent.learn(transition(9, 2, 1.0, 9, true)); // sets Q(9, 2) = 0.5
    require_near(agent.value(9, 2), 0.5, 1e-15, "Q(9, 2) is half the reward");
    error = agent.learn(transition(3, 0, 1.0, 9, false));
    // target = 1 + 0.9 * max(0, 0, 0.5) = 1.45; Q was 0, so the error is 1.45 and half applies.
    require_near(error, 1.45, 1e-12, "a live transition bootstraps the next state's best value");
    require_near(agent.value(3, 0), 0.725, 1e-12, "and the update applies alpha to it");
    // The same transition marked terminal ignores that future entirely.
    ml_scratch::TabularQLearning fresh{config};
    fresh.learn(transition(9, 2, 1.0, 9, true));
    require_near(fresh.learn(transition(3, 0, 1.0, 9, true)), 1.0, 1e-15,
                 "a terminal transition takes the reward alone as its target");

    require_invalid_argument(
        [] {
            ml_scratch::TabularQLearning::Config bad;
            bad.discount = 1.5;
            ml_scratch::TabularQLearning agent{bad};
        },
        "a discount above one is rejected");
    require_invalid_argument([&] { static_cast<void>(agent.act(99'999)); },
                             "an out-of-range state is rejected");
}

void test_exploration_is_seeded_and_decays() {
    ml_scratch::TabularQLearning::Config config;
    config.exploration = {1.0, 0.0, 100};
    config.seed = 4;
    ml_scratch::TabularQLearning first{config};
    ml_scratch::TabularQLearning second{config};
    std::vector<std::size_t> actions;
    for (std::size_t step = 0; step < 100; ++step) {
        actions.push_back(first.explore(step % 8));
    }
    std::vector<std::size_t> repeated;
    for (std::size_t step = 0; step < 100; ++step) {
        repeated.push_back(second.explore(step % 8));
    }
    require(actions == repeated, "the same seed explores identically");
    require(first.selections() == 100, "selections are counted");
    require_near(first.epsilon(), 0.0, 1e-15, "the schedule has run out");
    // At epsilon 1 the actions are not all the same, which is what exploring means.
    require(std::count(actions.begin(), actions.end(), actions.front()) < 90,
            "a fully exploring agent does not repeat one action");

    ml_scratch::TabularQLearning::Config other = config;
    other.seed = 5;
    ml_scratch::TabularQLearning different{other};
    std::vector<std::size_t> others;
    for (std::size_t step = 0; step < 100; ++step) {
        others.push_back(different.explore(step % 8));
    }
    require(others != actions, "a different seed explores differently");
}

void test_replay_buffer() {
    ml_scratch::ReplayBuffer buffer{4, 1};
    require(buffer.capacity() == 4 && buffer.size() == 0, "a fresh buffer is empty");
    bool threw = false;
    try {
        static_cast<void>(buffer.sample(1));
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "an empty buffer cannot be sampled");

    for (std::size_t index = 0; index < 4; ++index) {
        buffer.push(transition(index, 0, static_cast<double>(index), index, false));
    }
    require(buffer.size() == 4, "the buffer fills to its capacity");
    // Overwriting the oldest: pushing two more leaves states 2, 3, 4, 5.
    buffer.push(transition(4, 0, 4.0, 4, false));
    buffer.push(transition(5, 0, 5.0, 5, false));
    require(buffer.size() == 4, "the buffer does not grow past its capacity");
    std::vector<bool> seen(6, false);
    for (std::size_t draw = 0; draw < 400; ++draw) {
        for (const auto* sample : buffer.sample(4)) {
            seen[sample->state] = true;
        }
    }
    require(!seen[0] && !seen[1] && seen[2] && seen[3] && seen[4] && seen[5],
            "the two oldest transitions were overwritten and the newest four remain");
    ml_scratch::ReplayBuffer same{4, 1};
    ml_scratch::ReplayBuffer copy{4, 1};
    for (std::size_t index = 0; index < 4; ++index) {
        same.push(transition(index, 0, 0.0, index, false));
        copy.push(transition(index, 0, 0.0, index, false));
    }
    require(same.sample(8).size() == 8, "a sample can be larger than the buffer");
    require_invalid_argument([] { ml_scratch::ReplayBuffer bad{0, 1}; },
                             "a zero capacity is rejected");
}

// The network has to learn action values, not a policy: a state whose only reward comes from one
// action must end with that action's value above the others'.
void test_deep_q_learns_action_values() {
    ml_scratch::DeepQLearning::Config config;
    config.hidden = {32};
    config.learning_rate = 0.01;
    config.discount = 0.0; // one-step: the target is the reward, so the values are learnable exactly
    config.warmup = 32;
    config.batch_size = 16;
    config.replay_capacity = 2'000;
    config.exploration = {0.0, 0.0, 0};
    config.seed = 3;
    ml_scratch::DeepQLearning agent{config};

    // Two distinguishable states; in state A action 1 pays, in state B action 2 pays.
    for (std::size_t step = 0; step < 4'000; ++step) {
        const std::size_t state = step % 2 == 0 ? 0 : 1;
        const std::size_t action = step % ml_scratch::snake_action_count;
        const double reward = (state == 0 && action == 1) || (state == 1 && action == 2) ? 1.0 : 0.0;
        agent.observe(transition(state, action, reward, state, true));
    }
    std::vector<double> first(ml_scratch::SnakeGame::observation_size, 0.0);
    first[0] = 1.0;
    std::vector<double> second(ml_scratch::SnakeGame::observation_size, 0.0);
    second[1] = 1.0;
    require(agent.act(first) == 1 && agent.act(second) == 2,
            "the network should prefer the rewarding action in each state");
    const auto values = agent.values(first);
    require_near(values[1], 1.0, 0.2, "a discount of zero makes the value the reward itself");
    require_near(values[0], 0.0, 0.2, "an unrewarded action's value should stay near zero");
    require(agent.updates() > 3'000 && agent.replay().size() == 2'000,
            "an update per observation past the warmup, and a full buffer");

    // The warmup really holds learning off.
    ml_scratch::DeepQLearning waiting{config};
    for (std::size_t step = 0; step < 10; ++step) {
        require(waiting.observe(transition(0, 0, 1.0, 0, true)) == 0.0,
                "nothing is learned during the warmup");
    }
    require(waiting.updates() == 0, "and no updates are counted");
}

void test_deep_q_is_seeded_and_checkpoints() {
    ml_scratch::DeepQLearning::Config config;
    config.hidden = {16};
    config.warmup = 8;
    config.batch_size = 4;
    config.seed = 12;
    ml_scratch::DeepQLearning first{config};
    ml_scratch::DeepQLearning second{config};
    require(first.network().parameters() == second.network().parameters(),
            "the same seed gives the same network");
    std::vector<double> errors;
    std::vector<double> repeated;
    for (std::size_t step = 0; step < 200; ++step) {
        errors.push_back(first.observe(transition(step % 5, step % 3, 0.5, (step + 1) % 5, false)));
        repeated.push_back(
            second.observe(transition(step % 5, step % 3, 0.5, (step + 1) % 5, false)));
    }
    require(errors == repeated && first.network().parameters() == second.network().parameters(),
            "learning is deterministic");

    ml_scratch::DeepQLearning::Config other = config;
    other.seed = 13;
    require(ml_scratch::DeepQLearning{other}.network().parameters() !=
                first.network().parameters(),
            "a different seed gives a different network");

    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_dqn_test.checkpoint").string();
    first.save(path);
    const auto restored = ml_scratch::DeepQLearning::load(path, config);
    require(restored.network().parameters() == first.network().parameters(),
            "a checkpoint round-trips the network");
    std::vector<double> features(ml_scratch::SnakeGame::observation_size, 0.0);
    features[2] = 1.0;
    require(restored.values(features) == first.values(features),
            "the restored agent values states identically");
    std::filesystem::remove(path);

    // A checkpoint of the wrong shape is refused.
    ml_scratch::DeepQLearning::Config bigger = config;
    bigger.hidden = {32, 32};
    bool threw = false;
    try {
        ml_scratch::DeepQLearning{bigger}.save(path);
        static_cast<void>(ml_scratch::DeepQLearning::load(path, config));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "a checkpoint of another architecture is rejected");
    std::filesystem::remove(path);
}

void test_tabular_checkpoint() {
    ml_scratch::TabularQLearning::Config config;
    config.seed = 2;
    ml_scratch::TabularQLearning agent{config};
    for (std::size_t step = 0; step < 50; ++step) {
        agent.learn(transition(step * 7 % 2048, step % 3, 0.25, (step * 13) % 2048, step % 5 == 0));
    }
    const std::string path =
        (std::filesystem::temp_directory_path() / "ml_scratch_tabular_test.checkpoint").string();
    agent.save(path);
    const auto restored = ml_scratch::TabularQLearning::load(path);
    require(restored.table() == agent.table() &&
                restored.visited_states() == agent.visited_states(),
            "a checkpoint round-trips the table exactly");
    std::filesystem::remove(path);

    const std::string broken =
        (std::filesystem::temp_directory_path() / "ml_scratch_tabular_broken.checkpoint").string();
    {
        std::ofstream stream{broken};
        stream << "ml_scratch_tabular_q 1\nstates 7\nactions 3\nselections 0\n";
    }
    bool threw = false;
    try {
        static_cast<void>(ml_scratch::TabularQLearning::load(broken));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw, "a checkpoint with the wrong state count is rejected");
    std::filesystem::remove(broken);
}

// Evaluation plays the policy greedily on fixed seeds, so it measures the policy rather than the
// policy plus its exploration noise, and two agents see identical food.
void test_evaluation_is_greedy_and_fixed() {
    const ml_scratch::SnakeConfig config{10, 10};
    const auto always_straight = [](const ml_scratch::SnakeGame&) { return 0; };
    const auto statistics = ml_scratch::evaluate(always_straight, config, 8, 100);
    require(statistics.death_rate == 1.0, "a snake that never turns always hits a wall");
    require_near(statistics.mean_steps, 5.0, 1e-12, "it takes five steps to leave the board");
    // It scores only when a food happens to sit on its row ahead of it, and its return is that
    // score less the one death.
    require(statistics.mean_score < 1.0, "it cannot eat more than the food in its path");
    require_near(statistics.mean_return, statistics.mean_score - 1.0, 1e-12,
                 "the return is the food eaten less the death");

    const auto repeated = ml_scratch::evaluate(always_straight, config, 8, 100);
    require(repeated.mean_return == statistics.mean_return &&
                repeated.mean_steps == statistics.mean_steps,
            "the same seeds give the same evaluation");

    // A circling policy never dies but never eats: the two failure modes are distinguishable.
    std::size_t counter = 0;
    const auto circle = [&counter](const ml_scratch::SnakeGame&) {
        return counter++ % 4 == 0 ? 1 : 0;
    };
    const auto circling = ml_scratch::evaluate(circle, config, 4, 7);
    require(circling.death_rate < 1.0 && circling.mean_score == 0.0,
            "a circling policy is caught by the score, not by the death rate");
}

} // namespace

int main() {
    try {
        test_exploration_schedule();
        test_tabular_update_by_hand();
        test_exploration_is_seeded_and_decays();
        test_replay_buffer();
        test_deep_q_learns_action_values();
        test_deep_q_is_seeded_and_checkpoints();
        test_tabular_checkpoint();
        test_evaluation_is_greedy_and_fixed();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
