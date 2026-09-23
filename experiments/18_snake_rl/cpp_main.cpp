#include "ml_scratch/reinforcement.hpp"
#include "ml_scratch/snake.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t seed = 20260922;
constexpr ml_scratch::SnakeConfig board{10, 10, 1.0, -1.0, 0.0, 100};

// Training runs this many episodes for every learner, with a greedy evaluation every
// `evaluation_every` of them on a fixed set of held-out seeds.
constexpr std::size_t episodes = 3'000;
// The table is cheap enough per episode that the same wall-clock budget as the network buys it
// far more experience; this is how many episodes it gets in the equal-cost comparison.
constexpr std::size_t long_tabular_episodes = 30'000;
constexpr std::size_t evaluation_every = 150;
constexpr std::size_t evaluation_episodes = 200;
// The evaluation seeds are far from the training ones, so no agent is scored on an episode it
// trained on.
constexpr std::uint64_t evaluation_seed = 900'000;
constexpr std::uint64_t training_seed = 1;
// The last this many evaluations measure how settled a run is: an agent that oscillates between
// good and collapsed policies has a high spread here however good its final snapshot looks.
constexpr std::size_t settled_window = 6;

constexpr double discount = 0.95;
constexpr ml_scratch::ExplorationSchedule schedule{1.0, 0.02, 30'000};

std::string data_path(const char* variable, const std::string& fallback) {
    if (const char* override_path = std::getenv(variable)) {
        return override_path;
    }
    return fallback;
}

struct Curve {
    std::string label;
    std::vector<double> scores; // greedy mean score at each evaluation
    // The full statistics of each evaluation, so a collapsed snapshot can be told apart from a
    // merely worse one: an agent that circles has a low death rate, one that dies has a high one.
    std::vector<ml_scratch::EpisodeStatistics> snapshots;
    ml_scratch::EpisodeStatistics final_statistics{};
    double seconds{0.0};
    std::size_t updates{0};

    [[nodiscard]] double settled_mean() const {
        const std::size_t count = std::min(settled_window, scores.size());
        double total = 0.0;
        for (std::size_t index = scores.size() - count; index < scores.size(); ++index) {
            total += scores[index] / static_cast<double>(count);
        }
        return total;
    }
    [[nodiscard]] double settled_spread() const {
        const std::size_t count = std::min(settled_window, scores.size());
        const double mean = settled_mean();
        double total = 0.0;
        for (std::size_t index = scores.size() - count; index < scores.size(); ++index) {
            const double difference = scores[index] - mean;
            total += difference * difference / static_cast<double>(count);
        }
        return std::sqrt(total);
    }
    [[nodiscard]] double worst_settled() const {
        const std::size_t count = std::min(settled_window, scores.size());
        return *std::min_element(scores.end() - static_cast<std::ptrdiff_t>(count), scores.end());
    }
    [[nodiscard]] const ml_scratch::EpisodeStatistics& worst_snapshot() const {
        const std::size_t count = std::min(settled_window, scores.size());
        const auto worst = std::min_element(scores.end() - static_cast<std::ptrdiff_t>(count),
                                            scores.end());
        return snapshots[static_cast<std::size_t>(worst - scores.begin())];
    }
};

void print_statistics_header() {
    std::cout << "  policy                     mean score   best   mean return   mean steps   "
                 "death rate\n";
}

void print_statistics(const std::string& label, const ml_scratch::EpisodeStatistics& statistics) {
    std::string padded = label;
    padded.resize(25, ' ');
    std::cout << "  " << padded << std::setw(13) << statistics.mean_score << std::setw(7)
              << static_cast<int>(statistics.best_score) << std::setw(14)
              << statistics.mean_return << std::setw(13) << statistics.mean_steps << std::setw(13)
              << statistics.death_rate << '\n';
}

void print_curve_header() {
    std::cout << "  episodes";
    for (std::size_t index = 1; index * evaluation_every <= episodes; ++index) {
        if ((index * evaluation_every) % 600 == 0) {
            std::cout << std::setw(8) << index * evaluation_every;
        }
    }
    std::cout << '\n';
}

void print_curve(const Curve& curve) {
    std::string padded = curve.label;
    padded.resize(10, ' ');
    std::cout << "  " << padded;
    for (std::size_t index = 0; index < curve.scores.size(); ++index) {
        if (((index + 1) * evaluation_every) % 600 == 0) {
            std::cout << std::setw(8) << std::setprecision(1) << curve.scores[index];
        }
    }
    std::cout << std::setprecision(2) << '\n';
}

// Trains a tabular agent, evaluating greedily along the way.
Curve train_tabular(const std::string& label, ml_scratch::TabularQLearning& agent,
                    const std::size_t count = episodes) {
    Curve curve{label, {}, {}, {}, 0.0, 0};
    ml_scratch::SnakeGame game{board, training_seed};
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t episode = 1; episode <= count; ++episode) {
        game.reset();
        while (!game.finished()) {
            const std::size_t state = game.state_index();
            const std::size_t action = agent.explore(state);
            const auto result = game.step(static_cast<ml_scratch::SnakeAction>(action));
            agent.learn({{}, state, action, result.reward, {}, game.state_index(), result.died});
            ++curve.updates;
        }
        if (episode % (evaluation_every * count / episodes) == 0) {
            const auto statistics = ml_scratch::evaluate(
                [&agent](const ml_scratch::SnakeGame& state) { return agent.act(state.state_index()); },
                board, evaluation_episodes, evaluation_seed);
            curve.scores.push_back(statistics.mean_score);
            curve.snapshots.push_back(statistics);
            curve.final_statistics = statistics;
        }
    }
    const auto finish = std::chrono::steady_clock::now();
    curve.seconds = std::chrono::duration<double>(finish - start).count();
    return curve;
}

Curve train_deep(const std::string& label, ml_scratch::DeepQLearning& agent) {
    Curve curve{label, {}, {}, {}, 0.0, 0};
    ml_scratch::SnakeGame game{board, training_seed};
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t episode = 1; episode <= episodes; ++episode) {
        game.reset();
        while (!game.finished()) {
            std::vector<double> features = game.observation();
            const std::size_t action = agent.explore(features);
            const auto result = game.step(static_cast<ml_scratch::SnakeAction>(action));
            agent.observe({std::move(features), 0, action, result.reward, game.observation(), 0,
                           result.died});
        }
        if (episode % evaluation_every == 0) {
            const auto statistics = ml_scratch::evaluate(
                [&agent](const ml_scratch::SnakeGame& state) { return agent.act(state.observation()); },
                board, evaluation_episodes, evaluation_seed);
            curve.scores.push_back(statistics.mean_score);
            curve.snapshots.push_back(statistics);
            curve.final_statistics = statistics;
        }
    }
    const auto finish = std::chrono::steady_clock::now();
    curve.seconds = std::chrono::duration<double>(finish - start).count();
    curve.updates = agent.updates();
    return curve;
}

void print_settled_header() {
    std::cout << "  run                        final   settled mean   spread   worst   seconds   "
                 "updates     the last six evaluations\n";
}

void print_settled(const Curve& curve) {
    std::string padded = curve.label;
    padded.resize(25, ' ');
    std::cout << "  " << padded << std::setw(8) << curve.scores.back() << std::setw(15)
              << curve.settled_mean() << std::setw(9) << curve.settled_spread() << std::setw(8)
              << curve.worst_settled() << std::setw(10) << std::setprecision(2) << curve.seconds
              << std::setw(10) << curve.updates << "    ";
    const std::size_t shown = std::min(settled_window, curve.scores.size());
    for (std::size_t index = curve.scores.size() - shown; index < curve.scores.size(); ++index) {
        std::cout << std::setw(7) << std::setprecision(1) << curve.scores[index];
    }
    std::cout << std::setprecision(2) << '\n';
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(2)
              << "C++ reinforcement learning on Snake (seed=" << seed << ")\n\n"
              << "  board " << board.width << "x" << board.height << ", three relative actions, "
              << "+1 for food, " << board.death_reward << " for dying, 0 per step; an episode is "
              << "truncated after\n"
              << "  " << board.steps_without_food
              << " steps without eating, times one plus the score, and truncation pays no "
                 "penalty\n"
              << "  every learner: " << episodes << " training episodes, discount " << discount
              << ", epsilon " << schedule.start << " to " << schedule.end << " over "
              << schedule.steps << " selections\n"
              << "  every evaluation: " << evaluation_episodes
              << " greedy episodes on fixed held-out seeds, so what is reported is the policy "
                 "rather than\n"
              << "  the policy plus its exploration; the best possible score on this board is "
              << ml_scratch::SnakeGame{board}.maximum_score() << "\n\n";

    // ---------- Part one: baselines ----------
    std::cout << "part one: baselines\n";
    print_statistics_header();
    std::uint64_t random_state = seed;
    const auto random_policy = [&random_state](const ml_scratch::SnakeGame&) {
        random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<int>((random_state >> 33) % ml_scratch::snake_action_count);
    };
    const auto random_statistics =
        ml_scratch::evaluate(random_policy, board, evaluation_episodes, evaluation_seed);
    print_statistics("random", random_statistics);

    // A scripted policy with no learning in it: head for the food, and refuse a move that would
    // end the episode. It knows the rules; what it does not know is anything about the future.
    const auto scripted = [](const ml_scratch::SnakeGame& game) {
        const std::vector<double> features = game.observation();
        const bool want_up = features[7] > 0.5;
        const bool want_right = features[8] > 0.5;
        const bool want_down = features[9] > 0.5;
        const auto heading = static_cast<int>(game.heading());
        // The heading each action would produce, in action order.
        const int headings[] = {heading, (heading + 1) % 4, (heading + 3) % 4};
        const int wanted = want_up ? 0 : want_right ? 1 : want_down ? 2 : 3;
        for (int action = 0; action < 3; ++action) {
            if (headings[action] == wanted && features[action] < 0.5) {
                return action;
            }
        }
        for (int action = 0; action < 3; ++action) {
            if (features[action] < 0.5) {
                return action;
            }
        }
        return 0;
    };
    const auto scripted_statistics =
        ml_scratch::evaluate(scripted, board, evaluation_episodes, evaluation_seed);
    print_statistics("scripted food-seeker", scripted_statistics);

    // ---------- Part two: tabular Q-learning ----------
    std::cout << "\npart two: tabular Q-learning over the " << ml_scratch::SnakeGame::state_count
              << " observation bit patterns\n";
    ml_scratch::TabularQLearning::Config tabular_config;
    tabular_config.learning_rate = 0.1;
    tabular_config.discount = discount;
    tabular_config.exploration = schedule;
    tabular_config.seed = seed;
    ml_scratch::TabularQLearning tabular{tabular_config};
    const Curve tabular_curve = train_tabular("tabular", tabular);
    std::cout << "  " << tabular.visited_states() << " of "
              << ml_scratch::SnakeGame::state_count
              << " states were ever updated: the observation's eleven bits are not independent, "
                 "since the\n"
              << "  heading is one-hot and the food cannot be both left and right, so only "
              << tabular.visited_states() << " patterns can occur at all\n";
    print_statistics_header();
    print_statistics("tabular, greedy", tabular_curve.final_statistics);

    // ---------- Part three: deep Q-learning ----------
    std::cout << "\npart three: deep Q-learning, two 64-unit hidden layers on the same eleven "
                 "features\n";
    ml_scratch::DeepQLearning::Config deep_config;
    deep_config.hidden = {64, 64};
    deep_config.learning_rate = 0.001;
    deep_config.discount = discount;
    deep_config.exploration = schedule;
    deep_config.seed = seed;
    ml_scratch::DeepQLearning deep{deep_config};
    const Curve deep_curve = train_deep("DQN", deep);
    print_statistics_header();
    print_statistics("DQN, greedy", deep_curve.final_statistics);

    std::cout << "\n  greedy mean score every " << 600 / evaluation_every << " evaluations\n";
    print_curve_header();
    print_curve(tabular_curve);
    print_curve(deep_curve);

    // The table is still climbing at 3000 episodes. Since an episode costs it almost nothing,
    // the fair question is what it reaches for the same wall-clock budget the network spent.
    ml_scratch::TabularQLearning patient{tabular_config};
    const Curve patient_curve =
        train_tabular("tabular, 30k", patient, long_tabular_episodes);
    std::cout << "\n  the same table trained for " << long_tabular_episodes << " episodes instead, "
              << "in " << patient_curve.seconds << " s against the DQN's " << deep_curve.seconds
              << " s\n";
    print_statistics_header();
    print_statistics("tabular 30k, greedy", patient_curve.final_statistics);

    const std::string checkpoints = data_path("ML_SCRATCH_SNAKE_CHECKPOINTS", "data");
    try {
        tabular.save(checkpoints + "/snake_tabular.checkpoint");
        deep.save(checkpoints + "/snake_dqn.checkpoint");
        const auto restored_table =
            ml_scratch::TabularQLearning::load(checkpoints + "/snake_tabular.checkpoint");
        const auto restored_network =
            ml_scratch::DeepQLearning::load(checkpoints + "/snake_dqn.checkpoint", deep_config);
        const auto replayed = ml_scratch::evaluate(
            [&restored_network](const ml_scratch::SnakeGame& state) {
                return restored_network.act(state.observation());
            },
            board, evaluation_episodes, evaluation_seed);
        std::cout << "  checkpoints round-trip: table "
                  << (restored_table.table() == tabular.table() ? "yes" : "NO") << ", network "
                  << (replayed.mean_score == deep_curve.final_statistics.mean_score ? "yes" : "NO")
                  << '\n';
    } catch (const std::exception& error) {
        std::cout << "  checkpoints not written: " << error.what() << '\n';
    }

    // ---------- Part four: what the deep agent needs ----------
    std::cout << "\npart four: the two mechanisms that make a network trainable here, removed one "
                 "at a time\n"
              << "  spread and worst are over the last " << settled_window
              << " evaluations: a run that oscillates between good and collapsed policies has a "
                 "high spread\n"
              << "  however good the snapshot it happens to end on\n";
    ml_scratch::DeepQLearning::Config no_target = deep_config;
    no_target.target_update = 0;
    ml_scratch::DeepQLearning without_target{no_target};
    const Curve no_target_curve = train_deep("no target network", without_target);

    ml_scratch::DeepQLearning::Config no_replay = deep_config;
    // A buffer the size of one batch is no replay at all: every update sees the last 32
    // transitions, which come from one stretch of one episode.
    no_replay.replay_capacity = no_replay.batch_size;
    no_replay.warmup = no_replay.batch_size;
    ml_scratch::DeepQLearning without_replay{no_replay};
    const Curve no_replay_curve = train_deep("no replay", without_replay);

    ml_scratch::DeepQLearning::Config no_exploration = deep_config;
    no_exploration.exploration = {schedule.end, schedule.end, 0};
    ml_scratch::DeepQLearning without_exploration{no_exploration};
    const Curve no_exploration_curve = train_deep("no exploration decay", without_exploration);

    std::cout << '\n';
    print_settled_header();
    print_settled(deep_curve);
    print_settled(no_target_curve);
    print_settled(no_replay_curve);
    print_settled(no_exploration_curve);
    print_settled(tabular_curve);
    print_settled(patient_curve);

    // ---------- One episode, drawn ----------
    {
        ml_scratch::SnakeGame game{board, evaluation_seed};
        while (!game.finished() && game.steps() < 400) {
            game.step(static_cast<ml_scratch::SnakeAction>(tabular.act(game.state_index())));
        }
        std::cout << "\n  the tabular agent's first evaluation episode, at its end: score "
                  << game.score() << " in " << game.steps() << " steps\n"
                  << game.render();
    }

    // ---------- Summary and criteria ----------
    std::cout << "\nsummary\n"
              << "  greedy mean score: random " << random_statistics.mean_score << ", scripted "
              << scripted_statistics.mean_score << ", tabular "
              << tabular_curve.final_statistics.mean_score << ", DQN "
              << deep_curve.final_statistics.mean_score << " (best possible "
              << ml_scratch::SnakeGame{board}.maximum_score() << ")\n"
              << "  training cost: tabular " << tabular_curve.seconds << " s for "
              << tabular_curve.updates << " updates, DQN " << deep_curve.seconds << " s for "
              << deep_curve.updates << " batched updates — " << std::setprecision(0)
              << deep_curve.seconds / tabular_curve.seconds << "x" << std::setprecision(2)
              << '\n'
              << "  settled spread over the last " << settled_window << " evaluations: DQN "
              << deep_curve.settled_spread() << ", without replay "
              << no_replay_curve.settled_spread() << ", without a target network "
              << no_target_curve.settled_spread() << ", tabular "
              << tabular_curve.settled_spread() << "; the no-replay run dipped "
              << no_replay_curve.settled_mean() - no_replay_curve.worst_settled()
              << " below its own mean against the DQN's "
              << deep_curve.settled_mean() - deep_curve.worst_settled()
              << ", and at its worst evaluation its death rate was "
              << no_replay_curve.worst_snapshot().death_rate << " over "
              << no_replay_curve.worst_snapshot().mean_steps << " steps an episode\n"
              << "  neither the target network nor the exploration schedule changed the outcome "
                 "here: without them the DQN settled at "
              << no_target_curve.settled_mean() << " and " << no_exploration_curve.settled_mean()
              << " against " << deep_curve.settled_mean() << " with both\n"
              << "  the same table given " << long_tabular_episodes << " episodes reached "
              << patient_curve.final_statistics.mean_score << " in " << patient_curve.seconds
              << " s, still " << std::setprecision(0)
              << deep_curve.seconds / patient_curve.seconds << "x cheaper than the network"
              << std::setprecision(2) << '\n';

    // Both learners have to beat the random baseline by a wide margin: that is the whole claim
    // that anything was learned. Only the network beats the scripted food-seeker, which knows
    // the rules; the table's shortfall against it is reported rather than asserted away.
    const bool beat_random =
        tabular_curve.final_statistics.mean_score > 10.0 * random_statistics.mean_score &&
        deep_curve.final_statistics.mean_score > 10.0 * random_statistics.mean_score;
    const bool network_beat_the_scripted_policy =
        deep_curve.final_statistics.mean_score > scripted_statistics.mean_score;
    const bool network_beat_the_table =
        deep_curve.final_statistics.mean_score > tabular_curve.final_statistics.mean_score;
    // Given enough episodes the table catches up: same policy class, far less compute.
    const bool table_catches_up =
        patient_curve.final_statistics.mean_score > tabular_curve.final_statistics.mean_score;
    // The random policy dies almost every episode and eats almost nothing.
    const bool random_is_bad = random_statistics.mean_score < 2.0;
    // Replay is what the network cannot do without: removing it has to make the run less
    // settled, both in the spread of its last evaluations and in how far it dips below its own
    // mean — the measure of a run that keeps collapsing and recovering.
    const auto dip = [](const Curve& curve) {
        return curve.settled_mean() - curve.worst_settled();
    };
    const bool replay_matters =
        no_replay_curve.settled_spread() > 2.0 * deep_curve.settled_spread() &&
        dip(no_replay_curve) > 2.0 * dip(deep_curve);
    // The table reached only the reachable states, and every one of them.
    const bool table_is_small = tabular.visited_states() == 256;
    // The table is orders of magnitude cheaper per episode than the network.
    const bool table_is_cheaper = deep_curve.seconds > 20.0 * tabular_curve.seconds;

    if (!beat_random || !network_beat_the_scripted_policy || !network_beat_the_table ||
        !table_catches_up || !random_is_bad || !replay_matters || !table_is_small ||
        !table_is_cheaper) {
        std::cerr << "experiment failed its criteria: random " << random_statistics.mean_score
                  << ", scripted " << scripted_statistics.mean_score << ", tabular "
                  << tabular_curve.final_statistics.mean_score << ", DQN "
                  << deep_curve.final_statistics.mean_score << ", spreads "
                  << deep_curve.settled_spread() << '/' << no_replay_curve.settled_spread()
                  << ", visited " << tabular.visited_states() << ", seconds "
                  << tabular_curve.seconds << '/' << deep_curve.seconds << '\n';
        return 1;
    }
    return 0;
}
