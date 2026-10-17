#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace ml_scratch {

// The four headings a snake can travel in. Turning is relative to the current one, so the snake
// can never reverse into its own neck — a move that would end every episode at step two and
// teach the agent nothing.
enum class Heading { up, right, down, left };

// The three moves available at each step.
enum class SnakeAction { straight, turn_right, turn_left };

inline constexpr std::size_t snake_action_count = 3;

struct Cell {
    int x;
    int y;

    bool operator==(const Cell&) const = default;
};

struct SnakeStepResult {
    double reward;
    // The episode ended because the snake hit a wall or itself.
    bool died{false};
    // The episode was cut off at the step limit instead: not a failure of the policy, and the
    // distinction matters because a bootstrapped value must not treat a cut-off as terminal.
    bool truncated{false};
    bool ate{false};

    [[nodiscard]] bool finished() const noexcept { return died || truncated; }
};

struct SnakeConfig {
    std::size_t width{10};
    std::size_t height{10};
    double food_reward{1.0};
    double death_reward{-1.0};
    // Applied every step, food or not. A small negative value is what stops a policy from
    // circling forever; zero leaves circling costless.
    double step_reward{0.0};
    // Steps without eating before the episode is truncated. Scales with the board so a longer
    // snake gets proportionally more time to reach its food.
    std::size_t steps_without_food{100};

    bool operator==(const SnakeConfig&) const = default;
};

// A snake on a grid. The snake starts at the centre with a body of three cells heading right,
// and one food cell is placed uniformly at random over the free squares. Everything about an
// episode is a function of the seed, so two agents can be compared on identical food sequences.
class SnakeGame {
  public:
    explicit SnakeGame(SnakeConfig config = {}, std::uint64_t seed = 0);

    // Starts a new episode. With a seed, the food sequence restarts from it; without one it
    // continues from the current state of the stream, so consecutive episodes differ.
    void reset();
    void reset(std::uint64_t seed);

    SnakeStepResult step(SnakeAction action);

    // The eleven-feature observation both agents read: three danger flags for the cells the
    // three actions would move into, four one-hot heading flags, and four flags for where the
    // food lies relative to the head. Every value is zero or one, so the tabular agent can index
    // a table by their bit pattern and the network can take them as inputs unchanged.
    [[nodiscard]] std::vector<double> observation() const;
    // The same eleven bits packed into an index in [0, 2048).
    [[nodiscard]] std::size_t state_index() const;
    static constexpr std::size_t observation_size = 11;
    static constexpr std::size_t state_count = 1U << observation_size;

    [[nodiscard]] const std::deque<Cell>& body() const noexcept { return body_; }
    [[nodiscard]] Cell head() const noexcept { return body_.front(); }
    [[nodiscard]] Cell food() const noexcept { return food_; }
    [[nodiscard]] Heading heading() const noexcept { return heading_; }
    [[nodiscard]] std::size_t score() const noexcept { return score_; }
    [[nodiscard]] std::size_t steps() const noexcept { return steps_; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }
    [[nodiscard]] const SnakeConfig& config() const noexcept { return config_; }
    // Longest possible score: every cell but the starting body filled.
    [[nodiscard]] std::size_t maximum_score() const noexcept;

    // A text picture of the board, for an experiment to print one episode.
    [[nodiscard]] std::string render() const;

  private:
    [[nodiscard]] Heading turned(SnakeAction action) const noexcept;
    [[nodiscard]] Cell ahead(Heading heading) const noexcept;
    [[nodiscard]] bool blocked(Cell cell) const noexcept;
    void place_food();

    SnakeConfig config_;
    std::uint64_t state_;
    std::deque<Cell> body_;
    Cell food_{};
    Heading heading_{Heading::right};
    std::size_t score_{0};
    std::size_t steps_{0};
    std::size_t hungry_steps_{0};
    bool finished_{false};
};

} // namespace ml_scratch
