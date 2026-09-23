#include "ml_scratch/snake.hpp"

#include <algorithm>
#include <stdexcept>

namespace ml_scratch {
namespace {

// splitmix64, the same stream DeterministicRandom uses: the food sequence has to be identical on
// every standard library so two agents can be compared on the same episodes.
std::uint64_t next_random(std::uint64_t& state) noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

Cell offset(const Heading heading) noexcept {
    switch (heading) {
    case Heading::up:
        return {0, -1};
    case Heading::right:
        return {1, 0};
    case Heading::down:
        return {0, 1};
    case Heading::left:
        return {-1, 0};
    }
    return {0, 0};
}

} // namespace

SnakeGame::SnakeGame(SnakeConfig config, const std::uint64_t seed)
    : config_(config), state_(seed) {
    if (config_.width < 4 || config_.height < 4) {
        throw std::invalid_argument("the board must be at least four cells on a side");
    }
    if (config_.steps_without_food == 0) {
        throw std::invalid_argument("steps_without_food must be positive");
    }
    reset();
}

void SnakeGame::reset(const std::uint64_t seed) {
    state_ = seed;
    reset();
}

void SnakeGame::reset() {
    const int centre_x = static_cast<int>(config_.width / 2);
    const int centre_y = static_cast<int>(config_.height / 2);
    // Head first, so the body reads from the head backwards and growing is a push at the front.
    body_.assign({Cell{centre_x, centre_y}, Cell{centre_x - 1, centre_y},
                  Cell{centre_x - 2, centre_y}});
    heading_ = Heading::right;
    score_ = 0;
    steps_ = 0;
    hungry_steps_ = 0;
    finished_ = false;
    place_food();
}

void SnakeGame::place_food() {
    // Chosen uniformly over the free cells rather than by rejection sampling, so the number of
    // draws does not depend on how full the board is and the stream stays comparable.
    const std::size_t cells = config_.width * config_.height;
    if (body_.size() >= cells) {
        food_ = body_.front();
        return;
    }
    const std::size_t free_cells = cells - body_.size();
    std::size_t target = static_cast<std::size_t>(next_random(state_) % free_cells);
    for (std::size_t index = 0; index < cells; ++index) {
        const Cell candidate{static_cast<int>(index % config_.width),
                             static_cast<int>(index / config_.width)};
        if (std::find(body_.begin(), body_.end(), candidate) != body_.end()) {
            continue;
        }
        if (target == 0) {
            food_ = candidate;
            return;
        }
        --target;
    }
}

Heading SnakeGame::turned(const SnakeAction action) const noexcept {
    const int current = static_cast<int>(heading_);
    switch (action) {
    case SnakeAction::turn_right:
        return static_cast<Heading>((current + 1) % 4);
    case SnakeAction::turn_left:
        return static_cast<Heading>((current + 3) % 4);
    case SnakeAction::straight:
        break;
    }
    return heading_;
}

Cell SnakeGame::ahead(const Heading heading) const noexcept {
    const Cell step = offset(heading);
    return {body_.front().x + step.x, body_.front().y + step.y};
}

bool SnakeGame::blocked(const Cell cell) const noexcept {
    if (cell.x < 0 || cell.y < 0 || cell.x >= static_cast<int>(config_.width) ||
        cell.y >= static_cast<int>(config_.height)) {
        return true;
    }
    // The tail cell is about to move out of the way, so it does not block — unless the snake is
    // about to grow into it, which only happens when that cell also holds the food.
    const bool growing = cell == food_;
    const auto last = growing ? body_.end() : std::prev(body_.end());
    return std::find(body_.begin(), last, cell) != last;
}

SnakeStepResult SnakeGame::step(const SnakeAction action) {
    if (finished_) {
        throw std::logic_error("the episode has finished; reset before stepping again");
    }
    ++steps_;
    heading_ = turned(action);
    const Cell target = ahead(heading_);
    SnakeStepResult result{config_.step_reward};

    if (blocked(target)) {
        finished_ = true;
        result.died = true;
        result.reward += config_.death_reward;
        return result;
    }

    body_.push_front(target);
    if (target == food_) {
        ++score_;
        hungry_steps_ = 0;
        result.ate = true;
        result.reward += config_.food_reward;
        place_food();
    } else {
        body_.pop_back();
        ++hungry_steps_;
        if (hungry_steps_ >= config_.steps_without_food * (1 + score_)) {
            // Out of patience rather than dead: the value of the final state is still whatever
            // the agent thinks it is, so no death penalty is applied.
            finished_ = true;
            result.truncated = true;
        }
    }
    return result;
}

std::size_t SnakeGame::maximum_score() const noexcept {
    return config_.width * config_.height - 3;
}

std::vector<double> SnakeGame::observation() const {
    std::vector<double> features(observation_size, 0.0);
    // Danger: would each of the three moves end the episode? This is what makes the observation
    // enough to stay alive without knowing where the body is.
    features[0] = blocked(ahead(turned(SnakeAction::straight))) ? 1.0 : 0.0;
    features[1] = blocked(ahead(turned(SnakeAction::turn_right))) ? 1.0 : 0.0;
    features[2] = blocked(ahead(turned(SnakeAction::turn_left))) ? 1.0 : 0.0;
    features[3 + static_cast<std::size_t>(heading_)] = 1.0;
    const Cell head = body_.front();
    features[7] = food_.y < head.y ? 1.0 : 0.0;
    features[8] = food_.x > head.x ? 1.0 : 0.0;
    features[9] = food_.y > head.y ? 1.0 : 0.0;
    features[10] = food_.x < head.x ? 1.0 : 0.0;
    return features;
}

std::size_t SnakeGame::state_index() const {
    const std::vector<double> features = observation();
    std::size_t index = 0;
    for (std::size_t bit = 0; bit < observation_size; ++bit) {
        index = (index << 1) | (features[bit] > 0.5 ? 1U : 0U);
    }
    return index;
}

std::string SnakeGame::render() const {
    std::string picture;
    picture.reserve((config_.width + 3) * (config_.height + 2));
    picture.append(config_.width + 2, '#');
    picture.push_back('\n');
    for (std::size_t y = 0; y < config_.height; ++y) {
        picture.push_back('#');
        for (std::size_t x = 0; x < config_.width; ++x) {
            const Cell cell{static_cast<int>(x), static_cast<int>(y)};
            if (cell == body_.front()) {
                picture.push_back('@');
            } else if (std::find(body_.begin(), body_.end(), cell) != body_.end()) {
                picture.push_back('o');
            } else if (cell == food_) {
                picture.push_back('*');
            } else {
                picture.push_back(' ');
            }
        }
        picture.append("#\n");
    }
    picture.append(config_.width + 2, '#');
    picture.push_back('\n');
    return picture;
}

} // namespace ml_scratch
