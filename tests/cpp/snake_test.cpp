#include "ml_scratch/snake.hpp"

#include <algorithm>
#include <cmath>
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

void test_initial_state() {
    const ml_scratch::SnakeGame game{{10, 10}, 1};
    require(game.body().size() == 3, "the snake starts three cells long");
    require(game.head() == ml_scratch::Cell{5, 5}, "the head starts at the centre");
    require(game.body().back() == ml_scratch::Cell{3, 5}, "the body trails to the left");
    require(game.heading() == ml_scratch::Heading::right, "the snake starts heading right");
    require(game.score() == 0 && game.steps() == 0 && !game.finished(), "a fresh episode");
    require(std::find(game.body().begin(), game.body().end(), game.food()) == game.body().end(),
            "the food never starts under the snake");
    require(game.maximum_score() == 97, "the best possible score fills every free cell");
    require_invalid_argument([] { ml_scratch::SnakeGame bad{{3, 10}}; },
                             "a board narrower than four cells is rejected");
    require_invalid_argument([] { ml_scratch::SnakeGame bad{{10, 10, 1.0, -1.0, 0.0, 0}}; },
                             "a zero patience is rejected");
}

void test_movement_and_turning() {
    ml_scratch::SnakeGame game{{10, 10}, 1};
    const auto result = game.step(ml_scratch::SnakeAction::straight);
    require(!result.finished() && game.head() == ml_scratch::Cell{6, 5},
            "straight moves one cell along the heading");
    require(game.body().size() == 3, "the tail follows, so the length is unchanged");
    require(game.steps() == 1, "the step is counted");

    game.step(ml_scratch::SnakeAction::turn_right);
    require(game.heading() == ml_scratch::Heading::down && game.head() == ml_scratch::Cell{6, 6},
            "turning right from east heads south");
    game.step(ml_scratch::SnakeAction::turn_left);
    require(game.heading() == ml_scratch::Heading::right, "turning left undoes it");
    game.step(ml_scratch::SnakeAction::turn_left);
    game.step(ml_scratch::SnakeAction::turn_left);
    require(game.heading() == ml_scratch::Heading::left,
            "two left turns reverse the heading, which takes two steps rather than one");
}

void test_death_and_truncation() {
    // Drive the snake into the right wall: from the centre of a ten-wide board it takes four
    // straight steps to reach x = 9 and a fifth to leave the board.
    ml_scratch::SnakeGame game{{10, 10}, 1};
    ml_scratch::SnakeStepResult result{};
    for (int index = 0; index < 5; ++index) {
        require(!game.finished(), "the snake is still alive before the wall");
        result = game.step(ml_scratch::SnakeAction::straight);
    }
    require(result.died && !result.truncated && game.finished(), "the wall ends the episode");
    require(result.reward == -1.0, "death pays the death reward");
    bool threw = false;
    try {
        game.step(ml_scratch::SnakeAction::straight);
    } catch (const std::logic_error&) {
        threw = true;
    }
    require(threw, "stepping a finished episode is a logic error");

    // Circling forever truncates rather than dying, and truncation pays no death penalty.
    ml_scratch::SnakeConfig patient{10, 10, 1.0, -1.0, 0.0, 12};
    ml_scratch::SnakeGame circler{patient, 7};
    ml_scratch::SnakeStepResult last{};
    for (int index = 0; index < 200 && !circler.finished(); ++index) {
        // A four-step square, which cannot hit anything on an empty board's interior.
        last = circler.step(index % 4 == 0 ? ml_scratch::SnakeAction::turn_right
                                           : ml_scratch::SnakeAction::straight);
    }
    require(circler.finished(), "the step limit ends an endless episode");
    if (last.truncated) {
        require(!last.died && last.reward == 0.0,
                "truncation is not death and pays no penalty");
    }
}

void test_eating_grows_the_snake() {
    // Put the food directly ahead by searching seeds for one that places it on the snake's row
    // to the right of the head.
    ml_scratch::SnakeConfig config{10, 10};
    for (std::uint64_t seed = 0; seed < 200; ++seed) {
        ml_scratch::SnakeGame game{config, seed};
        const ml_scratch::Cell food = game.food();
        if (food.y != 5 || food.x <= 5) {
            continue;
        }
        const std::size_t length = game.body().size();
        ml_scratch::SnakeStepResult result{};
        for (int step = 5; step < food.x; ++step) {
            result = game.step(ml_scratch::SnakeAction::straight);
        }
        require(result.ate && result.reward == 1.0, "reaching the food pays the food reward");
        require(game.score() == 1, "the score counts the food");
        require(game.body().size() == length + 1, "eating grows the snake by one cell");
        require(game.food() != food, "new food appears somewhere else");
        require(std::find(game.body().begin(), game.body().end(), game.food()) ==
                    game.body().end(),
                "new food never appears under the snake");
        return;
    }
    require(false, "no seed placed the food ahead of the snake");
}

void test_observation() {
    ml_scratch::SnakeGame game{{10, 10}, 1};
    const auto features = game.observation();
    require(features.size() == ml_scratch::SnakeGame::observation_size, "eleven features");
    for (const double value : features) {
        require(value == 0.0 || value == 1.0, "every feature is a bit");
    }
    require(features[0] == 0.0 && features[1] == 0.0 && features[2] == 0.0,
            "nothing is dangerous in the middle of an empty board");
    require(features[3] == 0.0 && features[4] == 1.0, "the heading is one-hot at east");
    const ml_scratch::Cell food = game.food();
    require(features[7] == (food.y < 5 ? 1.0 : 0.0) && features[8] == (food.x > 5 ? 1.0 : 0.0) &&
                features[9] == (food.y > 5 ? 1.0 : 0.0) && features[10] == (food.x < 5 ? 1.0 : 0.0),
            "the food flags point at the food");

    // Against the right wall the danger flags fire for exactly the moves that would end it.
    ml_scratch::SnakeGame edge{{10, 10}, 1};
    for (int index = 0; index < 4; ++index) {
        edge.step(ml_scratch::SnakeAction::straight);
    }
    require(edge.head().x == 9, "the snake reached the wall");
    const auto danger = edge.observation();
    require(danger[0] == 1.0, "straight ahead is the wall");
    require(danger[1] == 0.0 && danger[2] == 0.0, "turning either way is still open");

    // The index is the eleven bits packed in order.
    const std::size_t index = edge.state_index();
    require(index < ml_scratch::SnakeGame::state_count, "the index fits the table");
    std::size_t expected = 0;
    for (const double value : danger) {
        expected = (expected << 1) | (value > 0.5 ? 1U : 0U);
    }
    require(index == expected, "state_index packs the observation");
}

void test_determinism() {
    // The same seed gives the same food sequence, so two agents can be compared on identical
    // episodes; a different seed gives a different one.
    const auto play = [](const std::uint64_t seed) {
        ml_scratch::SnakeGame game{{10, 10}, seed};
        std::vector<ml_scratch::Cell> food{game.food()};
        for (int index = 0; index < 40 && !game.finished(); ++index) {
            game.step(index % 3 == 0 ? ml_scratch::SnakeAction::turn_left
                                     : ml_scratch::SnakeAction::straight);
            food.push_back(game.food());
        }
        return food;
    };
    require(play(11) == play(11), "the same seed replays exactly");
    require(play(11) != play(12), "a different seed gives different food");

    ml_scratch::SnakeGame game{{10, 10}, 5};
    const ml_scratch::Cell first = game.food();
    game.step(ml_scratch::SnakeAction::straight);
    game.reset(5);
    require(game.food() == first && game.steps() == 0 && game.body().size() == 3,
            "reset with a seed restarts the same episode");
}

void test_render() {
    const ml_scratch::SnakeGame game{{6, 4}, 3};
    const std::string picture = game.render();
    require(std::count(picture.begin(), picture.end(), '\n') == 6, "a row per line plus borders");
    require(picture.find('@') != std::string::npos && picture.find('*') != std::string::npos,
            "the head and the food are drawn");
    require(std::count(picture.begin(), picture.end(), 'o') == 2, "the two body cells are drawn");
}

} // namespace

int main() {
    try {
        test_initial_state();
        test_movement_and_turning();
        test_death_and_truncation();
        test_eating_grows_the_snake();
        test_observation();
        test_determinism();
        test_render();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
