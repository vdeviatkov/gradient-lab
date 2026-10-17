#include "ml_scratch/reinforcement.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

namespace ml_scratch {

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
