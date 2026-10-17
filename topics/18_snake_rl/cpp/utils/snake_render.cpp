#include "ml_scratch/snake.hpp"
#include <string>

namespace ml_scratch {

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
