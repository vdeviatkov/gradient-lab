#include "ml_scratch/project_info.hpp"

#include <iostream>
#include <string_view>

int main() {
    constexpr std::string_view expected_name{"ml-from-scratch-cpp-python"};
    constexpr std::string_view expected_version{"0.1.0"};

    if (ml_scratch::project_name() != expected_name) {
        std::cerr << "unexpected project name\n";
        return 1;
    }
    if (ml_scratch::project_version() != expected_version) {
        std::cerr << "unexpected project version\n";
        return 1;
    }
    return 0;
}
