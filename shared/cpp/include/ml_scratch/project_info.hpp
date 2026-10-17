#pragma once

#include <string_view>

namespace ml_scratch {

[[nodiscard]] std::string_view project_name() noexcept;
[[nodiscard]] std::string_view project_version() noexcept;

} // namespace ml_scratch
