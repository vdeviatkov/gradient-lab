cmake_minimum_required(VERSION 3.20)

project(
    ml_from_scratch_cpp_python
    VERSION 0.1.0
    DESCRIPTION "Educational machine-learning implementations from scratch"
    LANGUAGES CXX
)

add_library(ml_scratch_cpp STATIC src/cpp/src/project_info.cpp)
target_include_directories(
    ml_scratch_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include>
)
target_compile_features(ml_scratch_cpp PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(ml_scratch_cpp PRIVATE /W4 /permissive-)
else()
    target_compile_options(ml_scratch_cpp PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(cpp_smoke_test tests/cpp/smoke_test.cpp)
    target_link_libraries(cpp_smoke_test PRIVATE ml_scratch_cpp)
    add_test(NAME cpp_smoke_test COMMAND cpp_smoke_test)
endif()
