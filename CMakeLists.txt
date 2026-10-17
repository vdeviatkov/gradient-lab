cmake_minimum_required(VERSION 3.20)
project(ml_from_scratch_cpp_python VERSION 0.1.0
        DESCRIPTION "Educational machine-learning implementations from scratch" LANGUAGES CXX)

# Keep the documented build/cpp_* executable paths stable across topic subdirectories.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
include(CTest)

function(enable_project_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(add_ml_test name source library)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${library})
    enable_project_warnings(${name})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Shared infrastructure is independent of any one learning milestone.
add_library(ml_scratch_cpp STATIC
    shared/cpp/src/dataset.cpp
    shared/cpp/src/gradient.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shared/cpp/include>")
target_compile_features(ml_scratch_cpp PUBLIC cxx_std_20)
enable_project_warnings(ml_scratch_cpp)

# Support may depend on models; model training/inference must not depend on support.
add_library(ml_scratch_utils STATIC
    shared/cpp/utils/cifar10.cpp
    shared/cpp/utils/classification_metrics.cpp
    shared/cpp/utils/dataset.cpp
    shared/cpp/utils/deterministic_random.cpp
    shared/cpp/utils/mnist.cpp
    shared/cpp/utils/project_info.cpp
    shared/cpp/utils/text_corpus.cpp
)
target_link_libraries(ml_scratch_utils PUBLIC ml_scratch_cpp)
enable_project_warnings(ml_scratch_utils)

add_subdirectory(topics)

if(BUILD_TESTING)
    add_ml_test(cpp_numerics_test tests/cpp/numerics_test.cpp ml_scratch_cpp)
    add_ml_test(cpp_core_test tests/cpp/core_test.cpp ml_scratch_cpp)
    add_ml_test(cpp_utilities_test tests/cpp/utilities_test.cpp ml_scratch_utils)
    add_ml_test(cpp_smoke_test tests/cpp/smoke_test.cpp ml_scratch_utils)
    add_ml_test(cpp_text_corpus_test tests/cpp/text_corpus_test.cpp ml_scratch_utils)
endif()
