# Implementation benchmarks
target_sources(ml_scratch_utils PRIVATE
    utils/benchmark.cpp
)

add_executable(cpp_benchmarks main.cpp)
target_link_libraries(cpp_benchmarks PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_benchmarks)

if(BUILD_TESTING)
    add_ml_test(cpp_benchmark_test tests/benchmark_test.cpp ml_scratch_utils)
endif()
