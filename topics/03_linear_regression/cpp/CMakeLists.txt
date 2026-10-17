# Linear regression
target_sources(ml_scratch_cpp PRIVATE
    linear_regression.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/regression.cpp
)

add_executable(cpp_linear_regression main.cpp)
target_link_libraries(cpp_linear_regression PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_linear_regression)

if(BUILD_TESTING)
    add_ml_test(cpp_linear_regression_test tests/linear_regression_test.cpp ml_scratch_utils)
endif()
