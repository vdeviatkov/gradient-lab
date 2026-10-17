# Binary logistic regression
target_sources(ml_scratch_cpp PRIVATE
    logistic_regression.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/evaluation.cpp
)

add_executable(cpp_logistic_regression main.cpp)
target_link_libraries(cpp_logistic_regression PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_logistic_regression)

if(BUILD_TESTING)
    add_ml_test(cpp_logistic_regression_test tests/logistic_regression_test.cpp ml_scratch_utils)
endif()
