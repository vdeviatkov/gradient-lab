# Softmax regression on MNIST
target_sources(ml_scratch_cpp PRIVATE
    softmax_regression.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/evaluation.cpp
)

add_executable(cpp_softmax_mnist main.cpp)
target_link_libraries(cpp_softmax_mnist PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_softmax_mnist)

if(BUILD_TESTING)
    add_ml_test(cpp_softmax_regression_test tests/softmax_regression_test.cpp ml_scratch_utils)
endif()
