# Convolutional networks
target_sources(ml_scratch_cpp PRIVATE
    convolution.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/convolution_checkpoint.cpp
    utils/evaluation.cpp
)

add_executable(cpp_cnn_mnist main.cpp)
target_link_libraries(cpp_cnn_mnist PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_cnn_mnist)

if(BUILD_TESTING)
    add_ml_test(cpp_convolution_test tests/convolution_test.cpp ml_scratch_utils)
endif()
