# Backpropagation and gradient checking
target_sources(ml_scratch_cpp PRIVATE
    neural_network.cpp
)
target_include_directories(ml_scratch_cpp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(ml_scratch_utils PRIVATE
    utils/evaluation.cpp
    utils/gradient_check.cpp
    utils/neural_network_checkpoint.cpp
)

add_executable(cpp_backpropagation main.cpp)
target_link_libraries(cpp_backpropagation PRIVATE ml_scratch_utils)
enable_project_warnings(cpp_backpropagation)

if(BUILD_TESTING)
    add_ml_test(cpp_neural_network_test tests/neural_network_test.cpp ml_scratch_utils)
endif()
