cmake_minimum_required(VERSION 3.20)

project(
    ml_from_scratch_cpp_python
    VERSION 0.1.0
    DESCRIPTION "Educational machine-learning implementations from scratch"
    LANGUAGES CXX
)

add_library(
    ml_scratch_cpp STATIC
    src/cpp/src/dataset.cpp
    src/cpp/src/decision_tree.cpp
    src/cpp/src/deterministic_random.cpp
    src/cpp/src/kmeans.cpp
    src/cpp/src/linear_regression.cpp
    src/cpp/src/logistic_regression.cpp
    src/cpp/src/logic_gates.cpp
    src/cpp/src/mnist.cpp
    src/cpp/src/neural_network.cpp
    src/cpp/src/pca.cpp
    src/cpp/src/perceptron.cpp
    src/cpp/src/project_info.cpp
    src/cpp/src/softmax_regression.cpp
    src/cpp/src/xor_mlp.cpp
)
target_include_directories(
    ml_scratch_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include>
)
target_compile_features(ml_scratch_cpp PUBLIC cxx_std_20)

function(enable_project_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

enable_project_warnings(ml_scratch_cpp)

add_executable(cpp_perceptron_logic_gates experiments/01_perceptron_logic_gates/cpp_main.cpp)
target_link_libraries(cpp_perceptron_logic_gates PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_perceptron_logic_gates)

add_executable(cpp_xor_mlp experiments/02_xor_mlp/cpp_main.cpp)
target_link_libraries(cpp_xor_mlp PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_xor_mlp)

add_executable(cpp_linear_regression experiments/03_linear_regression/cpp_main.cpp)
target_link_libraries(cpp_linear_regression PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_linear_regression)

add_executable(cpp_logistic_regression experiments/04_logistic_regression/cpp_main.cpp)
target_link_libraries(cpp_logistic_regression PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_logistic_regression)

add_executable(cpp_pca_kmeans experiments/05_pca_kmeans/cpp_main.cpp)
target_link_libraries(cpp_pca_kmeans PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_pca_kmeans)

add_executable(cpp_decision_tree experiments/06_decision_tree/cpp_main.cpp)
target_link_libraries(cpp_decision_tree PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_decision_tree)

add_executable(cpp_backpropagation experiments/07_backpropagation/cpp_main.cpp)
target_link_libraries(cpp_backpropagation PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_backpropagation)

add_executable(cpp_softmax_mnist experiments/08_softmax_mnist/cpp_main.cpp)
target_link_libraries(cpp_softmax_mnist PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_softmax_mnist)

include(CTest)
if(BUILD_TESTING)
    add_executable(cpp_smoke_test tests/cpp/smoke_test.cpp)
    target_link_libraries(cpp_smoke_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_smoke_test)
    add_test(NAME cpp_smoke_test COMMAND cpp_smoke_test)

    add_executable(cpp_logic_models_test tests/cpp/logic_models_test.cpp)
    target_link_libraries(cpp_logic_models_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_logic_models_test)
    add_test(NAME cpp_logic_models_test COMMAND cpp_logic_models_test)

    add_executable(cpp_linear_regression_test tests/cpp/linear_regression_test.cpp)
    target_link_libraries(cpp_linear_regression_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_linear_regression_test)
    add_test(NAME cpp_linear_regression_test COMMAND cpp_linear_regression_test)

    add_executable(cpp_logistic_regression_test tests/cpp/logistic_regression_test.cpp)
    target_link_libraries(cpp_logistic_regression_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_logistic_regression_test)
    add_test(NAME cpp_logistic_regression_test COMMAND cpp_logistic_regression_test)

    add_executable(cpp_pca_test tests/cpp/pca_test.cpp)
    target_link_libraries(cpp_pca_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_pca_test)
    add_test(NAME cpp_pca_test COMMAND cpp_pca_test)

    add_executable(cpp_kmeans_test tests/cpp/kmeans_test.cpp)
    target_link_libraries(cpp_kmeans_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_kmeans_test)
    add_test(NAME cpp_kmeans_test COMMAND cpp_kmeans_test)

    add_executable(cpp_decision_tree_test tests/cpp/decision_tree_test.cpp)
    target_link_libraries(cpp_decision_tree_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_decision_tree_test)
    add_test(NAME cpp_decision_tree_test COMMAND cpp_decision_tree_test)

    add_executable(cpp_neural_network_test tests/cpp/neural_network_test.cpp)
    target_link_libraries(cpp_neural_network_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_neural_network_test)
    add_test(NAME cpp_neural_network_test COMMAND cpp_neural_network_test)

    add_executable(cpp_softmax_regression_test tests/cpp/softmax_regression_test.cpp)
    target_link_libraries(cpp_softmax_regression_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_softmax_regression_test)
    add_test(NAME cpp_softmax_regression_test COMMAND cpp_softmax_regression_test)
endif()
