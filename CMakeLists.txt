cmake_minimum_required(VERSION 3.20)

project(
    ml_from_scratch_cpp_python
    VERSION 0.1.0
    DESCRIPTION "Educational machine-learning implementations from scratch"
    LANGUAGES CXX
)

add_library(
    ml_scratch_cpp STATIC
    src/cpp/src/cifar10.cpp
    src/cpp/src/convolution.cpp
    src/cpp/src/dataset.cpp
    src/cpp/src/decision_tree.cpp
    src/cpp/src/deterministic_random.cpp
    src/cpp/src/gan.cpp
    src/cpp/src/kmeans.cpp
    src/cpp/src/linear_regression.cpp
    src/cpp/src/logistic_regression.cpp
    src/cpp/src/logic_gates.cpp
    src/cpp/src/mnist.cpp
    src/cpp/src/neural_network.cpp
    src/cpp/src/optimizer.cpp
    src/cpp/src/pca.cpp
    src/cpp/src/perceptron.cpp
    src/cpp/src/project_info.cpp
    src/cpp/src/reinforcement.cpp
    src/cpp/src/rnn.cpp
    src/cpp/src/snake.cpp
    src/cpp/src/softmax_regression.cpp
    src/cpp/src/text_corpus.cpp
    src/cpp/src/transformer.cpp
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

add_executable(cpp_mlp_mnist experiments/09_mlp_mnist/cpp_main.cpp)
target_link_libraries(cpp_mlp_mnist PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_mlp_mnist)

add_executable(cpp_optimizers experiments/10_optimizers/cpp_main.cpp)
target_link_libraries(cpp_optimizers PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_optimizers)

add_executable(cpp_regularization experiments/11_regularization/cpp_main.cpp)
target_link_libraries(cpp_regularization PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_regularization)

add_executable(cpp_cnn_mnist experiments/12_cnn_mnist/cpp_main.cpp)
target_link_libraries(cpp_cnn_mnist PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_cnn_mnist)

add_executable(cpp_resnet_cifar10 experiments/13_residual/cpp_main.cpp)
target_link_libraries(cpp_resnet_cifar10 PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_resnet_cifar10)

add_executable(cpp_cgan_mnist experiments/14_cgan_mnist/cpp_main.cpp)
target_link_libraries(cpp_cgan_mnist PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_cgan_mnist)

add_executable(cpp_char_rnn experiments/15_char_rnn/cpp_main.cpp)
target_link_libraries(cpp_char_rnn PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_char_rnn)

add_executable(cpp_lstm_gru experiments/16_lstm_gru/cpp_main.cpp)
target_link_libraries(cpp_lstm_gru PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_lstm_gru)

add_executable(cpp_transformer experiments/17_transformer/cpp_main.cpp)
target_link_libraries(cpp_transformer PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_transformer)

add_executable(cpp_snake_rl experiments/18_snake_rl/cpp_main.cpp)
target_link_libraries(cpp_snake_rl PRIVATE ml_scratch_cpp)
enable_project_warnings(cpp_snake_rl)

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

    add_executable(cpp_residual_test tests/cpp/residual_test.cpp)
    target_link_libraries(cpp_residual_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_residual_test)
    add_test(NAME cpp_residual_test COMMAND cpp_residual_test)

    add_executable(cpp_convolution_test tests/cpp/convolution_test.cpp)
    target_link_libraries(cpp_convolution_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_convolution_test)
    add_test(NAME cpp_convolution_test COMMAND cpp_convolution_test)

    add_executable(cpp_regularization_test tests/cpp/regularization_test.cpp)
    target_link_libraries(cpp_regularization_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_regularization_test)
    add_test(NAME cpp_regularization_test COMMAND cpp_regularization_test)

    add_executable(cpp_optimizer_test tests/cpp/optimizer_test.cpp)
    target_link_libraries(cpp_optimizer_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_optimizer_test)
    add_test(NAME cpp_optimizer_test COMMAND cpp_optimizer_test)

    add_executable(cpp_softmax_regression_test tests/cpp/softmax_regression_test.cpp)
    target_link_libraries(cpp_softmax_regression_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_softmax_regression_test)
    add_test(NAME cpp_softmax_regression_test COMMAND cpp_softmax_regression_test)

    add_executable(cpp_gan_test tests/cpp/gan_test.cpp)
    target_link_libraries(cpp_gan_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_gan_test)
    add_test(NAME cpp_gan_test COMMAND cpp_gan_test)

    add_executable(cpp_text_corpus_test tests/cpp/text_corpus_test.cpp)
    target_link_libraries(cpp_text_corpus_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_text_corpus_test)
    add_test(NAME cpp_text_corpus_test COMMAND cpp_text_corpus_test)

    add_executable(cpp_rnn_test tests/cpp/rnn_test.cpp)
    target_link_libraries(cpp_rnn_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_rnn_test)
    add_test(NAME cpp_rnn_test COMMAND cpp_rnn_test)

    add_executable(cpp_transformer_test tests/cpp/transformer_test.cpp)
    target_link_libraries(cpp_transformer_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_transformer_test)
    add_test(NAME cpp_transformer_test COMMAND cpp_transformer_test)

    add_executable(cpp_snake_test tests/cpp/snake_test.cpp)
    target_link_libraries(cpp_snake_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_snake_test)
    add_test(NAME cpp_snake_test COMMAND cpp_snake_test)

    add_executable(cpp_reinforcement_test tests/cpp/reinforcement_test.cpp)
    target_link_libraries(cpp_reinforcement_test PRIVATE ml_scratch_cpp)
    enable_project_warnings(cpp_reinforcement_test)
    add_test(NAME cpp_reinforcement_test COMMAND cpp_reinforcement_test)
endif()
