# Topic order is the canonical learning progression; models may reuse earlier topics.
foreach(topic IN ITEMS
    01_perceptron_logic_gates
    02_xor_mlp
    03_linear_regression
    04_logistic_regression
    05_pca_kmeans
    06_decision_tree
    07_backpropagation
    08_softmax_mnist
    09_mlp_mnist
    10_optimizers
    11_regularization
    12_cnn_mnist
    13_residual
    14_cgan_mnist
    15_char_rnn
    16_lstm_gru
    17_transformer
    18_snake_rl
    19_benchmarks
)
    add_subdirectory(${topic}/cpp)
endforeach()
