#include "ml_scratch/char_ngram.hpp"
// Link this test against ml_scratch_cpp alone: training and inference must be usable
// without any experiment utilities, checkpoint readers, or evaluation implementations.
#include "ml_scratch/convolution.hpp"
#include "ml_scratch/decision_tree.hpp"
#include "ml_scratch/gan.hpp"
#include "ml_scratch/kmeans.hpp"
#include "ml_scratch/linear_regression.hpp"
#include "ml_scratch/logistic_regression.hpp"
#include "ml_scratch/pca.hpp"
#include "ml_scratch/perceptron.hpp"
#include "ml_scratch/reinforcement.hpp"
#include "ml_scratch/rnn.hpp"
#include "ml_scratch/softmax_regression.hpp"
#include "ml_scratch/transformer.hpp"
#include "ml_scratch/xor_mlp.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition) {
    if (!condition)
        throw std::runtime_error("core algorithm or parameter access failed");
}

void test_parameter_access() {
    ml_scratch::LinearRegression model{2};
    model.set_parameters({2.0, -3.0}, 1.0);
    require(model.predict({0.5, -0.5}) == 3.5);
    for (int invalid = 0; invalid < 3; ++invalid) {
        bool rejected = false;
        try {
            if (invalid == 0)
                model.set_parameters({1.0}, 0.0);
            if (invalid == 1)
                model.set_parameters({1.0, 2.0}, std::numeric_limits<double>::infinity());
            if (invalid == 2)
                model.set_parameters({1.0, std::numeric_limits<double>::quiet_NaN()}, 0.0);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected);
        require(model.predict({0.5, -0.5}) == 3.5); // Rejection must not partly overwrite state.
    }
    auto result = model.fit({{{0.0, 0.0}, 1.0}}, {.max_epochs = 1, .shuffle = false});
    require(result.epochs == 1 && model.weights() == std::vector<double>({0.0, 0.0}));
}

void test_core_linkage() {
    using namespace ml_scratch;
    const LabeledDataset labeled{{{-1.0}, 0}, {{1.0}, 1}};
    BinaryPerceptron perceptron;
    LogicDataset logic{{{{0.0, 0.0}, 0}, {{0.0, 1.0}, 1}, {{1.0, 0.0}, 1}, {{1.0, 1.0}, 0}}};
    require(perceptron.fit(logic, 1).epochs == 1);
    XorMlp xor_model;
    require(xor_model.fit(logic, 1).epochs == 1);
    BinaryLogisticRegression logistic{1};
    require(logistic.fit({{{-1.0}, 0}, {{1.0}, 1}}, {.max_epochs = 1}).epochs == 1);
    SoftmaxRegression softmax{1, 2};
    require(softmax.fit(labeled, {.max_epochs = 1}).epochs == 1);
    FeedForwardNetwork network{{{1, 2, Activation::identity}}, Loss::softmax_cross_entropy};
    require(network.fit(labeled, {.max_epochs = 1}).epochs == 1);
    ConvolutionalNetwork cnn{
        {1, 1, 1}, {flatten(), dense(2, Activation::identity)}, Loss::softmax_cross_entropy};
    require(cnn.fit(labeled, {.max_epochs = 1}).epochs == 1);
    PrincipalComponentAnalysis pca{1};
    pca.fit({{-1.0}, {1.0}});
    require(pca.transform(std::vector<double>{1.0}).size() == 1);
    KMeans clusters{2};
    require(clusters.fit({{-1.0}, {1.0}}, {.restarts = 1}).converged);
    DecisionTreeClassifier tree{2};
    tree.fit(labeled);
    static_cast<void>(tree.prune(labeled));
    require(tree.predict({1.0}) == 1);
    CharNgram ngram{1, 2};
    ngram.fit({0, 1, 0});
    require(std::isfinite(ngram.cross_entropy({0, 1})));
    CharRnn rnn{2, 2};
    require(rnn.fit({0, 1, 0, 1}, {.sequence_length = 2, .batch_size = 1}).epochs == 1);
    CharTransformer transformer{
        {.vocabulary_size = 2, .context_length = 2, .model_size = 2, .heads = 1, .layers = 1}};
    require(transformer.fit({0, 1, 0, 1}, {.batch_size = 1}).epochs == 1);
    ConditionalGan gan{1, 2, {{3, 1, Activation::identity}}, {{3, 1, Activation::identity}}};
    require(gan.fit(labeled, {.batch_size = 2}).epochs == 1);
    SnakeGame game;
    TabularQLearning table;
    require(table.act(game.state_index()) < snake_action_count);
    DeepQLearning dqn{{.hidden = {2}, .batch_size = 1, .warmup = 1}};
    auto observation = game.observation();
    require(std::isfinite(dqn.observe(
        {observation, game.state_index(), 0, 1.0, observation, game.state_index(), false})));
}
} // namespace

int main() {
    try {
        test_parameter_access();
        test_core_linkage();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
