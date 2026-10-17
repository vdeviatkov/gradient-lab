#include "ml_scratch/kmeans.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(const bool condition, const std::string_view message) {
    if (!condition) {
        throw std::runtime_error{std::string{message}};
    }
}

void require_near(const double actual, const double expected, const double tolerance,
                  const std::string_view message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

template <typename Function>
void require_invalid_argument(Function function, const std::string_view message) {
    bool rejected = false;
    try {
        function();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

// Two compact groups whose exact means are (0, 0) and (10, 10).
const ml_scratch::FeatureMatrix two_blobs{
    {-1.0, 0.0}, {1.0, 0.0},   {0.0, -1.0}, {0.0, 1.0},
    {9.0, 10.0}, {11.0, 10.0}, {10.0, 9.0}, {10.0, 11.0},
};

const std::vector<std::size_t> two_blob_labels{0, 0, 0, 0, 1, 1, 1, 1};

void test_recovers_separated_blobs() {
    ml_scratch::KMeansConfig config;
    config.seed = 7;
    config.restarts = 5;

    ml_scratch::KMeans model{2};
    const auto result = model.fit(two_blobs, config);

    require(result.converged, "k-means did not converge on separated blobs");
    require(result.restarts.size() == config.restarts, "wrong restart summary count");
    // Each blob contributes four points at squared distance one from its mean.
    require_near(result.inertia, 8.0, 1e-9, "incorrect inertia at the known optimum");
    require_near(model.inertia(two_blobs), result.inertia, 1e-12,
                 "reported inertia disagrees with the fitted model");
    require_near(ml_scratch::cluster_purity(model.labels(), two_blob_labels), 1.0, 1e-12,
                 "well-separated blobs were not recovered");

    const std::size_t first = model.predict({0.0, 0.0});
    const std::size_t second = model.predict({10.0, 10.0});
    require(first != second, "both blob means fell into one cluster");
    for (const double coordinate : model.centroids()[first]) {
        require_near(coordinate, 0.0, 1e-9, "incorrect centroid for the first blob");
    }
    for (const double coordinate : model.centroids()[second]) {
        require_near(coordinate, 10.0, 1e-9, "incorrect centroid for the second blob");
    }
}

void test_single_cluster_is_the_mean() {
    ml_scratch::KMeans model{1};
    const auto result = model.fit(two_blobs);

    for (const double coordinate : model.centroids()[0]) {
        require_near(coordinate, 5.0, 1e-12, "a single cluster is not the dataset mean");
    }
    // Total scatter about the mean: 8 points, each 50 + 1 away in squared distance.
    require_near(result.inertia, 8.0 * 51.0, 1e-9, "incorrect single-cluster inertia");
}

void test_more_clusters_reduce_inertia() {
    ml_scratch::KMeansConfig config;
    config.seed = 11;
    config.restarts = 8;

    std::vector<double> inertia_by_cluster_count;
    double previous = std::numeric_limits<double>::infinity();
    for (std::size_t clusters = 1; clusters <= 8; ++clusters) {
        ml_scratch::KMeans model{clusters};
        const double inertia = model.fit(two_blobs, config).inertia;
        require(inertia <= previous + 1e-9, "adding a cluster increased inertia");
        inertia_by_cluster_count.push_back(inertia);
        previous = inertia;
    }
    require(inertia_by_cluster_count[3] < inertia_by_cluster_count[1],
            "four clusters did not improve on two");
    // One centroid per sample drives inertia to zero, which is why inertia alone cannot choose k.
    require_near(inertia_by_cluster_count.back(), 0.0, 1e-9,
                 "one cluster per sample left non-zero inertia");
}

void test_restarts_and_initialization() {
    ml_scratch::KMeansConfig config;
    config.seed = 3;
    config.restarts = 6;

    ml_scratch::KMeans plus_plus_model{2};
    config.initialization = ml_scratch::KMeansInitialization::kmeans_plus_plus;
    const auto plus_plus = plus_plus_model.fit(two_blobs, config);

    ml_scratch::KMeans random_model{2};
    config.initialization = ml_scratch::KMeansInitialization::random_samples;
    const auto random = random_model.fit(two_blobs, config);

    // Both initializations solve this easy problem; the recorded seeds must differ per restart.
    require_near(plus_plus.inertia, random.inertia, 1e-9,
                 "initializations disagreed on a well-separated dataset");
    for (std::size_t restart = 0; restart < plus_plus.restarts.size(); ++restart) {
        require(plus_plus.restarts[restart].seed == config.seed + restart,
                "restart seeds are not derived from the configured seed");
        require(plus_plus.restarts[restart].inertia >= plus_plus.inertia,
                "a restart beat the reported best inertia");
    }
    require(plus_plus.restarts[plus_plus.best_restart].inertia == plus_plus.inertia,
            "best_restart does not identify the retained restart");
}

void test_seed_reproducibility() {
    ml_scratch::KMeansConfig config;
    config.seed = 2027;
    config.restarts = 4;
    config.max_iterations = 3;

    ml_scratch::KMeans first{3};
    ml_scratch::KMeans second{3};
    const auto first_result = first.fit(two_blobs, config);
    const auto second_result = second.fit(two_blobs, config);

    require(first_result == second_result, "identical seeds produced different results");
    require(first.centroids() == second.centroids(),
            "identical seeds produced different centroids");
    require(first.labels() == second.labels(), "identical seeds produced different labels");
}

void test_duplicate_points_and_empty_clusters() {
    // Only two distinct locations, so a third cluster cannot hold a distinct group.
    const ml_scratch::FeatureMatrix duplicates{
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {5.0, 5.0}, {5.0, 5.0}, {5.0, 5.0},
    };
    ml_scratch::KMeansConfig config;
    config.seed = 99;
    config.restarts = 3;

    ml_scratch::KMeans model{3};
    const auto result = model.fit(duplicates, config);

    require_near(result.inertia, 0.0, 1e-12, "duplicate points left non-zero inertia");
    require(model.centroids().size() == 3, "an empty cluster removed a centroid");
    for (const auto& centroid : model.centroids()) {
        for (const double coordinate : centroid) {
            require(std::isfinite(coordinate), "an empty cluster produced a non-finite centroid");
        }
    }
}

void test_cluster_purity() {
    require_near(ml_scratch::cluster_purity({0, 0, 1, 1}, {0, 0, 1, 1}), 1.0, 1e-12,
                 "incorrect perfect purity");
    require_near(ml_scratch::cluster_purity({0, 0, 0, 0}, {0, 0, 1, 1}), 0.5, 1e-12,
                 "incorrect single-cluster purity");
    require_near(ml_scratch::cluster_purity({0, 0, 0, 1}, {0, 0, 1, 1}), 0.75, 1e-12,
                 "incorrect mixed purity");
    require_invalid_argument([] { static_cast<void>(ml_scratch::cluster_purity({}, {})); },
                             "empty purity input was accepted");
    require_invalid_argument([] { static_cast<void>(ml_scratch::cluster_purity({0}, {0, 1})); },
                             "mismatched purity input was accepted");
}

void test_input_validation() {
    require_invalid_argument([] { ml_scratch::KMeans{0}; }, "zero cluster count was accepted");

    ml_scratch::KMeans model{3};
    require_invalid_argument([&] { model.fit({{1.0}, {2.0}}); },
                             "more clusters than samples was accepted");
    require_invalid_argument([&] { model.fit({}); }, "an empty dataset was accepted");
    require_invalid_argument([&] { model.fit({{1.0}, {2.0, 3.0}, {4.0}}); },
                             "a ragged dataset was accepted");

    ml_scratch::KMeansConfig config;
    config.max_iterations = 0;
    require_invalid_argument([&] { model.fit(two_blobs, config); },
                             "a zero iteration budget was accepted");
    config.max_iterations = 10;
    config.restarts = 0;
    require_invalid_argument([&] { model.fit(two_blobs, config); },
                             "a zero restart count was accepted");

    bool rejected = false;
    try {
        static_cast<void>(model.predict(std::vector<double>{1.0, 2.0}));
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "an unfitted model produced a prediction");

    ml_scratch::KMeans fitted{2};
    fitted.fit(two_blobs);
    require_invalid_argument([&] { static_cast<void>(fitted.predict(std::vector<double>{1.0})); },
                             "a wrong feature count was accepted");
    require_invalid_argument(
        [&] {
            static_cast<void>(
                fitted.predict(std::vector<double>{std::numeric_limits<double>::infinity(), 0.0}));
        },
        "a non-finite feature was accepted");
}

} // namespace

int main() {
    try {
        test_recovers_separated_blobs();
        test_single_cluster_is_the_mean();
        test_more_clusters_reduce_inertia();
        test_restarts_and_initialization();
        test_seed_reproducibility();
        test_duplicate_points_and_empty_clusters();
        test_cluster_purity();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
