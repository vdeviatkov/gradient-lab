#include "ml_scratch/kmeans.hpp"
#include "ml_scratch/pca.hpp"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t seed = 20260907;
constexpr std::size_t true_group_count = 3;
constexpr std::size_t samples_per_group = 40;
constexpr std::size_t train_per_group = 24;
constexpr std::size_t validation_per_group = 8;
constexpr std::size_t max_candidate_clusters = 6;
constexpr std::size_t kmeans_restarts = 10;

// A self-contained splitmix64 stream plus Box-Muller. The standard distributions are not specified
// bit-for-bit, so generating the dataset explicitly keeps it identical on every standard library.
class DeterministicGaussian {
  public:
    explicit DeterministicGaussian(const std::uint64_t initial_state) : state_(initial_state) {}

    double operator()() {
        // Box-Muller consumes two uniforms and returns one of the two normal deviates.
        const double first = std::max(next_uniform(), 1e-12);
        const double second = next_uniform();
        return std::sqrt(-2.0 * std::log(first)) * std::cos(2.0 * 3.14159265358979323846 * second);
    }

  private:
    double next_uniform() {
        state_ += 0x9E3779B97F4A7C15ULL;
        std::uint64_t value = state_;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        value ^= value >> 31;
        return static_cast<double>(value >> 11) / 9007199254740992.0; // 2^53
    }

    std::uint64_t state_;
};

struct LabeledSplit {
    ml_scratch::FeatureMatrix features;
    std::vector<std::size_t> labels;
};

struct Dataset {
    LabeledSplit train;
    LabeledSplit validation;
    LabeledSplit test;
};

// Three groups live in a two-dimensional latent plane. The four observed features are a fixed
// linear map of that plane plus small isotropic noise, so the data is intrinsically
// two-dimensional and PCA has something real to recover.
Dataset build_dataset() {
    constexpr double latent_centers[true_group_count][2]{{0.0, 0.0}, {6.0, 0.5}, {3.0, 5.0}};
    constexpr double group_spread = 0.6;
    constexpr double observation_noise = 0.05;

    DeterministicGaussian noise{seed};
    Dataset dataset;
    for (std::size_t group = 0; group < true_group_count; ++group) {
        for (std::size_t index = 0; index < samples_per_group; ++index) {
            const double x = latent_centers[group][0] + group_spread * noise();
            const double y = latent_centers[group][1] + group_spread * noise();
            const std::vector<double> sample{
                x + observation_noise * noise(),
                y + observation_noise * noise(),
                0.8 * x - 0.6 * y + observation_noise * noise(),
                0.3 * x + 0.4 * y + observation_noise * noise(),
            };

            LabeledSplit& split = index < train_per_group ? dataset.train
                                  : index < train_per_group + validation_per_group
                                      ? dataset.validation
                                      : dataset.test;
            split.features.push_back(sample);
            split.labels.push_back(group);
        }
    }
    return dataset;
}

// Elbow heuristic: choose the interior k whose inertia curve bends the most. This uses no labels,
// so cluster count is selected without the information the purity evaluation later reveals.
std::size_t select_cluster_count(const std::vector<double>& inertia_by_cluster_count) {
    std::size_t best = 2;
    double best_curvature = -1.0;
    for (std::size_t clusters = 2; clusters + 1 <= inertia_by_cluster_count.size(); ++clusters) {
        const double curvature = inertia_by_cluster_count[clusters - 2] -
                                 2.0 * inertia_by_cluster_count[clusters - 1] +
                                 inertia_by_cluster_count[clusters];
        if (curvature > best_curvature) {
            best_curvature = curvature;
            best = clusters;
        }
    }
    return best;
}

struct ClusteringReport {
    double train_inertia;
    double test_inertia;
    double test_purity;
    std::size_t iterations;
    std::size_t best_restart;
    bool converged;
};

ClusteringReport run_kmeans(const ml_scratch::FeatureMatrix& train, const LabeledSplit& test_split,
                            const std::size_t cluster_count,
                            const ml_scratch::KMeansInitialization initialization) {
    ml_scratch::KMeansConfig config;
    config.initialization = initialization;
    config.restarts = kmeans_restarts;
    config.seed = static_cast<std::uint32_t>(seed);

    ml_scratch::KMeans model{cluster_count};
    const auto result = model.fit(train, config);
    return {result.inertia,
            model.inertia(test_split.features),
            ml_scratch::cluster_purity(model.predict(test_split.features), test_split.labels),
            result.iterations,
            result.best_restart,
            result.converged};
}

void print_clustering(const std::string_view name, const ClusteringReport& report) {
    std::cout << name << ": train inertia=" << report.train_inertia
              << ", test inertia=" << report.test_inertia << ", test purity=" << report.test_purity
              << ", iterations=" << report.iterations << ", best restart=" << report.best_restart
              << ", converged=" << (report.converged ? "yes" : "no") << '\n';
}

} // namespace

int main() {
    const Dataset dataset = build_dataset();
    const std::size_t feature_count = dataset.train.features.front().size();

    std::cout << std::fixed << std::setprecision(6)
              << "C++ PCA and k-means experiment (seed=" << seed << ")\n"
              << "dataset: " << true_group_count * samples_per_group << " samples, "
              << feature_count << " features, " << true_group_count
              << " true groups; splits train=" << dataset.train.features.size()
              << ", validation=" << dataset.validation.features.size()
              << ", test=" << dataset.test.features.size() << "\n\n";

    ml_scratch::PrincipalComponentAnalysis spectrum{feature_count};
    spectrum.fit(dataset.train.features);
    std::cout << "PCA spectrum (fitted on the training split)\n";
    double cumulative = 0.0;
    for (std::size_t component = 0; component < feature_count; ++component) {
        cumulative += spectrum.explained_variance_ratio()[component];
        std::cout << "  component " << component + 1
                  << ": explained variance=" << spectrum.explained_variance()[component]
                  << ", ratio=" << spectrum.explained_variance_ratio()[component]
                  << ", cumulative=" << cumulative << '\n';
    }
    std::cout << "  total variance=" << spectrum.total_variance() << '\n';

    std::cout << "held-out reconstruction error per feature (test split)\n";
    ml_scratch::FeatureMatrix projected_train;
    ml_scratch::FeatureMatrix projected_test;
    double two_component_ratio = 0.0;
    for (std::size_t components = 1; components <= feature_count; ++components) {
        ml_scratch::PrincipalComponentAnalysis model{components};
        model.fit(dataset.train.features);
        std::cout << "  rank " << components << ": "
                  << model.reconstruction_error(dataset.test.features)
                  << " (cumulative ratio=" << model.cumulative_explained_variance_ratio() << ")\n";
        if (components == 2) {
            two_component_ratio = model.cumulative_explained_variance_ratio();
            projected_train = model.transform(dataset.train.features);
            projected_test = model.transform(dataset.test.features);
        }
    }

    std::cout << "\nk-means cluster-count search on the validation split (k-means++, "
              << kmeans_restarts << " restarts)\n";
    std::vector<double> validation_inertia;
    for (std::size_t clusters = 1; clusters <= max_candidate_clusters; ++clusters) {
        ml_scratch::KMeansConfig config;
        config.restarts = kmeans_restarts;
        config.seed = static_cast<std::uint32_t>(seed);
        ml_scratch::KMeans model{clusters};
        model.fit(dataset.train.features, config);
        validation_inertia.push_back(model.inertia(dataset.validation.features));
        std::cout << "  k=" << clusters << ": validation inertia=" << validation_inertia.back()
                  << '\n';
    }
    const std::size_t selected_clusters = select_cluster_count(validation_inertia);
    std::cout << "  selected k=" << selected_clusters << " by the elbow heuristic\n\n";

    const LabeledSplit projected_test_split{projected_test, dataset.test.labels};
    const auto plus_plus = run_kmeans(dataset.train.features, dataset.test, selected_clusters,
                                      ml_scratch::KMeansInitialization::kmeans_plus_plus);
    const auto random_init = run_kmeans(dataset.train.features, dataset.test, selected_clusters,
                                        ml_scratch::KMeansInitialization::random_samples);
    const auto on_projection = run_kmeans(projected_train, projected_test_split, selected_clusters,
                                          ml_scratch::KMeansInitialization::kmeans_plus_plus);
    const auto baseline = run_kmeans(dataset.train.features, dataset.test, 1,
                                     ml_scratch::KMeansInitialization::kmeans_plus_plus);

    std::cout << "clustering on the test split (fitted on train, evaluated once)\n";
    print_clustering("  single-cluster baseline (k=1)", baseline);
    print_clustering("  k-means++ on raw features    ", plus_plus);
    print_clustering("  random init on raw features  ", random_init);
    print_clustering("  k-means++ on 2-D projection  ", on_projection);

    const bool intrinsic_dimension_found = two_component_ratio >= 0.99;
    const bool selected_the_true_group_count = selected_clusters == true_group_count;
    const bool clustering_beat_the_baseline = plus_plus.test_inertia < baseline.test_inertia &&
                                              plus_plus.test_purity > baseline.test_purity;
    const bool projection_kept_the_structure = on_projection.test_purity >= plus_plus.test_purity;

    if (!intrinsic_dimension_found || !selected_the_true_group_count ||
        !clustering_beat_the_baseline || !projection_kept_the_structure) {
        std::cerr << "experiment failed its criteria: two-component ratio=" << two_component_ratio
                  << ", selected k=" << selected_clusters
                  << ", raw purity=" << plus_plus.test_purity
                  << ", projected purity=" << on_projection.test_purity << '\n';
        return 1;
    }
    return 0;
}
