#pragma once

#include "ml_scratch/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ml_scratch {

enum class KMeansInitialization {
    // Distinct samples drawn uniformly at random.
    random_samples,
    // D^2 seeding: each new centroid is drawn with probability proportional to the squared
    // distance from the closest centroid already chosen.
    kmeans_plus_plus,
};

struct KMeansConfig {
    KMeansInitialization initialization{KMeansInitialization::kmeans_plus_plus};
    std::size_t max_iterations{300};
    // Independent restarts; the lowest-inertia one is kept. Restart r uses seed + r.
    std::size_t restarts{10};
    // Convergence threshold on the total squared centroid movement of one iteration.
    double tolerance{1e-12};
    std::uint32_t seed{0};
};

struct KMeansRestartSummary {
    std::uint32_t seed;
    double inertia;
    std::size_t iterations;
    bool converged;

    bool operator==(const KMeansRestartSummary&) const = default;
};

struct KMeansTrainingResult {
    double inertia;
    std::size_t iterations;
    bool converged;
    std::size_t best_restart;
    std::vector<KMeansRestartSummary> restarts;

    bool operator==(const KMeansTrainingResult&) const = default;
};

// Fraction of samples whose true label is the majority label of their assigned cluster. Purity is
// an evaluation metric only: it needs labels that clustering never sees during training.
[[nodiscard]] double cluster_purity(const std::vector<std::size_t>& assignments,
                                    const std::vector<std::size_t>& true_labels);

class KMeans {
  public:
    explicit KMeans(std::size_t cluster_count);

    KMeansTrainingResult fit(const FeatureMatrix& data, const KMeansConfig& config = {});

    [[nodiscard]] std::size_t predict(const std::vector<double>& sample) const;
    [[nodiscard]] std::vector<std::size_t> predict(const FeatureMatrix& data) const;
    // Sum over samples of the squared distance to the closest centroid.
    [[nodiscard]] double inertia(const FeatureMatrix& data) const;

    [[nodiscard]] std::size_t cluster_count() const noexcept { return cluster_count_; }
    [[nodiscard]] std::size_t feature_count() const noexcept;
    [[nodiscard]] bool fitted() const noexcept { return !centroids_.empty(); }
    [[nodiscard]] const FeatureMatrix& centroids() const noexcept { return centroids_; }
    // Cluster assignment of each training sample from the retained restart.
    [[nodiscard]] const std::vector<std::size_t>& labels() const noexcept { return labels_; }

  private:
    void require_fitted() const;

    std::size_t cluster_count_;
    FeatureMatrix centroids_;
    std::vector<std::size_t> labels_;
};

} // namespace ml_scratch
