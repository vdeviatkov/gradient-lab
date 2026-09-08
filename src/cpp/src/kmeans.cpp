#include "ml_scratch/kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace ml_scratch {
namespace {

// Draws a value in [0, 1) directly from the engine instead of using a standard distribution.
// std::mt19937 output is specified exactly, while distribution implementations are not, so this
// keeps seeded runs identical across standard libraries.
double uniform_unit(std::mt19937& engine) {
    constexpr double range = 4294967296.0; // std::mt19937::max() + 1
    return static_cast<double>(engine()) / range;
}

std::size_t uniform_index(std::mt19937& engine, const std::size_t bound) {
    return static_cast<std::size_t>(uniform_unit(engine) * static_cast<double>(bound)) % bound;
}

struct Assignment {
    std::size_t cluster;
    double squared_distance;
};

Assignment closest_centroid(const std::vector<double>& sample, const FeatureMatrix& centroids) {
    Assignment best{0, squared_euclidean_distance(sample, centroids.front())};
    for (std::size_t cluster = 1; cluster < centroids.size(); ++cluster) {
        const double distance = squared_euclidean_distance(sample, centroids[cluster]);
        if (distance < best.squared_distance) {
            best = {cluster, distance};
        }
    }
    return best;
}

FeatureMatrix initialize_random(const FeatureMatrix& data, const std::size_t cluster_count,
                                std::mt19937& engine) {
    std::vector<std::size_t> order(data.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }
    // Partial Fisher-Yates shuffle: the first cluster_count entries become distinct sample indices.
    for (std::size_t position = 0; position < cluster_count; ++position) {
        const std::size_t choice = position + uniform_index(engine, order.size() - position);
        std::swap(order[position], order[choice]);
    }

    FeatureMatrix centroids;
    centroids.reserve(cluster_count);
    for (std::size_t position = 0; position < cluster_count; ++position) {
        centroids.push_back(data[order[position]]);
    }
    return centroids;
}

FeatureMatrix initialize_plus_plus(const FeatureMatrix& data, const std::size_t cluster_count,
                                   std::mt19937& engine) {
    FeatureMatrix centroids;
    centroids.reserve(cluster_count);
    centroids.push_back(data[uniform_index(engine, data.size())]);

    std::vector<double> squared_distances(data.size(), std::numeric_limits<double>::infinity());
    while (centroids.size() < cluster_count) {
        double total = 0.0;
        for (std::size_t index = 0; index < data.size(); ++index) {
            squared_distances[index] =
                std::min(squared_distances[index],
                         squared_euclidean_distance(data[index], centroids.back()));
            total += squared_distances[index];
        }

        std::size_t chosen = data.size() - 1;
        if (total <= 0.0) {
            // Every remaining sample duplicates a centroid, so any choice is equally good.
            chosen = uniform_index(engine, data.size());
        } else {
            double target = uniform_unit(engine) * total;
            for (std::size_t index = 0; index < data.size(); ++index) {
                target -= squared_distances[index];
                if (target <= 0.0) {
                    chosen = index;
                    break;
                }
            }
        }
        centroids.push_back(data[chosen]);
    }
    return centroids;
}

} // namespace

double cluster_purity(const std::vector<std::size_t>& assignments,
                      const std::vector<std::size_t>& true_labels) {
    if (assignments.empty()) {
        throw std::invalid_argument("assignments must not be empty");
    }
    if (assignments.size() != true_labels.size()) {
        throw std::invalid_argument("assignments and true_labels must have equal sizes");
    }

    const std::size_t cluster_count = *std::max_element(assignments.begin(), assignments.end()) + 1;
    const std::size_t label_count = *std::max_element(true_labels.begin(), true_labels.end()) + 1;
    std::vector<std::vector<std::size_t>> counts(cluster_count,
                                                 std::vector<std::size_t>(label_count, 0));
    for (std::size_t index = 0; index < assignments.size(); ++index) {
        ++counts[assignments[index]][true_labels[index]];
    }

    std::size_t majority_total = 0;
    for (const auto& cluster_counts : counts) {
        majority_total += *std::max_element(cluster_counts.begin(), cluster_counts.end());
    }
    return static_cast<double>(majority_total) / static_cast<double>(assignments.size());
}

KMeans::KMeans(const std::size_t cluster_count) : cluster_count_(cluster_count) {
    if (cluster_count == 0) {
        throw std::invalid_argument("cluster_count must be positive");
    }
}

void KMeans::require_fitted() const {
    if (!fitted()) {
        throw std::logic_error("the model must be fitted first");
    }
}

std::size_t KMeans::feature_count() const noexcept {
    return centroids_.empty() ? 0 : centroids_.front().size();
}

std::size_t KMeans::predict(const std::vector<double>& sample) const {
    require_fitted();
    if (sample.size() != feature_count()) {
        throw std::invalid_argument("feature count does not match the fitted model");
    }
    for (const double feature : sample) {
        if (!std::isfinite(feature)) {
            throw std::invalid_argument("features must be finite");
        }
    }
    return closest_centroid(sample, centroids_).cluster;
}

std::vector<std::size_t> KMeans::predict(const FeatureMatrix& data) const {
    require_fitted();
    static_cast<void>(validate_feature_matrix(data));
    std::vector<std::size_t> assignments;
    assignments.reserve(data.size());
    for (const auto& sample : data) {
        assignments.push_back(predict(sample));
    }
    return assignments;
}

double KMeans::inertia(const FeatureMatrix& data) const {
    require_fitted();
    const std::size_t features = validate_feature_matrix(data);
    if (features != feature_count()) {
        throw std::invalid_argument("feature count does not match the fitted model");
    }

    double total = 0.0;
    for (const auto& sample : data) {
        total += closest_centroid(sample, centroids_).squared_distance;
    }
    return total;
}

KMeansTrainingResult KMeans::fit(const FeatureMatrix& data, const KMeansConfig& config) {
    const std::size_t features = validate_feature_matrix(data);
    if (cluster_count_ > data.size()) {
        throw std::invalid_argument("cluster_count must not exceed the sample count");
    }
    if (config.max_iterations == 0) {
        throw std::invalid_argument("max_iterations must be positive");
    }
    if (config.restarts == 0) {
        throw std::invalid_argument("restarts must be positive");
    }
    if (!std::isfinite(config.tolerance) || config.tolerance < 0.0) {
        throw std::invalid_argument("tolerance must be finite and non-negative");
    }

    KMeansTrainingResult result{std::numeric_limits<double>::infinity(), 0, false, 0, {}};
    result.restarts.reserve(config.restarts);
    FeatureMatrix best_centroids;
    std::vector<std::size_t> best_labels;

    for (std::size_t restart = 0; restart < config.restarts; ++restart) {
        const auto restart_seed = static_cast<std::uint32_t>(config.seed + restart);
        std::mt19937 engine{restart_seed};
        FeatureMatrix centroids = config.initialization == KMeansInitialization::kmeans_plus_plus
                                      ? initialize_plus_plus(data, cluster_count_, engine)
                                      : initialize_random(data, cluster_count_, engine);

        std::vector<std::size_t> assignments(data.size(), 0);
        std::vector<double> squared_distances(data.size(), 0.0);
        std::size_t iterations = 0;
        bool converged = false;

        while (iterations < config.max_iterations && !converged) {
            ++iterations;
            for (std::size_t index = 0; index < data.size(); ++index) {
                const Assignment assignment = closest_centroid(data[index], centroids);
                assignments[index] = assignment.cluster;
                squared_distances[index] = assignment.squared_distance;
            }

            FeatureMatrix sums(cluster_count_, std::vector<double>(features, 0.0));
            std::vector<std::size_t> counts(cluster_count_, 0);
            for (std::size_t index = 0; index < data.size(); ++index) {
                ++counts[assignments[index]];
                for (std::size_t feature = 0; feature < features; ++feature) {
                    sums[assignments[index]][feature] += data[index][feature];
                }
            }

            // An empty cluster has no mean. Reseeding it with the worst-served sample of a
            // cluster that can spare one keeps exactly cluster_count_ centroids and cannot
            // increase inertia. A donor with a single member is skipped so that emptying one
            // cluster never empties another.
            for (std::size_t cluster = 0; cluster < cluster_count_; ++cluster) {
                if (counts[cluster] != 0) {
                    continue;
                }

                bool found = false;
                std::size_t worst = 0;
                for (std::size_t index = 0; index < data.size(); ++index) {
                    if (counts[assignments[index]] < 2) {
                        continue;
                    }
                    if (!found || squared_distances[index] > squared_distances[worst]) {
                        worst = index;
                        found = true;
                    }
                }
                if (!found) {
                    continue;
                }

                const std::size_t donor = assignments[worst];
                --counts[donor];
                for (std::size_t feature = 0; feature < features; ++feature) {
                    sums[donor][feature] -= data[worst][feature];
                    sums[cluster][feature] = data[worst][feature];
                }
                counts[cluster] = 1;
                assignments[worst] = cluster;
                squared_distances[worst] = 0.0;
            }

            double movement = 0.0;
            for (std::size_t cluster = 0; cluster < cluster_count_; ++cluster) {
                if (counts[cluster] == 0) {
                    continue;
                }
                std::vector<double> updated(features, 0.0);
                for (std::size_t feature = 0; feature < features; ++feature) {
                    updated[feature] =
                        sums[cluster][feature] / static_cast<double>(counts[cluster]);
                }
                movement += squared_euclidean_distance(centroids[cluster], updated);
                centroids[cluster] = std::move(updated);
            }
            converged = movement <= config.tolerance;
        }

        double restart_inertia = 0.0;
        for (std::size_t index = 0; index < data.size(); ++index) {
            const Assignment assignment = closest_centroid(data[index], centroids);
            assignments[index] = assignment.cluster;
            restart_inertia += assignment.squared_distance;
        }
        result.restarts.push_back({restart_seed, restart_inertia, iterations, converged});

        if (restart_inertia < result.inertia) {
            result.inertia = restart_inertia;
            result.iterations = iterations;
            result.converged = converged;
            result.best_restart = restart;
            best_centroids = centroids;
            best_labels = assignments;
        }
    }

    centroids_ = std::move(best_centroids);
    labels_ = std::move(best_labels);
    return result;
}

} // namespace ml_scratch
