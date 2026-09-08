#include "ml_scratch/pca.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ml_scratch {
namespace {

std::size_t validate_symmetric_matrix(const FeatureMatrix& matrix) {
    const std::size_t size = validate_feature_matrix(matrix);
    if (matrix.size() != size) {
        throw std::invalid_argument("matrix must be square");
    }
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t column = row + 1; column < size; ++column) {
            const double difference = std::abs(matrix[row][column] - matrix[column][row]);
            const double scale =
                std::max({1.0, std::abs(matrix[row][column]), std::abs(matrix[column][row])});
            if (difference > 1e-9 * scale) {
                throw std::invalid_argument("matrix must be symmetric");
            }
        }
    }
    return size;
}

double off_diagonal_norm(const FeatureMatrix& matrix) {
    double total = 0.0;
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t column = row + 1; column < matrix.size(); ++column) {
            total += 2.0 * matrix[row][column] * matrix[row][column];
        }
    }
    return std::sqrt(total);
}

// Flips each eigenvector so that its largest-magnitude entry is positive. Eigenvectors are only
// defined up to sign, so without this convention repeated runs could report mirrored components.
void normalize_signs(FeatureMatrix& eigenvectors) {
    for (auto& vector : eigenvectors) {
        std::size_t dominant = 0;
        for (std::size_t index = 1; index < vector.size(); ++index) {
            if (std::abs(vector[index]) > std::abs(vector[dominant])) {
                dominant = index;
            }
        }
        if (vector[dominant] < 0.0) {
            for (double& value : vector) {
                value = -value;
            }
        }
    }
}

} // namespace

FeatureMatrix covariance_matrix(const FeatureMatrix& data) {
    const std::size_t feature_count = validate_feature_matrix(data);
    if (data.size() < 2) {
        throw std::invalid_argument("covariance requires at least two samples");
    }

    const std::vector<double> means = column_means(data);
    FeatureMatrix covariance(feature_count, std::vector<double>(feature_count, 0.0));
    for (const auto& sample : data) {
        for (std::size_t row = 0; row < feature_count; ++row) {
            const double centered_row = sample[row] - means[row];
            for (std::size_t column = row; column < feature_count; ++column) {
                covariance[row][column] += centered_row * (sample[column] - means[column]);
            }
        }
    }

    const double inverse_degrees_of_freedom = 1.0 / static_cast<double>(data.size() - 1);
    for (std::size_t row = 0; row < feature_count; ++row) {
        for (std::size_t column = row; column < feature_count; ++column) {
            covariance[row][column] *= inverse_degrees_of_freedom;
            covariance[column][row] = covariance[row][column];
        }
    }
    return covariance;
}

SymmetricEigenDecomposition jacobi_eigen_decomposition(const FeatureMatrix& matrix,
                                                       const std::size_t max_sweeps,
                                                       const double tolerance) {
    const std::size_t size = validate_symmetric_matrix(matrix);
    if (max_sweeps == 0) {
        throw std::invalid_argument("max_sweeps must be positive");
    }
    if (!std::isfinite(tolerance) || tolerance <= 0.0) {
        throw std::invalid_argument("tolerance must be finite and positive");
    }

    FeatureMatrix working = matrix;
    // Columns of `rotations` accumulate the eigenvectors; it starts as the identity.
    FeatureMatrix rotations(size, std::vector<double>(size, 0.0));
    for (std::size_t index = 0; index < size; ++index) {
        rotations[index][index] = 1.0;
    }

    double scale = 0.0;
    for (std::size_t index = 0; index < size; ++index) {
        scale = std::max(scale, std::abs(working[index][index]));
    }
    const double threshold = tolerance * std::max(scale, 1.0);

    for (std::size_t sweep = 0; sweep < max_sweeps && off_diagonal_norm(working) > threshold;
         ++sweep) {
        for (std::size_t pivot = 0; pivot < size; ++pivot) {
            for (std::size_t other = pivot + 1; other < size; ++other) {
                if (std::abs(working[pivot][other]) <= threshold) {
                    continue;
                }

                // Choose the rotation angle that zeroes working[pivot][other]. The stable form
                // below avoids cancellation when the diagonal entries are nearly equal.
                const double difference = working[other][other] - working[pivot][pivot];
                const double theta = difference / (2.0 * working[pivot][other]);
                const double signed_root = (theta >= 0.0 ? 1.0 : -1.0) /
                                           (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double cosine = 1.0 / std::sqrt(signed_root * signed_root + 1.0);
                const double sine = signed_root * cosine;

                for (std::size_t index = 0; index < size; ++index) {
                    const double pivot_entry = working[index][pivot];
                    const double other_entry = working[index][other];
                    working[index][pivot] = cosine * pivot_entry - sine * other_entry;
                    working[index][other] = sine * pivot_entry + cosine * other_entry;
                }
                for (std::size_t index = 0; index < size; ++index) {
                    const double pivot_entry = working[pivot][index];
                    const double other_entry = working[other][index];
                    working[pivot][index] = cosine * pivot_entry - sine * other_entry;
                    working[other][index] = sine * pivot_entry + cosine * other_entry;
                }
                for (std::size_t index = 0; index < size; ++index) {
                    const double pivot_entry = rotations[index][pivot];
                    const double other_entry = rotations[index][other];
                    rotations[index][pivot] = cosine * pivot_entry - sine * other_entry;
                    rotations[index][other] = sine * pivot_entry + cosine * other_entry;
                }
            }
        }
    }

    std::vector<std::size_t> order(size);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&working](const std::size_t left, const std::size_t right) {
                  if (working[left][left] == working[right][right]) {
                      return left < right;
                  }
                  return working[left][left] > working[right][right];
              });

    SymmetricEigenDecomposition decomposition;
    decomposition.eigenvalues.reserve(size);
    decomposition.eigenvectors.reserve(size);
    for (const std::size_t index : order) {
        decomposition.eigenvalues.push_back(working[index][index]);
        std::vector<double> eigenvector(size);
        for (std::size_t row = 0; row < size; ++row) {
            eigenvector[row] = rotations[row][index];
        }
        decomposition.eigenvectors.push_back(std::move(eigenvector));
    }
    normalize_signs(decomposition.eigenvectors);
    return decomposition;
}

PrincipalComponentAnalysis::PrincipalComponentAnalysis(const std::size_t component_count)
    : component_count_(component_count) {
    if (component_count == 0) {
        throw std::invalid_argument("component_count must be positive");
    }
}

void PrincipalComponentAnalysis::require_fitted() const {
    if (!fitted()) {
        throw std::logic_error("the model must be fitted first");
    }
}

void PrincipalComponentAnalysis::fit(const FeatureMatrix& data) {
    const std::size_t features = validate_feature_matrix(data);
    if (component_count_ > features) {
        throw std::invalid_argument("component_count must not exceed the feature count");
    }

    const FeatureMatrix covariance = covariance_matrix(data);
    const SymmetricEigenDecomposition decomposition = jacobi_eigen_decomposition(covariance);

    mean_ = column_means(data);
    components_.assign(decomposition.eigenvectors.begin(),
                       decomposition.eigenvectors.begin() +
                           static_cast<std::ptrdiff_t>(component_count_));
    explained_variance_.assign(decomposition.eigenvalues.begin(),
                               decomposition.eigenvalues.begin() +
                                   static_cast<std::ptrdiff_t>(component_count_));
    // Rounding can leave a tiny negative eigenvalue on a rank-deficient covariance matrix.
    for (double& variance : explained_variance_) {
        variance = std::max(variance, 0.0);
    }

    total_variance_ = 0.0;
    for (const double eigenvalue : decomposition.eigenvalues) {
        total_variance_ += std::max(eigenvalue, 0.0);
    }
    explained_variance_ratio_.assign(component_count_, 0.0);
    if (total_variance_ > 0.0) {
        for (std::size_t component = 0; component < component_count_; ++component) {
            explained_variance_ratio_[component] = explained_variance_[component] / total_variance_;
        }
    }
}

std::vector<double> PrincipalComponentAnalysis::transform(const std::vector<double>& sample) const {
    require_fitted();
    if (sample.size() != feature_count()) {
        throw std::invalid_argument("feature count does not match the fitted model");
    }

    std::vector<double> projection(component_count_, 0.0);
    for (std::size_t component = 0; component < component_count_; ++component) {
        double coordinate = 0.0;
        for (std::size_t feature = 0; feature < feature_count(); ++feature) {
            if (!std::isfinite(sample[feature])) {
                throw std::invalid_argument("features must be finite");
            }
            coordinate += components_[component][feature] * (sample[feature] - mean_[feature]);
        }
        projection[component] = coordinate;
    }
    return projection;
}

FeatureMatrix PrincipalComponentAnalysis::transform(const FeatureMatrix& data) const {
    require_fitted();
    static_cast<void>(validate_feature_matrix(data));
    FeatureMatrix projections;
    projections.reserve(data.size());
    for (const auto& sample : data) {
        projections.push_back(transform(sample));
    }
    return projections;
}

std::vector<double>
PrincipalComponentAnalysis::inverse_transform(const std::vector<double>& projection) const {
    require_fitted();
    if (projection.size() != component_count_) {
        throw std::invalid_argument("projection size does not match the component count");
    }

    std::vector<double> reconstruction = mean_;
    for (std::size_t component = 0; component < component_count_; ++component) {
        if (!std::isfinite(projection[component])) {
            throw std::invalid_argument("projection values must be finite");
        }
        for (std::size_t feature = 0; feature < feature_count(); ++feature) {
            reconstruction[feature] += projection[component] * components_[component][feature];
        }
    }
    return reconstruction;
}

double PrincipalComponentAnalysis::reconstruction_error(const FeatureMatrix& data) const {
    require_fitted();
    const std::size_t features = validate_feature_matrix(data);
    if (features != feature_count()) {
        throw std::invalid_argument("feature count does not match the fitted model");
    }

    double squared_error = 0.0;
    for (const auto& sample : data) {
        squared_error += squared_euclidean_distance(sample, inverse_transform(transform(sample)));
    }
    return squared_error / static_cast<double>(data.size() * features);
}

double PrincipalComponentAnalysis::cumulative_explained_variance_ratio() const noexcept {
    return std::accumulate(explained_variance_ratio_.begin(), explained_variance_ratio_.end(), 0.0);
}

} // namespace ml_scratch
