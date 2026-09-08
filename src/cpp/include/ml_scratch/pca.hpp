#pragma once

#include "ml_scratch/dataset.hpp"

#include <cstddef>
#include <vector>

namespace ml_scratch {

struct SymmetricEigenDecomposition {
    // Sorted by descending eigenvalue. eigenvectors[i] is the unit eigenvector of eigenvalues[i].
    std::vector<double> eigenvalues;
    FeatureMatrix eigenvectors;

    bool operator==(const SymmetricEigenDecomposition&) const = default;
};

// Unbiased sample covariance, normalized by (sample_count - 1).
[[nodiscard]] FeatureMatrix covariance_matrix(const FeatureMatrix& data);

// Cyclic Jacobi rotations for a real symmetric matrix. Eigenvector signs are normalized so that
// each vector's largest-magnitude entry is positive, which makes results comparable across runs.
[[nodiscard]] SymmetricEigenDecomposition jacobi_eigen_decomposition(const FeatureMatrix& matrix,
                                                                     std::size_t max_sweeps = 100,
                                                                     double tolerance = 1e-12);

class PrincipalComponentAnalysis {
  public:
    explicit PrincipalComponentAnalysis(std::size_t component_count);

    void fit(const FeatureMatrix& data);

    [[nodiscard]] std::vector<double> transform(const std::vector<double>& sample) const;
    [[nodiscard]] FeatureMatrix transform(const FeatureMatrix& data) const;
    [[nodiscard]] std::vector<double>
    inverse_transform(const std::vector<double>& projection) const;
    // Mean squared error per feature between the samples and their rank-k reconstructions.
    [[nodiscard]] double reconstruction_error(const FeatureMatrix& data) const;

    [[nodiscard]] std::size_t component_count() const noexcept { return component_count_; }
    [[nodiscard]] std::size_t feature_count() const noexcept { return mean_.size(); }
    [[nodiscard]] bool fitted() const noexcept { return !mean_.empty(); }
    [[nodiscard]] const std::vector<double>& mean() const noexcept { return mean_; }
    // components()[i] is the i-th principal direction as a unit vector in feature space.
    [[nodiscard]] const FeatureMatrix& components() const noexcept { return components_; }
    // Variance captured by each kept component: the leading eigenvalues of the covariance matrix.
    [[nodiscard]] const std::vector<double>& explained_variance() const noexcept {
        return explained_variance_;
    }
    [[nodiscard]] const std::vector<double>& explained_variance_ratio() const noexcept {
        return explained_variance_ratio_;
    }
    [[nodiscard]] double cumulative_explained_variance_ratio() const noexcept;
    // Sum of all eigenvalues, equal to the total variance of the centered data.
    [[nodiscard]] double total_variance() const noexcept { return total_variance_; }

  private:
    void require_fitted() const;

    std::size_t component_count_;
    std::vector<double> mean_;
    FeatureMatrix components_;
    std::vector<double> explained_variance_;
    std::vector<double> explained_variance_ratio_;
    double total_variance_{0.0};
};

} // namespace ml_scratch
