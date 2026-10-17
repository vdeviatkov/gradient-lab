#include "ml_scratch/pca.hpp"

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

// Exactly collinear points along the direction (2, 1)/sqrt(5), centered on (3, 4).
const ml_scratch::FeatureMatrix collinear_data{
    {-1.0, 2.0}, {1.0, 3.0}, {3.0, 4.0}, {5.0, 5.0}, {7.0, 6.0},
};

void test_covariance_matrix() {
    const ml_scratch::FeatureMatrix data{{1.0, 10.0}, {2.0, 8.0}, {3.0, 6.0}, {4.0, 4.0}};
    const auto covariance = ml_scratch::covariance_matrix(data);

    // Variances use the unbiased (n - 1) normalization: var(x) = 5/3, var(y) = 4 * var(x).
    require_near(covariance[0][0], 5.0 / 3.0, 1e-12, "incorrect first variance");
    require_near(covariance[1][1], 20.0 / 3.0, 1e-12, "incorrect second variance");
    require_near(covariance[0][1], -10.0 / 3.0, 1e-12, "incorrect covariance");
    require_near(covariance[0][1], covariance[1][0], 0.0, "covariance matrix is not symmetric");
}

void test_eigen_decomposition() {
    const ml_scratch::FeatureMatrix matrix{{2.0, 1.0}, {1.0, 2.0}};
    const auto decomposition = ml_scratch::jacobi_eigen_decomposition(matrix);

    require(decomposition.eigenvalues.size() == 2, "wrong eigenvalue count");
    require_near(decomposition.eigenvalues[0], 3.0, 1e-12, "incorrect leading eigenvalue");
    require_near(decomposition.eigenvalues[1], 1.0, 1e-12, "incorrect trailing eigenvalue");

    const double inverse_root_two = 1.0 / std::sqrt(2.0);
    require_near(decomposition.eigenvectors[0][0], inverse_root_two, 1e-12,
                 "incorrect leading eigenvector");
    require_near(decomposition.eigenvectors[0][1], inverse_root_two, 1e-12,
                 "incorrect leading eigenvector");
    // The sign convention makes the dominant entry positive, so the second vector is (1, -1).
    require_near(decomposition.eigenvectors[1][0], inverse_root_two, 1e-12,
                 "sign convention was not applied");
    require_near(decomposition.eigenvectors[1][1], -inverse_root_two, 1e-12,
                 "incorrect trailing eigenvector");

    for (std::size_t index = 0; index < decomposition.eigenvalues.size(); ++index) {
        const auto& vector = decomposition.eigenvectors[index];
        double norm = 0.0;
        for (const double value : vector) {
            norm += value * value;
        }
        require_near(norm, 1.0, 1e-12, "eigenvector is not a unit vector");

        // A x must equal lambda x for every returned pair.
        for (std::size_t row = 0; row < matrix.size(); ++row) {
            double product = 0.0;
            for (std::size_t column = 0; column < matrix.size(); ++column) {
                product += matrix[row][column] * vector[column];
            }
            require_near(product, decomposition.eigenvalues[index] * vector[row], 1e-10,
                         "eigenpair does not satisfy A x = lambda x");
        }
    }
}

void test_eigen_decomposition_of_a_diagonal_matrix() {
    const ml_scratch::FeatureMatrix matrix{{1.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 3.0}};
    const auto decomposition = ml_scratch::jacobi_eigen_decomposition(matrix);

    require_near(decomposition.eigenvalues[0], 5.0, 1e-12, "diagonal matrix was not sorted");
    require_near(decomposition.eigenvalues[1], 3.0, 1e-12, "diagonal matrix was not sorted");
    require_near(decomposition.eigenvalues[2], 1.0, 1e-12, "diagonal matrix was not sorted");
    require_near(decomposition.eigenvectors[0][1], 1.0, 1e-12, "incorrect diagonal eigenvector");
    require_near(decomposition.eigenvectors[1][2], 1.0, 1e-12, "incorrect diagonal eigenvector");
    require_near(decomposition.eigenvectors[2][0], 1.0, 1e-12, "incorrect diagonal eigenvector");
}

void test_pca_recovers_a_known_direction() {
    ml_scratch::PrincipalComponentAnalysis model{1};
    model.fit(collinear_data);

    require(model.feature_count() == 2, "incorrect feature count");
    require_near(model.mean()[0], 3.0, 1e-12, "incorrect mean");
    require_near(model.mean()[1], 4.0, 1e-12, "incorrect mean");

    const double root_five = std::sqrt(5.0);
    require_near(model.components()[0][0], 2.0 / root_five, 1e-10, "incorrect principal direction");
    require_near(model.components()[0][1], 1.0 / root_five, 1e-10, "incorrect principal direction");

    // The data is exactly one-dimensional, so one component explains all of the variance and the
    // reconstruction is exact.
    require_near(model.cumulative_explained_variance_ratio(), 1.0, 1e-10,
                 "collinear data was not fully explained");
    require_near(model.reconstruction_error(collinear_data), 0.0, 1e-18,
                 "collinear data was not reconstructed exactly");
    require_near(model.explained_variance()[0], model.total_variance(), 1e-10,
                 "explained variance does not match the total variance");
}

void test_projection_and_reconstruction() {
    const ml_scratch::FeatureMatrix data{
        {-2.0, -1.0, 0.5}, {-1.0, 0.0, -0.5}, {0.0, 1.0, 0.5},
        {1.0, 2.0, -0.5},  {2.0, 3.0, 0.5},   {0.5, -2.0, -0.5},
    };

    ml_scratch::PrincipalComponentAnalysis full{3};
    full.fit(data);
    // Keeping every component is a change of basis, so nothing is lost.
    require_near(full.cumulative_explained_variance_ratio(), 1.0, 1e-10,
                 "a full-rank projection lost variance");
    require_near(full.reconstruction_error(data), 0.0, 1e-20,
                 "a full-rank projection lost information");

    ml_scratch::PrincipalComponentAnalysis reduced{2};
    reduced.fit(data);
    require(reduced.reconstruction_error(data) > full.reconstruction_error(data),
            "dropping a component should cost reconstruction accuracy");
    require(reduced.explained_variance()[0] >= reduced.explained_variance()[1],
            "explained variance is not sorted");
    require(reduced.cumulative_explained_variance_ratio() < 1.0,
            "a truncated projection cannot explain all variance");

    // Projected coordinates are centered and uncorrelated by construction.
    const auto projections = reduced.transform(data);
    require(projections.size() == data.size(), "transform returned the wrong sample count");
    double first_sum = 0.0;
    double cross_sum = 0.0;
    for (const auto& projection : projections) {
        first_sum += projection[0];
        cross_sum += projection[0] * projection[1];
    }
    require_near(first_sum, 0.0, 1e-10, "projected coordinates are not centered");
    require_near(cross_sum, 0.0, 1e-10, "projected coordinates are correlated");

    // Variance of a projected coordinate equals its component's eigenvalue.
    double first_variance = 0.0;
    for (const auto& projection : projections) {
        first_variance += projection[0] * projection[0];
    }
    first_variance /= static_cast<double>(data.size() - 1);
    require_near(first_variance, reduced.explained_variance()[0], 1e-10,
                 "projected variance does not match the eigenvalue");
}

void test_determinism() {
    ml_scratch::PrincipalComponentAnalysis first{2};
    ml_scratch::PrincipalComponentAnalysis second{2};
    const ml_scratch::FeatureMatrix data{
        {1.0, 0.2}, {2.0, 1.9}, {3.0, 3.4}, {4.0, 3.8}, {5.0, 5.6},
    };
    first.fit(data);
    second.fit(data);

    require(first.components() == second.components(), "PCA is not deterministic");
    require(first.explained_variance() == second.explained_variance(), "PCA is not deterministic");
}

void test_input_validation() {
    require_invalid_argument([] { ml_scratch::PrincipalComponentAnalysis{0}; },
                             "zero component count was accepted");
    require_invalid_argument([] { static_cast<void>(ml_scratch::covariance_matrix({{1.0, 2.0}})); },
                             "a single sample was accepted for covariance");
    require_invalid_argument(
        [] { static_cast<void>(ml_scratch::covariance_matrix({{1.0}, {1.0, 2.0}})); },
        "a ragged matrix was accepted");
    require_invalid_argument(
        [] {
            static_cast<void>(
                ml_scratch::covariance_matrix({{std::numeric_limits<double>::quiet_NaN()}, {1.0}}));
        },
        "a non-finite feature was accepted");
    require_invalid_argument(
        [] { static_cast<void>(ml_scratch::jacobi_eigen_decomposition({{1.0, 2.0}, {3.0, 4.0}})); },
        "an asymmetric matrix was accepted");
    require_invalid_argument(
        [] { static_cast<void>(ml_scratch::jacobi_eigen_decomposition({{1.0, 0.0}}, 0)); },
        "a zero sweep budget was accepted");

    ml_scratch::PrincipalComponentAnalysis model{3};
    require_invalid_argument([&] { model.fit(collinear_data); },
                             "more components than features was accepted");

    bool rejected = false;
    try {
        static_cast<void>(model.transform(std::vector<double>{1.0, 2.0}));
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "an unfitted model produced a projection");

    ml_scratch::PrincipalComponentAnalysis fitted{1};
    fitted.fit(collinear_data);
    require_invalid_argument([&] { static_cast<void>(fitted.transform(std::vector<double>{1.0})); },
                             "a wrong feature count was accepted");
    require_invalid_argument(
        [&] { static_cast<void>(fitted.inverse_transform(std::vector<double>{1.0, 2.0})); },
        "a wrong projection size was accepted");
}

} // namespace

int main() {
    try {
        test_covariance_matrix();
        test_eigen_decomposition();
        test_eigen_decomposition_of_a_diagonal_matrix();
        test_pca_recovers_a_known_direction();
        test_projection_and_reconstruction();
        test_determinism();
        test_input_validation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
