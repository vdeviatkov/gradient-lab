import math
import random
from dataclasses import replace

import pytest
from ml_scratch.pure.pca_kmeans import KMeans, KMeansConfig, KMeansInitialization, cluster_purity

BLOBS = (
    (-1.0, 0.0),
    (1.0, 0.0),
    (0.0, -1.0),
    (0.0, 1.0),
    (9.0, 10.0),
    (11.0, 10.0),
    (10.0, 9.0),
    (10.0, 11.0),
)
LABELS = (0,) * 4 + (1,) * 4


@pytest.mark.parametrize("initialization", list(KMeansInitialization))
def test_recovers_hand_computed_centroids_and_inertia(initialization):
    model = KMeans(2)
    result = model.fit(BLOBS, KMeansConfig(initialization=initialization, seed=7, restarts=5))
    assert result.converged
    assert result.inertia == pytest.approx(8.0)
    assert model.inertia(BLOBS) == result.inertia
    assert set(model.centroids) == {(0.0, 0.0), (10.0, 10.0)}
    assert model.labels == model.predict_many(BLOBS)
    assert cluster_purity(model.labels, LABELS) == 1
    assert model.predict((0.0, 0.0)) != model.predict((10.0, 10.0))


def test_single_centroid_is_the_mean_and_not_the_first_sample():
    model = KMeans(1)
    result = model.fit(BLOBS)
    assert model.centroids == ((5.0, 5.0),)
    assert result.inertia == pytest.approx(408.0)


def test_restarts_retain_the_best_and_lloyd_steps_do_not_increase_inertia():
    model = KMeans(3)
    result = model.fit(BLOBS, KMeansConfig(seed=3, restarts=6))
    assert [r.seed for r in result.restarts] == list(range(3, 9))
    assert result.inertia == min(r.inertia for r in result.restarts)
    assert result.restarts[result.best_restart].inertia == result.inertia
    for restart in result.restarts:
        assert restart.iterations == len(restart.inertia_per_iteration)
        assert restart.inertia == restart.inertia_per_iteration[-1]
        assert all(
            b <= a + 1e-12
            for a, b in zip(restart.inertia_per_iteration, restart.inertia_per_iteration[1:])
        )


def test_more_clusters_and_one_cluster_per_sample_expose_the_selection_problem():
    inertias = [
        KMeans(k).fit(BLOBS, KMeansConfig(seed=11, restarts=8)).inertia
        for k in range(1, len(BLOBS) + 1)
    ]
    assert all(b <= a + 1e-12 for a, b in zip(inertias, inertias[1:]))
    assert inertias[-1] == 0


def test_empty_cluster_repair_keeps_finite_centroids_with_duplicate_points():
    data = ((0.0, 0.0),) * 3 + ((5.0, 5.0),) * 3
    model = KMeans(3)
    result = model.fit(data, KMeansConfig(seed=99, restarts=3))
    assert result.converged
    assert result.inertia == 0
    assert len(model.centroids) == 3
    assert all(math.isfinite(x) for row in model.centroids for x in row)
    assert set(model.labels) == {model.predict((0.0, 0.0)), model.predict((5.0, 5.0))}


def test_budget_exhaustion_is_reported_and_labels_match_final_centroids():
    model = KMeans(2)
    result = model.fit(BLOBS, KMeansConfig(max_iterations=1, restarts=1, seed=7))
    assert not result.converged
    assert result.iterations == 1
    assert result.inertia == model.inertia(BLOBS)
    assert model.labels == model.predict_many(BLOBS)


def test_seed_reproducibility_refits_and_no_global_rng_side_effect():
    config = KMeansConfig(seed=7, restarts=3)
    first, second = KMeans(3), KMeans(3)
    global_state = random.getstate()
    result = first.fit(BLOBS, config)
    assert result == second.fit(BLOBS, config)
    assert first.centroids == second.centroids
    assert result == first.fit(BLOBS, config)
    assert random.getstate() == global_state
    wrapped = KMeans(1).fit(BLOBS, replace(config, seed=2**32 - 1))
    assert [r.seed for r in wrapped.restarts] == [2**32 - 1, 0, 1]


def test_nearest_centroid_tie_chooses_lower_index():
    model = KMeans(2)
    model.fit(((-1.0,), (1.0,)))
    assert model.predict((0.0,)) == 0


def test_purity_is_permutation_invariant_and_uses_only_evaluation_labels():
    assert cluster_purity((0, 0, 1, 1), (4, 4, 9, 9)) == 1
    assert cluster_purity((100, 100, 20, 20), (4, 4, 9, 9)) == 1
    assert cluster_purity((0, 0, 0, 1), (0, 0, 1, 1)) == 0.75
    assert cluster_purity((0, 0, 0, 0), (0, 0, 1, 1)) == 0.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("restarts", 0),
        ("max_iterations", 0),
        ("restarts", 1.5),
        ("tolerance", -1),
        ("tolerance", math.nan),
        ("seed", -1),
        ("seed", 2**32),
        ("initialization", "unknown"),
    ],
)
def test_invalid_config_preserves_fitted_state(field, value):
    model = KMeans(2)
    model.fit(BLOBS)
    original = model.centroids, model.labels
    with pytest.raises(ValueError):
        model.fit(BLOBS, replace(KMeansConfig(), **{field: value}))
    assert (model.centroids, model.labels) == original


@pytest.mark.parametrize("data", [(), ((),), ((1.0,), (1.0, 2.0)), ((math.inf,), (1.0,))])
def test_bad_dataset(data):
    with pytest.raises(ValueError):
        KMeans(2).fit(data)


def test_unfitted_shape_count_and_overflow_errors():
    with pytest.raises(ValueError):
        KMeans(0)
    with pytest.raises(ValueError):
        KMeans(3).fit(((0.0,), (1.0,)))
    with pytest.raises(ValueError, match="fitted"):
        KMeans(1).predict((0.0,))
    model = KMeans(2)
    model.fit(BLOBS)
    with pytest.raises(ValueError):
        model.predict((0.0,))
    with pytest.raises(ValueError):
        model.inertia(((math.nan, 0.0),))
    with pytest.raises(ArithmeticError):
        KMeans(1).fit(((-1e200,), (1e200,)))


@pytest.mark.parametrize(
    "assignments,labels", [((), ()), ((0,), (0, 1)), ((-1,), (0,)), ((0,), (1.5,))]
)
def test_purity_rejects_invalid_inputs(assignments, labels):
    with pytest.raises(ValueError):
        cluster_purity(assignments, labels)
