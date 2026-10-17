import pytest
from ml_scratch.pure.pca_kmeans.experiment import (
    SEED,
    _SplitMix64,
    build_dataset,
    main,
    run_experiment,
    select_cluster_count,
)


def test_splitmix64_known_first_word_and_dataset_reproducibility():
    assert _SplitMix64(0).uniform() == (0xE220A8397B1DCDAF >> 11) / 2**53
    data = build_dataset()
    assert data == build_dataset(SEED)
    assert data != build_dataset(SEED + 1)
    for split, size in ((data.train, 72), (data.validation, 24), (data.test, 24)):
        assert len(split.features) == len(split.labels) == size
        assert all(len(row) == 4 for row in split.features)
        assert [split.labels.count(label) for label in range(3)] == [size // 3] * 3
    assert not set(data.train.features) & set(data.test.features)


def test_validation_elbow_uses_only_inertias_and_lower_k_breaks_ties():
    assert select_cluster_count((100.0, 80.0, 10.0, 8.0, 7.0)) == 3
    assert select_cluster_count((4.0, 3.0, 2.0, 1.0)) == 2
    for values in ((), (1.0, 0.0), (3.0, -1.0, 0.0), (3.0, float("nan"), 1.0)):
        with pytest.raises(ValueError):
            select_cluster_count(values)


def test_experiment_recovers_plane_and_groups_and_matches_cpp_recorded_data():
    result = run_experiment()
    assert result == run_experiment()
    assert result.selected_clusters == 3
    assert sum(result.variance_ratio[:2]) >= 0.99
    assert result.reconstruction_errors[1] < result.reconstruction_errors[0] / 100
    assert result.reconstruction_errors[-1] < 1e-24
    assert result.raw.test_purity == result.projected.test_purity == 1
    assert result.baseline.test_purity == pytest.approx(1 / 3)
    assert result.raw.test_inertia < result.baseline.test_inertia
    # Independent C++ experiment's documented values, with tolerance for printed precision.
    assert result.explained_variance[:2] == pytest.approx((11.047807, 7.101859), abs=1e-6)
    assert result.baseline.test_inertia == pytest.approx(396.979729, abs=1e-6)


def test_cli_reports_current_seed_selection_and_held_out_metrics(capsys):
    assert main([]) == 0
    text = capsys.readouterr().out
    assert f"seed={SEED}" in text
    assert "selected k=3" in text
    assert "test purity=1.000000" in text
