import pytest
from ml_scratch.pure.decision_tree.experiment import build_dataset, run_experiment


def test_dataset_reproducibility_and_split_sizes():
    data = build_dataset()
    assert data == build_dataset()
    assert data != build_dataset(1)
    assert [len(s.labels) for s in (data.train, data.validation, data.test)] == [300, 150, 150]
    rows = [set(s.features) for s in (data.train, data.validation, data.test)]
    assert not rows[0] & rows[1] and not rows[0] & rows[2] and not rows[1] & rows[2]


def test_measured_cpp_reference_results(capsys):
    depth, before, after, nodes = run_experiment()
    assert depth == 2
    assert before == pytest.approx((1, 0.74, 121 / 150))
    assert after == pytest.approx((257 / 300, 0.84, 131 / 150))
    assert nodes == 13
    assert "seed=20260908" in capsys.readouterr().out
