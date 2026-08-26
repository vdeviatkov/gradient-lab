import pytest

from ml_scratch.pure.datasets import logic_gate_dataset


@pytest.mark.parametrize(
    ("gate", "targets"),
    [
        ("and", (0, 0, 0, 1)),
        ("or", (0, 1, 1, 1)),
        ("xor", (0, 1, 1, 0)),
    ],
)
def test_logic_gate_truth_tables(gate: str, targets: tuple[int, ...]) -> None:
    dataset = logic_gate_dataset(gate)  # type: ignore[arg-type]

    assert tuple(features for features, _ in dataset) == (
        (0.0, 0.0),
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
    )
    assert tuple(target for _, target in dataset) == targets


def test_unknown_gate_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown gate"):
        logic_gate_dataset("nand")  # type: ignore[arg-type]
