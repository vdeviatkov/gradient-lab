"""Depth selection and pruning on the same seeded noisy dataset as C++."""

import math
from dataclasses import dataclass

from .model import DecisionTreeClassifier, DecisionTreeConfig

SEED = 20260908


class _SplitMix64:
    """Dataset-only stream matching the C++ deterministic generator, including Box-Muller."""

    def __init__(self, seed: int):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
            raise ValueError("seed must be a uint64 integer")
        self.state = seed

    def uniform(self):
        mask = 2**64 - 1
        self.state = (self.state + 0x9E3779B97F4A7C15) & mask
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        value ^= value >> 31
        return (value >> 11) / 2**53

    def gaussian(self):
        first = max(self.uniform(), 1e-12)
        second = self.uniform()
        return math.sqrt(-2 * math.log(first)) * math.cos(2 * math.pi * second)


@dataclass(frozen=True)
class Split:
    features: tuple[tuple[float, ...], ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class Dataset:
    train: Split
    validation: Split
    test: Split


def build_dataset(seed=SEED):
    rng = _SplitMix64(seed)
    splits = [([], []) for _ in range(3)]
    for label in range(2):
        for index in range(300):
            center = 2.2 * label
            row = (
                center + 1.2 * rng.gaussian(),
                center + 1.2 * rng.gaussian(),
                rng.gaussian(),
                rng.gaussian(),
                rng.gaussian(),
            )
            observed = 1 - label if rng.uniform() < 0.08 else label
            features, labels = splits[0 if index < 150 else 1 if index < 225 else 2]
            features.append(row)
            labels.append(observed)
    return Dataset(*(Split(tuple(x), tuple(y)) for x, y in splits))


def evaluate(tree, data):
    return tuple(
        tree.accuracy(split.features, split.labels)
        for split in (data.train, data.validation, data.test)
    )


def run_experiment():
    data = build_dataset()
    print(f"Python decision tree experiment (seed={SEED})")
    print("splits: train=300, validation=150, test=150; 5 features; 8% label noise")
    reports = []
    for depth in range(1, 13):
        tree = DecisionTreeClassifier().fit(
            data.train.features, data.train.labels, DecisionTreeConfig(max_depth=depth)
        )
        scores = evaluate(tree, data)
        reports.append(scores)
        print(
            f"max_depth={depth:2}: leaves={tree.leaf_count:2}; "
            f"train={scores[0]:.4f}, validation={scores[1]:.4f}, test={scores[2]:.4f}"
        )
    selected_depth = max(range(1, 13), key=lambda d: reports[d - 1][1])
    print(f"selected max_depth={selected_depth} by validation accuracy (shallowest tie)")
    entropy = DecisionTreeClassifier().fit(
        data.train.features,
        data.train.labels,
        DecisionTreeConfig(criterion="entropy", max_depth=selected_depth),
    )
    print(
        "selected-depth entropy train/validation/test:",
        " / ".join(f"{score:.4f}" for score in evaluate(entropy, data)),
    )
    tree = DecisionTreeClassifier().fit(data.train.features, data.train.labels)
    before = evaluate(tree, data)
    nodes_before = tree.node_count
    print(
        f"unlimited: nodes={tree.node_count}, leaves={tree.leaf_count}, depth={tree.depth}; "
        + " / ".join(f"{score:.4f}" for score in before)
    )
    collapsed = tree.prune(data.validation.features, data.validation.labels)
    after = evaluate(tree, data)
    print(
        f"pruned: nodes={tree.node_count}, leaves={tree.leaf_count}, depth={tree.depth}; "
        + " / ".join(f"{score:.4f}" for score in after)
    )
    print(f"collapsed {collapsed} internal nodes")
    majority = max(range(2), key=data.train.labels.count)
    baseline = data.test.labels.count(majority) / len(data.test.labels)
    print(f"majority baseline test={baseline:.4f}")
    if not (
        before[0] == 1
        and reports[selected_depth - 1][2] > before[2]
        and tree.node_count < nodes_before
        and after[2] >= before[2]
        and reports[selected_depth - 1][2] > baseline
    ):
        raise RuntimeError("experiment failed its criteria")
    return selected_depth, before, after, tree.node_count


def main():
    run_experiment()


if __name__ == "__main__":
    main()
