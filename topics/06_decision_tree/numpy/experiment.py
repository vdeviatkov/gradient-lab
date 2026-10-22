"""Depth selection and reduced-error pruning on the scalar experiment's seeded noisy dataset."""

import argparse
from dataclasses import dataclass

from ml_scratch.pure.decision_tree.experiment import SEED, build_dataset

from .model import DecisionTreeClassifier, DecisionTreeConfig

DEPTHS = range(1, 13)


@dataclass(frozen=True)
class TreeReport:
    nodes: int
    leaves: int
    depth: int
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class ExperimentResult:
    depth_sweep: tuple[TreeReport, ...]  # max_depth = 1..12
    selected_depth: int
    entropy: TreeReport
    unlimited: TreeReport
    pruned: TreeReport
    collapsed: int
    baseline: float


def _report(tree, data):
    return TreeReport(
        tree.node_count,
        tree.leaf_count,
        tree.depth,
        *(
            tree.accuracy(split.features, split.labels)
            for split in (data.train, data.validation, data.test)
        ),
    )


def run_experiment(seed=SEED):
    data = build_dataset(seed)

    def grow(config=DecisionTreeConfig()):
        return DecisionTreeClassifier().fit(data.train.features, data.train.labels, config)

    sweep = tuple(_report(grow(DecisionTreeConfig(max_depth=d)), data) for d in DEPTHS)
    # max() keeps the first maximum, so validation ties select the shallowest tree.
    selected = max(DEPTHS, key=lambda d: sweep[d - 1].validation)
    entropy = _report(grow(DecisionTreeConfig(criterion="entropy", max_depth=selected)), data)
    tree = grow()
    unlimited = _report(tree, data)
    collapsed = tree.prune(data.validation.features, data.validation.labels)
    pruned = _report(tree, data)
    majority = max(range(2), key=data.train.labels.count)
    baseline = data.test.labels.count(majority) / len(data.test.labels)
    result = ExperimentResult(sweep, selected, entropy, unlimited, pruned, collapsed, baseline)
    if not (
        unlimited.train == 1
        and sweep[selected - 1].test > unlimited.test
        and pruned.nodes < unlimited.nodes
        and pruned.test >= unlimited.test
        and sweep[selected - 1].test > baseline
    ):
        raise RuntimeError("decision tree experiment failed its criteria")
    return result


def _format(report):
    return (
        f"nodes={report.nodes}, leaves={report.leaves}, depth={report.depth}; "
        f"train={report.train:.4f}, validation={report.validation:.4f}, test={report.test:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(f"NumPy decision tree experiment (seed={args.seed})")
    result = run_experiment(args.seed)
    for depth, report in zip(DEPTHS, result.depth_sweep, strict=True):
        print(f"max_depth={depth:2}: {_format(report)}")
    print(f"selected max_depth={result.selected_depth} by validation accuracy (shallowest tie)")
    print(f"selected-depth entropy: {_format(result.entropy)}")
    print(f"unlimited: {_format(result.unlimited)}")
    print(f"pruned: {_format(result.pruned)}")
    print(f"collapsed {result.collapsed} internal nodes")
    print(f"majority baseline test={result.baseline:.4f}")


if __name__ == "__main__":
    main()
