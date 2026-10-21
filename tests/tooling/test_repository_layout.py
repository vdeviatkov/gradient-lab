"""Guard the topic-first ownership contract, not the contents of individual algorithms."""

import hashlib
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def topic_directories():
    return sorted(path for path in (ROOT / "topics").iterdir() if path.is_dir())


def test_each_topic_owns_its_idea_cpp_entry_point_and_source_registration():
    topics = topic_directories()
    assert topics
    assert [int(path.name.split("_", 1)[0]) for path in topics] == list(range(1, len(topics) + 1))
    home = (ROOT / "README.md").read_text()
    for topic in topics:
        for name in ["README.md", "idea.md", "cpp/README.md", "cpp/main.cpp", "cpp/CMakeLists.txt"]:
            assert (topic / name).is_file(), topic / name
        for name in ["README.md", "idea.md", "cpp/README.md"]:
            assert str((topic / name).relative_to(ROOT)) in home


def test_python_package_map_installs_topic_sources_without_a_mirror_tree():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    mapping = config["tool"]["setuptools"]["package-dir"]
    packages = config["tool"]["setuptools"]["packages"]
    assert set(mapping) == set(packages)
    declared_topics = set()
    for name, location in mapping.items():
        directory = ROOT / location
        assert (directory / "__init__.py").is_file()
        if name in {"ml_scratch", "ml_scratch.pure", "ml_scratch.numpy"}:
            assert location.startswith("shared/python/")
        else:
            assert location.startswith("topics/")
            assert location.endswith(
                "/numpy" if name.startswith("ml_scratch.numpy.") else "/python"
            )
            for file in ["model.py", "experiment.py", "README.md"]:
                assert (directory / file).is_file()
            assert list((directory / "tests").glob("test_*.py"))
            declared_topics.add(directory.resolve())
    assert declared_topics == {
        path.resolve()
        for backend in ("python", "numpy")
        for path in (ROOT / "topics").glob(f"*/{backend}")
    }
    for entrypoint in config["project"]["scripts"].values():
        module, function = entrypoint.split(":")
        package, leaf = module.rsplit(".", 1)
        assert package in mapping and leaf == "experiment" and function == "main"


def test_topic_algorithms_have_one_canonical_source_copy():
    owners = {}
    files = list((ROOT / "topics").glob("*/cpp/*.cpp"))
    files += list((ROOT / "topics").glob("*/python/model.py"))
    files += list((ROOT / "topics").glob("*/numpy/model.py"))
    for path in files:
        if path.name == "main.cpp":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest not in owners, f"Copied algorithm: {owners.get(digest)} and {path}"
        owners[digest] = path
    # Former parallel trees must not retain model source after a migration.
    for old in [ROOT / "src", ROOT / "experiments"]:
        assert not list(old.rglob("*.cpp"))
        assert not list(old.rglob("*.py"))


def test_benchmark_inventory_resolves_and_counts_relocated_sources_once():
    benchmark = ROOT / "topics/19_benchmarks/cpp/main.cpp"
    text = benchmark.read_text()
    section = text.split("const std::vector<Module> modules{", 1)[1].split(
        "auto measured_modules", 1
    )[0]
    paths = re.findall(r'"((?:topics|shared)/[^"\n]+\.(?:cpp|hpp))"', section)
    assert paths
    assert len(paths) == len(set(paths)), "A relocated source is counted more than once"
    for path in paths:
        assert (ROOT / path).is_file(), path
