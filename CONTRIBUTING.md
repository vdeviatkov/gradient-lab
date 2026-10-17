# Contributing

Start with [setup](docs/getting-started.md), the [repository layout](docs/design/repository-layout.md),
and the [implementation status](docs/status.md). Follow the contributor requirements in
[AGENTS.md](AGENTS.md).

## Make a focused change

1. Choose an existing topic or the next milestone in the [roadmap](docs/roadmap.md).
2. Put reusable algorithms in the appropriate implementation and orchestration in its experiment.
3. Explain mathematical and numerical decisions in the topic's derivation or design documentation.
4. Add tests with implementations. Use deterministic seeds and independent mathematical expectations.
5. Run the relevant tests and quality checks below.
6. Update the topic README, implementation guide, status matrix, and root implementation summary
   when completing a milestone.

Keep source paths and topic numbers stable where practical. Add documentation through the existing
indexes, and check local links when moving or renaming a file. The root README is an entry point;
long derivations, test inventories, and result analysis belong in dedicated guides.

## Verify the change

```bash
python -m pytest
ruff check .
python scripts/check_docs.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
git diff --check
```

Run checks appropriate to the changed implementation; [testing](docs/testing.md) explains focused
commands. Format C++ changes with the repository's `.clang-format` configuration.

## Preserve the educational contract

Write out the algorithm being taught. Framework APIs may support an implementation or comparison,
but must not substitute for that algorithm. Keep equivalent implementations behaviorally comparable
and document differences in initialization, loss scaling, updates, and stopping criteria.

Never invent experimental results. A report must include the command, configuration, seed,
environment, and measurement method. Follow the [results policy](results/README.md), and keep
large datasets, generated models, build outputs, credentials, and local environments out of Git.

## Describe the result

Use the [pull-request template](.github/pull_request_template.md). A pull request should explain the problem, resulting behavior, important design decisions, and
validation. Include measured outcomes only when they were actually produced, with enough detail to
reproduce them. Mention relevant limitations and update links to the final implementation.
