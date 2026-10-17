# Contributing

Start with [setup](docs/getting-started.md), the [repository layout](docs/architecture.md),
and the [implementation status](README.md#topics). Follow the contributor requirements in
[AGENTS.md](AGENTS.md).

## Make a focused change

1. Choose an existing topic or the next milestone in the [topic progression](README.md#topics).
2. Put algorithms, experiment orchestration, and tests in the topic’s backend folder.
3. Explain mathematical and numerical decisions in the topic's derivation or design documentation.
4. Add tests with implementations. Use deterministic seeds and independent mathematical expectations.
5. Run the relevant tests and quality checks below.
6. Update the topic README and idea, implementation guide, and root status matrix
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

Use the [pull-request template](.github/pull_request_template.md). Explain the problem, resulting
behavior, important design decisions, and validation. Include measured outcomes only when they were actually produced, with enough detail to
reproduce them. Mention relevant limitations and update links to the final implementation.

## Milestone requirements

Follow the numbered progression in the [root topic table](README.md#topics), which is the only
completion matrix. A completed milestone needs a topic-local idea, explicit implementation,
automated tests, reproducible experiment, and measured report.

- Match datasets, objectives, initialization, updates, stopping rules, and metrics across backends,
  or explain the differences.
- Derive mathematical and numerical decisions in the existing `idea.md`.
- Test hand-computed expectations, independent numerical checks, stability limits, malformed
  input, non-convergence, and seeded reproducibility.
- Use validation data for selection, then evaluate the frozen choice on held-out test data.
- Test checkpoint round trips where persistence matters and CPU references for accelerated code.
- Separate compilation, first-call, warm-up, and steady-state performance measurements.

Use inspectable data for classical models, matching splits for vision comparisons, a common
vocabulary/corpus/objective for sequence models, and evaluation beyond adversarial loss for GANs.
Evaluate reinforcement policies without exploration noise on held-out seeds. Leave unfinished
backends marked planned and follow the [results policy](results/README.md).
