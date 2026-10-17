# Milestone requirements

[Topic progression and implementation status](../README.md#topics) · [Contributor guide](../CONTRIBUTING.md)

The numbered topics in the root table define the learning progression. Work through that same
sequence when adding a backend. The table is the only completion matrix; do not maintain a second
list here.

A milestone is complete when its topic contains a documented idea, an explicit implementation,
automated tests, a reproducible experiment, and an honest report of observed behavior. Update the
root table when it lands.

## Completion criteria

- Match the shared dataset, objective, initialization contract, updates, stopping conditions,
  and metrics, or explain intentional differences.
- Derive the important mathematics and numerical choices in the existing `idea.md`.
- Test against hand-computed expectations or independent numerical checks, including malformed
  inputs, stability limits, non-convergence, and deterministic seeds.
- Keep train, validation, and test roles separate. Select models on validation data and freeze the
  choice before reporting held-out performance.
- Add checkpoint round-trip tests where model persistence matters and CPU reference checks for
  accelerated implementations.
- Record command, configuration, seed, environment, and measurement method with results.
- For performance work, separate compilation, first-call, warm-up, and steady-state costs.
- Keep code and information in their canonical topic owner; link reused implementations.

Classical models should use inspectable data and make preprocessing explicit. Vision comparisons
should match splits and preprocessing. Sequence models should share their vocabulary, corpus,
and next-character objective. Generative models need evaluation beyond adversarial loss alone.
Reinforcement agents are evaluated without exploration noise on fixed held-out seeds.

Benchmark only implemented behavior. Leave unimplemented backends marked planned, and never fill
in missing measurements with estimates. See the [results policy](../results/README.md).
