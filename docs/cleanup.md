# OGO publication cleanup

This source release consolidates the existing OGO migration and the 2026-10-28
policy-optimization scope. It is a source/plan publication, not a new campaign
launch or a claim that ongoing experiments have completed.

## Changed

- Publish the `ogo` package, verified ImageNet adaptation code, separate profiling
  and production CLIs, and the existing immutable campaign configuration.
- Make synthetic mechanisms, ImageNet-64/256 and T2I the core plan; keep T2V
  optional. Use the same five initial baselines with explicit task adaptation.
- Add a machine-readable planning catalog and static checks for five-method
  coverage, 16/32-step deployment, implementation status and frozen config hashes.
- Repair immutable JSON metadata comparisons in both active writers. Python
  tuples serialize as lists; compare serialized values on resume while rejecting
  actual changes. Serialize before writing so invalid metadata cannot damage a
  committed record. Add a fresh-process regression without loading model code.

## Removed

- Retired `fct` namespace, pre-canonical prototype code, obsolete E1-E5 launch
  specifications and distillation-era documentation from the active source tree.
- Generated prototype plots, binary editor artifacts and the obsolete uploaded
  FCT experiment-plan HTML from GitHub main's active tree.
- Stale README claims that the separate standalone generator study remains active.

Previous committed versions remain in Git history. The pre-publication working
state, including untracked OGO sources, was separately snapshotted and hashed
before cleanup. Existing numerical evidence, reference inputs and deployed
campaign releases are not deleted by this publication.

## Naming and public interfaces

The installed namespace and commands are `ogo`; no `fct` compatibility alias is
installed. Historical fixture schema IDs, reference IDs and source-hash receipts
retain their original strings because they identify evidence. The synthetic
fixture runner retains its `teacher_steps` field for that established contract;
it does not introduce a new teacher/student comparison into the scientific plan.

The new plan is not accepted as an executable training config. T2I/T2V remain
explicitly unimplemented; no placeholder trainer or fake result is supplied.

## Open-source hygiene

Assets, private manifests, credentials, outputs, large weights and cluster
launchers stay outside source control. Public documentation describes explicit
path/hash contracts. Frozen configuration files preserve exact bytes through
Git attributes, including the profiling contract's original line endings.

## Validation and limits

Use `python tools/check_repository.py` and the metadata unit-test command in
[verification.md](verification.md). The new CI job performs those lightweight
checks without downloading weights or datasets. Historical numerical receipts
keep their own source hashes and are not relabeled as tests of this version.

This publication changes package source identity. It must not be copied over an
immutable deployed release or used to bypass a checkpoint's identity check.
The existing campaign's private, verified recovery mechanism remains separate.
Real-model parity, full numerical regression and every new task's adaptation
gates remain necessary before accepting measurements from a new source release.
