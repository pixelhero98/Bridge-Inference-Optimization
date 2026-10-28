# Numerical verification

## Checks of the public source

`python tools/check_repository.py` performs standard-library source/import,
entry-point, JSON/TOML, local-document-link and experiment-plan checks. It also
checks that the frozen ImageNet campaign and profiling configuration hashes
match the preserved contract. It runs no model or numerical experiment.

`python -B -m unittest discover -s tests -p test_metadata_records.py -v` checks
the two actual metadata writers across fresh processes, accepts tuple/list JSON
round trips, rejects changed identities and nonfinite metadata, and preserves
existing bytes on failure. These checks cover the public metadata repair only.
They do not replace numerical or allocated real-model verification.

## Historical and allocated verification

The OGO profiling release has separate numerical and real-model acceptance
checks. `ogo-study verify` writes the configuration and source identity together
with independent reference calculations. `ogo-study audit` rejects incomplete
case coverage and missing real-model verification.

The numerical groups cover Gaussian density/gradient formulas; EDM score and
SiT stochastic-drift conversions; PPO clipping; lambda statistics and detached
weights; D3PO reference-ratio clipping; DRaFT truncated differentiation; and
LoRA reference disabling, updates and exact AdamW state replay. Central finite
differences and direct formulas are distinct checks.

Every measured backbone/method/horizon must additionally pass initial
zero-correction Euler equivalence, paired LPIPS parity, on-policy likelihood
replay where applicable, finite gradients, frozen-backbone checks and exact
interrupted/resumed weights and optimizer state. Matching references and
independent development references occupy separate immutable partitions.
No case is accepted solely because the numerical fixtures passed.

The frozen profiling thresholds are in `configs/ogo-profile.json`. Timings
carry their tested source hashes. Cleanup changes made after a timed release
receive a separate verification receipt; a later package is never relabeled
as the source that produced earlier measurements.

The retained canonical scalar/nonlinear fixtures and
`configs/gradient-correctness.json` preserve their original thresholds and
historical schema identifier. The compact [original verification record](verification-results.json)
identifies its historical source. Earlier E1-E5 component documentation is in
the project evidence archive; it is not an active experiment specification.

These checks establish the tested implementation behavior. They do not
establish ImageNet generation quality, population convergence, or equivalence
to the original papers' different architectures and samplers.
