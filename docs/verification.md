# Verification record

The installed canonical package passed its complete declared verification in job
**7138497**, starting **2026-10-07 19:56:15 UTC**, on one NVIDIA GH200 120GB.
The job completed with exit code zero in 65 seconds; numerical E0 verification took
18.31 seconds. Tests and commands ran outside the checkout against the installed
wheel. [Machine-readable evidence](verification-results.json) records the outcomes,
tested source and configuration hashes, wheel hash, and environment versions.

| Check | Result |
|---|---|
| Structural and training-contract tests | 21 passed |
| Scalar closed-form fixtures | 30 passed across CPU/CUDA |
| Nonlinear 2D fixtures | 306 passed across CPU/CUDA |
| CPU/GPU objective and gradient comparisons | 36 passed |
| Maximum autograd–adjoint absolute difference | 7.77156e-16 |
| Worst full-gradient three-perturbation window ratio | 0.0325078; acceptance limit 1 |
| Maximum CPU/GPU gradient absolute difference | 4.44089e-16 |
| Backbone-input derivative omission detected | 120/120 applicable fixtures |
| Final-step-only derivative detected | 120/120 applicable fixtures |
| One-sided generated-kernel derivative detected | 80/80 applicable fixtures |
| Extra ensemble mean detected | 120/120 applicable fixtures |
| Independent saved-evidence audit | 230,850 rows; 15,390 direction groups; 336 fixtures; 12 source files |
| Installed-wheel same-step FP32 / reduced-step FP64 smoke runs | Both passed |

Control counts are diagnostic fixtures, not independent statistical replicates.
There were no failing full-gradient fixtures. Expected failures of incomplete
controls remain in the raw measurements.

The 21 tests include independent objective loops, both generated kernel arguments,
normalization, context separation, K<2 rejection when matching, zero-correction
equality, independent target and paired-baseline identities, multiple targets,
latent plus decoded perceptual gradient flow, frozen-module invariants, CPU/GPU
checkpoint/offload equivalence, and exact checkpoint/resume before, at, and after
the matching-to-joint transition. The perceptual test uses small frozen decoder
and encoder stand-ins; it does not validate a pretrained DINO integration.

## Fixed acceptance protocol

`configs/gradient-correctness.json` retains the original fixtures and thresholds:

- Full-gradient directional FD error <= 1e-9 + 1e-6 * max(|g.d|, |FD|), at three
  consecutive perturbations in every direction; ten directions and 15 perturbations
  per nonlinear fixture, with the scalar derivative checked directly.
- Autograd/adjoint and scalar-oracle agreement: 1e-10 absolute + 1e-8 relative.
- Independent objective values: 1e-12 absolute + 1e-12 relative.
- All four incomplete controls must be detected on applicable nonzero multistep cases.

The run used aarch64, Python 3.11.7, PyTorch 2.9.0+cu128, CUDA 12.8 and NumPy 2.2.6.
CPU float64 is the reference; representative fixtures repeat in GPU float64.
Source hashes in the published record were checked against the released package,
tests and configurations. Subsequent edits affect documentation and archive cleanup.

## Evidence and scope

Raw directional measurements, fixture arrays, gradients, summaries, agreement tables,
error-versus-perturbation figures, checkpoints, environment logs and the installed
wheel are preserved in the project evidence store. The public record includes their
key hashes and compact results; private cluster configuration is excluded.
Reproduce a new run using the commands in the repository README.

The original independent E0 reference (job 7080639) also passed its 11 structural
tests, 336 scalar/nonlinear fixtures, 36 CPU/GPU comparisons and all four controls.
Its source and evidence remain unchanged. The canonical run supplies separate
regression evidence; no result is inferred from that earlier run.

Local lightweight checks additionally passed syntax parsing, the 19 retained archive
hashes and wheel contents. All 21 original archive hashes were checked before the
requested SeedVR Markdown cleanup; removals and the edited summary are recorded in
the archive inventory. The wheel contains the 12 active modules, all three console
entry points, and no legacy files.

Passing supports correct differentiation of the tested finite sample objectives
and the tested engineering contracts. Population-estimator bias/variance and
conditional-coverage diagnostics remain pending E0 work; see the
[experiment plan](experiments.md). Optimization convergence, pretrained perceptual
features and generation quality are not established by these checks. Acceptance
thresholds were not changed after inspecting results.
