# Finite-time corrective transport

This repository implements the finite, uniform-Euler objective for a frozen
generative backbone with a trainable **velocity correction**:

\[
x_{k+1}=x_k+\frac{1}{N}\left[v_0(x_k,k/N;c)+c_\phi(x_k,k/N,c)\right].
\]

Both velocities are evaluated at the current generated state. Training differentiates
the full rollout, including the frozen backbone's derivatives with respect to its
input. Inference uses exactly the same operator. Only correction parameters train.

Two training regimes are supported:

- **Same step:** student and teacher use equal budgets; train utility plus matching.
- **Reduced step:** use fewer student steps; train matching first, then add utility
  while retaining matching, teacher identity, optimizer moments, RNG, and data cursor.

Matching is a sum of conditional kernel scores in fixed feature spaces. It can
combine latent features with decoded-image perceptual features such as DINO, each
with an explicit weight, kernel, bandwidth, and preprocessing specification.
The core is not restricted to latent space. Large-backbone and DINO integrations
are adapter work; no pretrained weights are included or downloaded automatically.

## Install and run

Use Python 3.10+ and a platform-compatible PyTorch installation. For reports:

```bash
python -m pip install '.[report]'
python -m fct.train --config configs/synthetic-same-step.json --output runs/same-step
python -m fct.train --config configs/synthetic-reduced-step.json --output runs/reduced-step --stop-after 3
python -m fct.train --config configs/synthetic-reduced-step.json --output runs/reduced-step --resume
python -m unittest discover -s tests -v
python -m fct.verify --config configs/gradient-correctness.json --output runs/e0 --devices cpu cuda
python -m fct.audit_evidence --run runs/e0 --source /path/to/installed/fct
python -m fct.report --run runs/e0 --figures runs/e0-figures
```

Numerical workloads should run on allocated compute resources. Use `--devices cpu`
for CPU-only verification; the complete declared acceptance run includes CUDA.
The synthetic training configurations are smoke tests, not tuned experiment settings.
New runs reject existing output directories. `--resume` requires an identical declared
configuration, source, and adapter identity and trusted locally generated checkpoints.
`--stop-after` specifies an absolute cumulative update boundary.

## Method and extension points

Read [the formulation and pseudocode](docs/method.md),
[the adapter contract](docs/adapters.md), [equation-to-code audit](docs/equation-to-code.md),
and [experiment plan](docs/experiments.md).

- `src/fct/`: canonical sampler, objectives, staged trainer, and verification tools.
- `configs/`: immutable verification protocol and small training examples.
- `tests/`: structural, gradient, multi-space, and resume checks.
- `legacy/`: archived pre-canonical files with an inventory and source commit.

Superseded post-step residual and standalone-student implementations are not valid
realizations of this canonical method. They are not installed, imported, or tested
as active implementations. Their checkpoints are rejected rather than converted.

Both the original E0 finite-gradient reference and the installed canonical package
passed their declared checks. The canonical run passed 21 structural tests, 336
scalar/nonlinear fixtures, 36 CPU/GPU comparisons, and all four omission controls;
see [verification records](docs/verification.md) for evidence and scope. Passing these tests supports differentiation of
the tested finite objectives; it does not establish population unbiasedness,
optimization convergence, or generation quality.
