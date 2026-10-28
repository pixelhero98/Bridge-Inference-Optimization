# OGO: generative policy optimization

OGO optimizes a generative policy by learning a corrective velocity field along
a frozen pretrained backbone. It maximizes task reward while regularizing the
generated distribution toward a fixed reference distribution. Backbone, decoder
and reward/feature parameters stay frozen; their input derivatives are retained.

$$
x_{k+1}=x_k+\frac{1}{N}\big[v_0(x_k,k/N;c)+u_\phi(x_k,k/N,c)\big].
$$

Every evaluated policy uses **deterministic uniform Euler at 16 or 32 steps**.
The high-step sampler defining a reference distribution is a separate frozen
asset, not an optimized baseline.

## Experiment scope

| Task | Backbone | Scope | Implementation |
|---|---|---|---|
| Synthetic | Analytic distributions and small frozen dynamics | Core mechanism tests | Finite-gradient fixtures retained; broader optimization study planned |
| ImageNet-64 | EDM-64 | Core | Profiling adapters available; production pending |
| ImageNet-256 | SiT-XL/2 | Core, first production campaign | Ten baseline runs: seed 0, 512 updates, FID-10K; completion requires audited artifacts |
| Text-to-image | SD3.5-Medium | Core, after ImageNet | Planned; adapters and resource budget unverified |
| Text-to-video | Wan2.1-T2V-1.3B | Optional extension | Planned; feasibility and separate budget required |

Each non-synthetic task starts with exactly **five baselines**: **DRaFT-K,
DDPO-IS, Flow-GRPO, lambda-Controlled GRPO, and offline trajectory-DPO based on
D3PO**. OGO is the proposed method; frozen-backbone controls are untrained.
There are 12 trained method/horizon configurations per task and seed, 36 for
the three core image tasks, or 48 with video. The current ImageNet-256 campaign
contains only its ten baseline configurations.

The [experiment plan](docs/experiment-plan.md) defines rewards, evaluation,
adaptation gates and execution order. The [baseline rationale](docs/baselines.md)
explains the common set and later candidates.
[configs/experiment-plan.json](configs/experiment-plan.json) is a planning
catalog, **not an executable campaign configuration**.

## Install and inspect

```bash
pip install -e '.[imagenet]'
```

Repository checks need only Python 3.11+ standard-library modules:

```bash
python tools/check_repository.py
python -B -m unittest discover -s tests -p test_metadata_records.py -v
```

On an allocated compute node, retained synthetic fixtures run with:

```bash
ogo-train --config configs/synthetic-same-step.json --output runs/synthetic
```

These short four/eight-step examples check correctness; they are not the
16/32-step scientific comparison. ImageNet profiling and production are separate:

```bash
ogo-study verify --config configs/ogo-profile.json --assets ASSETS.json --output PROFILE
ogo-study profile --config configs/ogo-profile.json --assets ASSETS.json --output PROFILE --deadline UNIX_TIME
ogo-campaign verify --config configs/imagenet256-baselines-512.json --assets ASSETS.json --output CAMPAIGN --deadline UNIX_TIME
```

See the [profiling protocol](docs/profiling.md), [production campaign](docs/imagenet256-campaign.md)
and [asset contract](docs/assets.md) before numerical execution. Full ImageNet
commands require explicit pinned assets, source repositories, reference caches,
an evaluator environment and an allocation. They do not download missing assets.
T2I/T2V adapters are not implemented by these commands.

## Source and evidence

- `src/ogo/`: corrective policy, objectives, numerical fixtures, ImageNet adapters,
  profiling and the separate production runner.
- `configs/`: numerical/campaign contracts and the planning catalog.
- `experiments/e0/`: historical numerical evidence with original identities.
- `tests/`: regression checks; `tools/`: lightweight source/plan validation.

The [method contract](docs/method.md) separates reward, matching and evaluation.
The [verification policy](docs/verification.md) distinguishes historical receipts
from checks of this source. See the [cleanup record](docs/cleanup.md) for retired
experiments and the metadata-resume repair.

Deployed releases remain immutable. A new package cannot replace an old run's
source: resume binds source, configuration, assets and reference identities.
No completed ImageNet quality comparison, T2I/T2V implementation or general
convergence result is claimed here.
