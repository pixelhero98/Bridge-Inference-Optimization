# Initial T2V comparison: finalized methods and horizons

Decision: 2026-10-28. Retain Wan2.1-T2V-1.3B as the planned backbone and the
agreed common five baselines. Video remains an optional, separately budgeted
extension. This decision freezes comparison identities and sampling horizons;
it does not certify the unimplemented Wan adapters or authorize a training run.

## Exactly five baselines

| Baseline | Role in the video comparison | Required adaptation |
|---|---|---|
| DRaFT-K | Direct differentiable-reward optimization | Video decoder/reward input gradients; full forward trajectory with last K=4 steps differentiated |
| DDPO-IS | Established online importance-sampled policy gradient | Wan stochastic Euler kernel, frozen-policy replay and declared likelihood convention |
| Flow-GRPO | Flow-model group-relative optimization | Prompt groups, flow time/sign conversion and native reference regularization |
| lambda-Controlled GRPO | Analytic stabilization of Flow-GRPO ratios | Path-variance law for the actual video latent dimension/kernel and detached damping |
| Offline trajectory-DPO based on D3PO | Fixed-data preference optimization | Immutable video pairs, labels and frozen transition likelihoods in CPU-resident shards |

OGO is the sixth trained method and keeps its corrective velocity field. Each
baseline keeps its own optimization rule and trainable parameterization. These
are Wan adaptations, not claims that the original papers verified these exact
video settings. The two GRPO variants deliberately test stabilization; this set
does not claim complete coverage of current video-alignment methods.

[VADER](https://arxiv.org/abs/2407.08737) is a useful video reward-gradient
reference, and [VideoDPO](https://arxiv.org/abs/2412.14167) is a distinct weighted
video-preference method. Neither is simply a new name for DRaFT-K or our offline
D3PO implementation. DanceGRPO and DiffusionNFT remain later candidates under
the [baseline selection policy](baselines.md), not additions to the five.

## Optimize separately for 16 and 32 inference steps

| Setting | Role | Policies |
|---|---|---|
| 32 uniform Euler steps | Primary video-quality comparison | Train all six methods specifically for N=32 |
| 16 uniform Euler steps | Lower-cost deployment comparison | Train all six methods separately for N=16 |

There are **12 trained configurations per training seed**, plus frozen Wan
controls at both horizons. No 64-step optimized arm is included. Do not train
only at 32 and present 16-step resampling of that checkpoint as the separately
optimized 16-step result. Euler steps are distinct from optimizer updates and
from the differentiated depth K.

- OGO and DRaFT-K execute N deterministic Euler forward steps during training.
  OGO differentiates the full trajectory; DRaFT-K differentiates the final four.
- DDPO-IS, Flow-GRPO and lambda-Controlled GRPO collect N positive-variance
  stochastic Euler transitions. Their likelihoods refer to that training kernel.
  All three deploy deterministically at the same N.
- Offline D3PO collects a separate frozen-initial-policy preference dataset for
  each N, then replays those transitions. It does not collect a fresh rollout
  for every optimizer update.

The primary protocol has no hidden training-step reduction or method-specific
timestep subsampling. Use a uniform grid in the explicitly defined normalized
integration time. Do not silently inherit a shifted native scheduler and label
its nonuniform model-time grid uniform Euler. Pin the Wan time/sign conversion,
guidance and endpoint treatment; count actual network evaluations, including
CFG and recomputation. Equal Euler steps do not imply equal wall-clock cost.

## Acceptance before execution

Evaluate the frozen model at both Euler horizons on a fixed, representative
development prompt set before training. Record visual quality, prompt alignment,
motion/dynamic degree, temporal consistency, diversity and latency. Do not screen
out difficult prompts using rewards. A recent [Wan velocity-matching study](https://arxiv.org/html/2608.23664v1)
uses a shifted DPM-2 schedule for its 16-step ODE runs; those results do not
validate this study's 16-step uniform-Euler policy.

Fix the same resolution, frame count, FPS, duration, prompt cases, reward and
evaluation protocol across all methods and both horizons. Require a tested
differentiable video-level reward for OGO and DRaFT-K; use the same scalar for
online rewards and offline preference labels. Frozen high-step reference pools
are separate assets and are not extra optimized baseline configurations.

Profile all 12 configurations, including reference preparation, offline data,
decoder/reward backward work, replay, audit metrics and storage. Freeze objective
and gradient parity tests, numerical thresholds, manifests, optimizer settings,
training updates and the resource ceiling before launch. If a horizon or method
fails feasibility, report it and revise the protocol explicitly; do not silently
drop it or substitute another sampler. The existing ImageNet budget is unchanged.
