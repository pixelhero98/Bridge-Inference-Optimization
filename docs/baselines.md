# Initial five baselines

The same initial set applies to ImageNet-64, ImageNet-256, T2I and optional T2V.
For T2I, both SD3.5-Medium and SANA-1.6B receive the complete set independently.
The [T2V contract](t2v-protocol.md) fixes the same five at separate 16/32-step
optimization horizons, with 32 steps as the primary video setting.
It preserves the existing ImageNet protocol and enables a consistent cross-task
comparison. This is a selection of optimization mechanisms, not a claim that
these are the five best-performing methods on every benchmark.

| Baseline | Role | Preserve during adaptation |
|---|---|---|
| DRaFT-K | Differentiable reward optimization | Full forward trajectory, last K backward steps, declared adapter scope |
| DDPO-IS | Importance-sampled policy-gradient reference | Advantages, clipped surrogate and replay semantics |
| Flow-GRPO | Group-relative optimization for flows | Group advantages, stochastic flow kernel and reference regularization |
| lambda-Controlled GRPO | Analytic control of ratio instability | Path-variance normalization and detached timestep damping |
| Offline trajectory-DPO based on D3PO | Fixed-data preference optimization | Frozen pairs, labels and reference likelihoods; per-transition preference objective |

[DRaFT-K](https://arxiv.org/abs/2309.17400) is the direct reward-gradient
comparator, not a score-function RL algorithm. [DDPO](https://arxiv.org/abs/2305.13301)
is the established likelihood-ratio comparator. [Flow-GRPO](https://arxiv.org/abs/2505.05470)
and [lambda-Controlled GRPO](https://arxiv.org/abs/2609.22041v2) deliberately
compare a base method with its stabilization extension.
[D3PO](https://arxiv.org/abs/2311.13231) supplies the trajectory-preference
formulation; fixed offline collection and changed kernels are explicit project
adaptations, not full reproductions of the original experimental pipeline.

## Adaptation gates

| Backbone | Required checks |
|---|---|
| EDM-64 | Denoiser-to-velocity/time conversion, pixel stochastic kernel, frozen Heun reference and paired reward |
| SiT-256 | Latent velocity, CFG channels, VAE scale, latent likelihoods, decoded reward and merged-LoRA parity |
| SD3.5 | Prompt/pooled conditioning, rectified-flow time, CFG, attention adapters, VAE reward gradients and positive-variance kernels |
| SANA-1.6B | Gemma conditioning, linear-attention LoRA targets, DC-AE latent/decoder geometry, flow sign/time, uniform Euler parity and stochastic likelihoods |
| Wan | Video latent layout, temporal decoder/reward gradients, conditioning, kernels, memory and trajectory shards |

Current ImageNet settings are in [method.md](method.md) and frozen by
`configs/ogo-profile.json`. New backbones need their own pinned implementation
and loss/gradient checks; SiT verification does not certify SD3.5, SANA or Wan. Use
the same scalar reward and data access within a task. Disclose parameter counts,
replay and deviations from source. Do not force every baseline into OGO's field.

Offline D3PO stores stochastic trajectories and uses the verified clipped
reference-ratio convention. It is not Diffusion-DPO's denoising-objective
preference surrogate. Mean-coordinate Gaussian log probabilities, where used by
a released surrogate, must not be described as full joint-density ratios.

## Deferred alternatives and limitations

- [AlignProp](https://arxiv.org/abs/2310.03739) overlaps with DRaFT-K's
  differentiable reward-backpropagation role. Its omission avoids spending two
  slots on that role; it is not evidence of inferior performance.
- [DiffusionNFT](https://arxiv.org/abs/2509.16117) is a strong later candidate.
  Its forward-process online reinforcement avoids reverse-trajectory likelihoods
  and was evaluated on SD3.5-Medium with a different guidance/training setup.
  It is not silently substituted after the current comparison has begun.
- [Diffusion-DPO](https://arxiv.org/abs/2311.12908) is a useful later offline
  endpoint-data comparator. A flow-matching adaptation requires a documented
  surrogate and gradient tests; renaming D3PO does not implement it.
- [VADER](https://arxiv.org/abs/2407.08737),
  [DanceGRPO](https://arxiv.org/abs/2505.07818) and
  [VideoDPO](https://arxiv.org/abs/2412.14167) are video-specific candidates for
  reward gradients, video-tested GRPO and weighted preferences. They are not
  extra members of the initial five.

Flow-GRPO and lambda-Controlled GRPO are related, and this first matrix omits
forward-process methods such as DiffusionNFT. Report comparisons against the
tested adaptations rather than state-of-the-art claims across all algorithms.
Any later substitution requires a new protocol version for the complete task
matrix, selected before its evaluation outcomes are inspected.
