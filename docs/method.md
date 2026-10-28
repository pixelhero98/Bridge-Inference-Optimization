# Method and ImageNet adaptation contract

OGO is a generative policy optimization method. Its trainable policy is a
corrective field; it maximizes reward with fixed-reference distributional
regularization. The [experiment plan](experiment-plan.md) extends the scope to
synthetic mechanisms, ImageNet and planned text-conditioned tasks. The concrete
architecture, LPIPS objective and hyperparameters below describe the verified
ImageNet adapters, not a universal reward or an implemented T2I/T2V adapter.

OGO optimizes a corrective field along a fixed backbone. Backbone, decoder and
feature/reward parameters remain frozen; their input Jacobians are retained.
The field is a six-block width-384 RoPE transformer with zero output initially.
Its current-state velocity is multiplied by the Euler step exactly once.

The loss is paired LPIPS plus conditional Euclidean-Laplace kernel matching in
pixel/latent and normalized DINO spaces, with equal branch weights. Generated
self-pairs are excluded; the target-only constant is omitted. Matching draws
are independent of paired reward references. The target distribution is fixed.

## Backbone and stochastic-kernel adaptations

Time increases from noise to data. EDM uses a=(1+80t)/81, b=80(1-t)/81,
x_initial=(80/81)z and v=(D(x/a,b/a,c)-x)/(1-t). SiT uses its native velocity,
CFG 1.5 on the official first three latent channels and VAE scale 0.18215.

For a linear interpolant with a+b=1 and constant a', the ideal score is
s=(a*v/a'-x)/b. Euler-Maruyama uses mean x+(v+g^2*s/2)/N, variance g^2/N,
and g^2=gamma^2*b/a. At SiT's noise endpoint, the diffusion denominator uses
the next grid value, following Flow-GRPO; the score is exactly -x. All trained
transitions have positive variance. Exact marginal preservation requires an
exact interpolant velocity and continuous-time integration; it is not claimed
for finite grids or approximate models.

These are EDM/SiT adaptations, not reproductions of original Stable Diffusion
experiments. DDPO/D3PO DDIM kernels become the agreed Gaussian Euler kernel;
the policy objectives remain intact. Flow-GRPO's sign/time conversion is
checked against its official descending-sigma expression. Mean-coordinate log
probabilities follow the implementations: they produce surrogate ratios, not
full joint-density ratios. Summed Gaussian densities are tested separately.
Deterministic evaluation has no stochastic transition likelihood.

## Optimization rules

| Method | Trainable policy | Update |
|---|---|---|
| OGO | Corrective field | Full trajectory LPIPS and fixed-reference matching gradients |
| DRaFT-K | Rank-8 linear/convolution LoRA | Full forward; last K=4 steps differentiated |
| DDPO-IS | Rank-4 attention LoRA | Batch advantages; clipped surrogate; two replay passes |
| Flow-GRPO | Rank-32 attention LoRA, scale 2 | Group advantages; clipped surrogate; Gaussian reference KL |
| lambda-Controlled GRPO | Rank-32 attention LoRA, scale 2 | LambdaNorm-T and detached damp-only timestep weights |
| Offline D3PO | Rank-4 attention LoRA | Fixed pairs; LPIPS labels; clipped reference ratios; per-transition DPO |

Attention adapters cover fused QKV and output projections. Actual parameter
counts, scopes and scales are recorded. Baselines are not forced into OGO's
corrective field. DRaFT's K is backward depth, separate from inference steps.

Lambda statistics separate mean-square normalized shift (centering) from that
quantity divided by latent dimension (variance). Main-text equations 3.11-3.12
use nu=epsilon^2, epsilon=.1, q=.99 and an explicit numerical floor. At zero
shift the published surrogate equals exp(-nu/2); that behavior is retained.
The damping statistic is averaged over each four-trajectory microbatch before
gradient accumulation over 32 trajectories. This estimator batch size is an
explicit adaptation, separate from the optimizer batch size.

Offline D3PO stores reference likelihoods at collection time. Ties have zero
preference gradient. Timing uses fixed ordering and full-batch accumulation
across every transition before each update.

Sources: [DRaFT](https://arxiv.org/abs/2309.17400),
[DDPO](https://github.com/kvablack/ddpo-pytorch),
[Flow-GRPO](https://github.com/yifan123/flow_grpo),
[lambda-Controlled GRPO](https://arxiv.org/abs/2609.22041),
[D3PO](https://github.com/yk7333/d3po).
