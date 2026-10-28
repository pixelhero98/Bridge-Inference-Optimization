# OGO experiment plan

Protocol decision: 2026-10-28. OGO is generative policy optimization through a
corrective velocity field. The central question is whether it improves terminal
reward while retaining distributional coverage at a fixed inference budget.
Shorter sampling is a deployment constraint, not a distillation experiment.

## Scope and matrix

| Task | Backbone | Comparison | Deployment | Status |
|---|---|---|---|---|
| Synthetic | Analytic/low-dimensional frozen dynamics | Mechanism controls; no five-baseline requirement | Diagnostic grids | Existing gradient fixtures; optimization study planned |
| ImageNet-64 | EDM-64 | OGO + common five | Uniform Euler 16, 32 | Profiled; production pending |
| ImageNet-256 | SiT-XL/2 | OGO + common five | Uniform Euler 16, 32 | Ten-baseline campaign contract frozen |
| T2I | SD3.5-Medium, initially 512 x 512 | OGO + common five, task adapted | Uniform Euler 16, 32 | Core planned task |
| T2V | Wan2.1-T2V-1.3B, initially 480p | OGO + common five, task adapted | Uniform Euler 16, 32 | Optional; launch gate closed |

The five are **DRaFT-K, DDPO-IS, Flow-GRPO, lambda-Controlled GRPO, and offline
trajectory-DPO based on D3PO**. Each non-synthetic task receives all five.
Backbone controls and OGO ablations are separate from the five. Adaptation changes
the backbone interface, decoder, conditioning, stochastic kernel and reward,
without silently replacing a baseline objective. See [baseline rationale](baselines.md).

There are 12 main trained configurations per task and training seed: 36 across
the three core image tasks, or 48 with T2V. Counts exclude controls, ablations,
replications and additional reward campaigns. They are scope counts, not compute
allocations. No 64-step deployment arm, I2V, standalone one-step generator or
distillation campaign is active.

## Objective and fairness

The planning objective is

$$
\max_\phi\;\mathbb{E}[r_c(X_N^\phi)]
-\beta\,\mathbb{E}_c[D(q_{\phi,N}(\cdot\mid c),q_{\mathrm{ref}}(\cdot\mid c))].
$$

Freeze the reference law and feature/kernel calibration before training.
Matching draws are independent of generated samples and paired reward references,
conditional on the same class or prompt. A soft discrepancy penalty does not
guarantee a hard drift bound. Constrained synthetic variants require explicit
constraint checks and are labeled separately.

All methods start from the same pinned backbone, use the same task reward and
fixed train/development/test identities, and train separately for each horizon.
Keep native update rules and parameterizations; OGO retains its corrective field.
Freeze each new task's optimizer, adapter rank/scope, precision, guidance,
normalization, time map and clipping before scientific runs. Transfer of a SiT
learning rate to another architecture requires a declared calibration protocol.

Uniform Euler uses a uniform grid in the declared integration variable; disclose
any nonlinear time/sigma mapping. Within each task all methods share deployment
grid, guidance, resolution and conditioning. Use positive-variance stochastic
transitions where training likelihoods require them. Verify density conventions,
endpoint handling and reference likelihoods. Deterministic Euler has no Gaussian
transition likelihood; finite-grid ODE/SDE equivalence is not assumed.

Report performance versus updates, fresh trajectories, reward queries and total
compute. Count rollouts, replay, offline collection, references, reward/features,
warmups, failures, VAE decode and final metrics. Report actual network evaluations,
parameter counts, peak memory and deployment latency. Equal updates or Euler
steps do not imply equal compute. Do not add OGO's matching penalty to every
baseline in an attempt to make their objectives identical.

## Synthetic: mechanism and limits

Use analytic Gaussians and nonlinear conditional mixtures. Include well/ill-
conditioned and rank-deficient parameterizations; aligned, conflicting and
orthogonal rewards; and conditional mode coverage. Main references are stationary.
Moving-reference tracking is a separate diagnostic, not stationary-energy descent.

Controls are frozen dynamics, utility-only correction, matching-only correction,
joint OGO, scalar learning-rate control, and a small exact/high-accuracy oracle.
Where tractable include damped Gauss-Newton to distinguish step size from update
geometry. These controls are not five neural-generator baselines.

Measure reward, exact Gaussian W2 where available, fixed-energy changes,
feature/population discrepancy, conditional coverage and compute. For output
Jacobian J, prescribed movement d and parameter direction p, separate
||(I-JJ^dagger)d|| from min_{alpha>=0}||d-alpha Jp||. Check finite differences,
nonlinear remainders and a Gaussian rotation control: particle movement is not
distributional movement. Only affine cases support global representability claims.

Retain E0's existing exact-gradient/finite-difference fixtures and thresholds as
implementation evidence. New optimization seeds, audit samples, effect-size
criteria and budgets need a separate frozen executable protocol. Do not revive
retired amortized-WGF or normalized-Drifting campaigns under this label.

## ImageNet: finish the existing protocol

Both resolutions use all 1,000 classes, balanced manifests and sampling fixed
without reward screening. Reward is paired negative LPIPS against a pinned
high-step reference with the same noise and class. OGO additionally uses
independent fixed-reference matching draws. This tests reference-fidelity
optimization, not general human-preference alignment. EDM Heun and SiT Euler
high-step references remain distinct from evaluated uniform-Euler policies.

The [ImageNet-256 contract](imagenet256-campaign.md) remains unchanged: seed 0,
ten baseline/horizon runs, 512 updates, batch 32, development 2K at 128/256/384,
final 10K at 512, and two frozen controls. FID uses 10,000 generated images and
the official 50,000-image validation reference. Also report paired LPIPS,
top-1/top-5 accuracy and within-class DINO diversity. No checkpoint selection;
label it single-seed FID-10K.

OGO-256 and the 12-case ImageNet-64 study follow baseline audit and their own
preflight. The initial target is also 512 updates, but reference counts, OGO
matching draws, calibration and evaluators require new campaign configs. The
current ten-run config cannot launch OGO or EDM. Multi-seed claims require
separately budgeted training replications.

## T2I: optimization beyond reference fidelity

Use [SD3.5-Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
at an initial fixed 512 x 512 resolution. Primary utility is a frozen,
differentiable PickScore scalar for the image/prompt pair. Pin its model,
preprocessing and scalar convention; calibrate affine normalization on
training-only reference outputs. Use the same scalar for online rewards and
offline labels. Backpropagation methods require input gradients through the
reward and decoder; their weights and the prompt encoders stay frozen.

Freeze a prompt manifest stratified by objects, attributes, relations and
styles. Deduplicate prompts across train/development/test and exclude benchmark
test prompts from training. Match references with independent noise for the
same prompt; unrelated prompts are not interchangeable conditional targets.

Evaluate full GenEval, disjoint held-out prompts, optimized reward, independent
preference/semantic scorers and within-prompt diversity. Report prompt-clustered
uncertainty; multiple samples from one prompt are not independent prompts.
Blind human preference is separately budgeted. OCR or other nondifferentiable
rewards can be held-out metrics; a surrogate gradient changes the training task.

Profile all 12 combinations before fixing update counts, prompt counts and
evaluation multiplicities. Use 512 updates as a pilot planning target, not a
convergence assertion. T2I needs a separate budget; ImageNet's 110-hour allowance
does not extend to it. The shared five are the initial comparison, not a claim
to cover every current alignment algorithm.

## T2V: optional and budget gated

Start with [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B).
Freeze width/height, frame count, FPS, duration, latent layout, decoder chunking,
time map and guidance together. Initial 480p is a planning target, not a verified
memory claim. Require a differentiable video-level preference/quality reward
with tested temporal input gradients. Frame-only rewards cannot establish
motion optimization.

Adapt the common five and OGO. Offline pairs share the prompt and declared
initial-noise coupling, with independent stochastic transition noise. Preserve
pair IDs, labels and reference likelihoods; count trajectory/video storage.
Validate gradient checkpointing and decoder gradients before choosing batch size.

Use the full applicable VBench protocol, held-out rewards/prompts, temporal
consistency, motion/dynamic degree, flicker and within-prompt diversity. Include
temporal features in reference matching before claiming temporal coverage.
Identify optimized reward, matching features and audit metrics separately.

Activate video only after all six methods pass adaptation checks and measured
training/reference/evaluation/storage costs fit a separately approved allocation.
If a baseline is infeasible, defer video and document why. Do not silently omit
it, shorten clips for only one method, or rename a replacement as that baseline.

## Execution order and evidence

1. Preserve and finish the existing ImageNet-256 baseline protocol and audits.
2. Finalize synthetic mechanisms and validate/price OGO-256 and ImageNet-64.
3. Implement, validate and profile T2I; freeze a new executable protocol and cap.
4. Consider T2V after the image core and its separate feasibility gate.

Freeze source/config/model hashes, seeds and role manifests, rewards/features,
hyperparameters, exposure, endpoint, evaluator versions, thresholds and limits
before each campaign. Verify zero-correction Euler parity, reward gradients,
likelihoods, native objective/gradient behavior, reference isolation, optimizer/
RNG replay and fresh-process resume. Use independent audit samples. Budget OGO
utility-only/matching-only/joint ablations separately from the five baselines.

Publish per-case metrics, hashes, terminal checkpoints, failures and allocation
accounting. Plot reward versus discrepancy, quality versus updates and compute,
coverage/diversity and deployment cost. Classify hypotheses as supported,
contradicted or inconclusive within their tested scope. Pending adapters are not
implemented methods; synthetic results do not establish global neural convergence.
