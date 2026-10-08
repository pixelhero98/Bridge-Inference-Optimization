# FCT experiment plan

Source: the user-supplied FCT experiment-matrix image, reconciled with the two
canonical regimes. Historical README experiment "E" is a separate legacy run.
The completed synthetic diagnostic is E0's gradient-correctness component, not E1.

| ID | Task / backbone | Regime and purpose | Status |
|---|---|---|---|
| E0 | Conditional synthetic distributions; frozen ODE + residual velocity | Verify gradients, estimators, conditional coverage and matching | Original reference and canonical finite-gradient regression complete; estimator/coverage studies pending |
| E1 | Class-conditional images; SiT-XL/2, ImageNet-256 | Reduced-step distribution matching; main estimator, feature and correction-capacity ablations | Planned |
| E2 | Text-to-image alignment; SD3.5-Medium | Same-step utility + matching, equal student/reference budgets | Planned |
| E3 | Text-to-image distillation + utility; SD3.5-Medium | Reduced-step matching followed by joint training | Planned |
| E4 | Text-to-video; Wan2.1-T2V-1.3B, 480p | Transfer to video and measured quality/cost tradeoffs | Planned after image core |
| E5 | Image-to-video; Wan2.2-TI2V-5B, native 720p | Focused conditioning/backbone transfer | Extension |
| E6 | Video restoration; multi-step SeedVR-3B, deterministic sampler | Reconstruction/perception tradeoff and acceleration | Planned after image core |

## Comparisons and readouts

- **E0 (complete):** exact/autograd/adjoint/FD gradients and four omission controls,
  including the installed canonical package. See [verification](verification.md).
- **E0 (pending):** compare estimator bias and variance against a known synthetic
  population reference while varying student/target counts and independent versus
  reused target seeds. Separately compare pooled versus conditional matching on
  conditional mode coverage and population discrepancy. These studies are not
  launched, and finite-sample derivative agreement does not answer their questions.
- **E1:** frozen N and N_T; paired endpoint regression; progressive distillation
  adapted to SiT; optional DMD2-style adaptation. Use ImageNet-256 reference statistics
  and separate validation tuning. Report FID50K, precision/recall, class accuracy and IS.
- **E2:** frozen same-budget generator; utility-only full rollout; matching-only;
  joint FCT; Flow-GRPO / lambda Flow-GRPO comparison. Use full GenEval and disjoint held-out prompts;
  report unoptimized preference, human preference, within-prompt diversity and latency.
- **E3:** frozen N/N_T; AC-DMD, RTDMD, DP-DMD and paired regression comparisons subject
  to a compatible backbone/operator audit. Compare matching-only continuation,
  matching-then-joint, and joint-from-initialization at equal total updates and samples.
  Use E2's evaluation suite plus teacher discrepancy in held-out feature spaces.
- **E4:** frozen N/N_T, paired regression, bidirectional AnyFlow; FastWan as a system
  reference. Full standard VBench prompts; fix resolution, length, FPS and preprocessing.
  Report quality/semantics, temporal consistency, flicker, motion, dynamic degree and latency.
- **E5:** frozen N/N_T, paired regression, matching-only and selected gradient/estimator
  ablations. VBench-I2V with fixed resizing/conditioning; report subject/background
  fidelity, consistency, motion, dynamic degree and latency.
- **E6:** frozen N/N_T, reconstruction-only correction, paired regression, joint FCT;
  SeedVR2 and FlashVSR as system references. Start with 4x SR on paired REDS30/YouHQ40
  and real VideoLQ under a declared degradation protocol. Report PSNR/SSIM, LPIPS/DISTS,
  temporal diagnostics, MUSIQ/DOVER, human assessment and latency.

Method names above come from the supplied matrix; compatibility and reproduction
requirements must be checked before launching each comparison. They are not implementations
provided by this core release.

## Specifications required before each campaign

Freeze the task/backbone revision, normalized-time mapping, N/N_T, training data and
seed streams, conditioning, correction architecture, feature branches (including
latent/perceptual ablations), kernel/bandwidth calibration, utility transform,
optimizer, stage lengths, utility ramp, checkpoint selection, and evaluation protocol.
Calibration uses training data only. Record every choice in the run configuration.

Match updates and generated samples when testing a method effect; also report compute
budgets. Report backbone evaluations, correction evaluations, teacher-training cost,
wall-clock latency, peak memory and parameter count separately. Equal step counts do
not imply equal runtime. Continue matching-only for the same extra stage-two budget
to distinguish utility effects from simply training longer.

The next discussion should finalize E1's task specification and E2/E3's utility and
multi-space matching definitions. No large-backbone campaign is launched by the
canonicalization work, and successful E0 checks do not establish training quality.
