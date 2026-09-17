# Bridge-Inference-Optimization

Finite-NFE corrective fields for frozen generative models.

- [Method summary and joint feature-kernel objective](finite_nfe_corrective_bridge_summary_github_ultrasafe.md)
- [Metrics, reward design, and implementation guide](method_reward_guide.md)
- **[SeedVR: Softplus vs Linear reward parameterization — formulas and completed results](seedvr_reward_parameterization.md)**

## SeedVR update — 2026-09-17

The current reward-term options are **Softplus** and **Linear**. With 200 optimizer
updates and 200 generated training trajectories, Linear achieves higher final
restoration advantage and PSNR than the matched Softplus control, while LPIPS
and feature-kernel scores worsen. Early validation favors Softplus, so this is
not evidence of uniformly faster convergence. GPU time is not matched to the
previous corrector-only Flow-GRPO reference.

The note provides the loss formulas, results, and findings. Softplus attenuates gradients after improvement over the baseline;
it is not a KL penalty or trust-region constraint. A Softplus + Linear hybrid is
described as an **untested candidate**, not a completed SeedVR experiment.
