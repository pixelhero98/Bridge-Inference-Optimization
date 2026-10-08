# Canonical finite objective

The state belongs to the backbone's state space; context can contain a class,
prompt, image, observation, or other task conditioning. The frozen adapter returns
velocity in normalized generative time [0,1]. Set h=1/N and t_k=k/N. The only active
operator is x_next=x+h*(v0(x,t,c)+c_phi(x,t,c)). Zero correction reproduces the
frozen N-step sampler. Reduced-step inference still evaluates the frozen backbone.

For one context, x_i is a corrected endpoint from independent student noise z_i.
Targets y_m are frozen teacher endpoints from independent noises eta_m conditional
on that same context. The teacher uses N_T uniform Euler steps. Observed conditional
data may replace teacher endpoints in a declared task adapter, independently of z_i.

\[
S_r=\frac{1}{2K(K-1)}\sum_{i\ne j}
k_r(\Psi_r(x_i),\Psi_r(x_j))
-\frac{1}{KM}\sum_{i,m}k_r(\Psi_r(x_i),\Psi_r(y_m)),
\qquad
L_c=a_u\frac1K\sum_i U(x_i;c)+\lambda\sum_r\beta_rS_r.
\]

Average L_c equally across contexts; never pool pairs across contexts. Use K>=2
when matching is active and M>=1 targets. Differentiate both generated kernel
arguments. The generated ensemble factor occurs exactly once; gradient contributions
are summed after the loss's reductions. Target-only constants are omitted, so the
score need not be positive. No weight is normalized implicitly.

U is a loss to minimize. A reward can use U=-R, or an explicitly specified smooth
transform. A same-noise frozen utility baseline may be part of U, but is a separate
input from the matching target. Nonlinear paired utility can depend on the coupling;
it must not be described as purely a function of the student marginal distribution.

## Multiple spaces

Each feature branch supplies a fixed differentiable map, a symmetric kernel and a
fixed weight. For a latent generator, examples include
Psi_latent(x)=flatten(x) and Psi_DINO(x)=DINO(preprocess(decode(x))). Branches can
have different feature dimensions and bandwidths. Generated decode/feature operations
retain their input derivatives even though all associated parameters are frozen.
Target features are computed without gradients. Multi-space feature matching does
not guarantee equality of the original output distribution if the maps discard information.

The provided smooth Gaussian kernel uses exp(-||u-v||^2/(2*sigma^2)). Other symmetric
kernels can implement the callable interface and require their own smoothness and
gradient checks. Do not silently replace a distance, add a floor, or learn bandwidths.
Kernel self-pairs are excluded before kernel evaluation. Distinct samples can still
coincide; nonsmooth kernels need a separately documented derivative convention.

## Staging and pseudocode

Same-step: N=N_T, one joint stage. Reduced-step: N<N_T, one matching-only stage
followed by one joint stage. Both are declared in the original configuration.
Matching weight remains fixed and positive. An optional predeclared utility ramp
uses local stage updates 1..ramp_updates and multiplies the utility loss externally.
There is no online weight/bandwidth calibration in the core.

```python
freeze(backbone, teacher, decoder, feature_encoders, utility_models)
initialize_correction_with_zero_output()
optimizer = AdamW(correction.parameters(), task_spec.optimizer)
for update in declared_schedule:
    optimizer.zero_grad()
    a = schedule.utility_weight(update)
    for context in batch_contexts:
        z = independent_draws(role="student", context=context)
        eta = independent_draws(role="matching-teacher", context=context)
        y = frozen_uniform_euler(teacher, eta, N_T)
        baseline = optional_frozen_utility_reference(z, context)
        x = uniform_euler(backbone, correction, z, context, N)
        loss = a * mean(utility(x, context, baseline))
        loss += matching_weight * sum(
            branch.weight * conditional_kernel_score(branch(x), branch(y))
            for branch in feature_branches
        )
        (loss / number_of_contexts).backward()
    optimizer.step()
    checkpoint_at_declared_boundaries()
```

The implementation skips utility evaluation when a=0. Checkpoints preserve model,
AdamW state, all framework RNG streams, adapter sampler state, cursor, update count,
and stage schedule. Explicit resume restores the last committed update; failed
attempt logs remain as evidence. Legacy displacement checkpoints are incompatible.

For a backbone defined in native time s=g(t), its adapter must implement the
corresponding normalized-time velocity, including the derivative ds/dt, and verify
against the declared frozen sampler. Nonuniform native grids must not be silently
substituted for this uniform normalized-time operator.
