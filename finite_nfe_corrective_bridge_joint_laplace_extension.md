# Finite-NFE Corrective Bridge: Method Summary and Joint-Laplace Extension

**Scope.** Sections 1–9 retain the original method and its reported empirical findings. Sections 10–15 specify the proposed **single-objective extension**; no experimental results for that extension are available in the supplied material. Section 4 refines the terminal adjoint and distinguishes exact discrete differentiation from a first-order perturbation approximation. The original source file is unchanged.

## 1. Core idea

We start from a frozen generative bridge / probability-flow field:

```math
\dot{x}_t = v_0(x_t,t;c).
```

We keep the deployed inference procedure fixed: **uniform-grid Euler with a prescribed NFE** $`N`$.

We learn an NFE- and context-conditioned residual field:

```math
c_\phi(x_t,t,c,N).
```

The deployed dynamics are

```math
x_{k+1}
=
x_k
+
h\left[
v_0(x_k,t_k;c)
+
c_\phi(x_k,t_k,c,N)
\right],
\qquad
h=\frac{1}{N},
\qquad
t_k=\frac{k}{N}.
```

The final method uses only this **unrestricted corrective field**. We do not learn a clock, a solver, or a separate translation-only model.

The central empirical finding is that the correction is **not primarily compensating numerical discretization error**. It learns a task-optimal transport for the downstream objective. Most of the measured gain can be explained by conditional mean displacement, while the remaining gain comes from deformation of the centered distribution, which trades diversity / coverage for additional fidelity.

---

## 2. Paired terminal objective

For the same context $`c`$, target $`y`$, initial noise $`z`$, and NFE $`N`$, define

```math
X_0 = F_0^N(z,c)
```

as the frozen-bridge endpoint and

```math
X_\phi = F_\phi^N(z,c)
```

as the corrected endpoint.

For a lower-is-better terminal task loss $`\ell`$, define the paired excess loss

```math
\Delta_\phi
=
\ell(X_\phi,y)
-
{\rm sg}\left[\ell(X_0,y)\right].
```

Here ${\rm sg}[\cdot]$ denotes stop-gradient.

To avoid introducing an additional special-function macro, define the smooth paired penalty explicitly:

```math
s_\tau(r)
=
\tau
\log
\left(
1+
\exp
\left[
\frac{r+m}{\tau}
\right]
\right).
```

The paired objective is

```math
\mathcal{L}_{\rm pair}(\phi)
=
\mathbb{E}_{c,y,z,N}
\left[
s_\tau(\Delta_\phi)
\right].
```

Here $`\tau`$ is a temperature and $`m`$ is an optional improvement margin.

The frozen endpoint acts as a **same-noise counterfactual baseline**. Empirically, paired training:

- learns substantially smaller corrections than direct terminal optimization;
- remains better after lowering the learning rate of direct training;
- generalizes better across unseen NFEs and contexts.

This indicates an **objective-level effect**, not merely an effective step-size difference.

---

## 3. Discrete-then-optimize

The key modeling choice is the order of operations.

We do **not** first optimize an ideal continuous-time bridge and then ask how well a numerical solver approximates it.

For each target NFE $`N`$, we first define the actual deployed discrete operator:

```math
F_\phi^N
=
G_{\phi,N-1}
\circ
\cdots
\circ
G_{\phi,0},
```

where

```math
G_{\phi,k}(x)
=
x
+
h\left[
v_0(x,t_k;c)
+
c_\phi(x,t_k,c,N)
\right].
```

We then optimize this realized finite-time operator directly:

```math
\phi^\star
=
\arg\min_\phi
\mathcal{L}_{\rm pair}
\left(
F_\phi^N,
F_0^N
\right).
```

Hence the method is:

> **discretize first, then optimize the deployed finite-NFE transport.**

This differs from:

- timestep / clock optimization;
- solver design;
- distillation toward a dense numerical trajectory;
- pure numerical-error compensation.

Dense-rollout ablations show that most of the improvement remains when the corrected field is integrated densely. Therefore the learned correction mainly changes the **task-optimal transport itself**, rather than merely repairing Euler truncation error.

---

## 4. Implicit flow-map / finite-time operator model

The correction is parameterized locally at the velocity / score level, but supervision is applied through the completed rollout.

A concise description is:

> **field-level parameterization + flow-map-level objective**

The residual field implicitly parameterizes the finite-time map

```math
F_\phi^N : z \mapsto X_\phi.
```

A local correction at step $`k`$ matters through its propagated effect on the terminal objective. For one paired example, the terminal adjoint must differentiate the **actual paired penalty**, not just the underlying task loss:

```math
a_N
=
\nabla_{x_N}s_\tau(\Delta_\phi)
=
\frac{1}{1+\exp[-(\Delta_\phi+m)/\tau]}
\nabla_{x_N}\ell(x_N,y).
```

Write $`b_\phi=v_0+c_\phi`$. With full differentiation through the Euler rollout, the discrete backward recursion is **exact**, wherever the derivatives exist:

```math
a_k
=
\left[
I+hJ_xb_\phi(x_k,t_k;c,N)
\right]^\top a_{k+1},
```

and the parameter gradient is

```math
\nabla_\phi s_\tau(\Delta_\phi)
=
\sum_{k=0}^{N-1}
h
\left(
\frac{\partial c_\phi(x_k,t_k,c,N)}{\partial\phi}
\right)^\top a_{k+1}.
```

The partial parameter derivatives above hold the current state fixed. For a batch or ensemble objective, initialize each terminal adjoint from that complete scalar objective, including its reductions and cross-sample dependencies. The same recursion then applies to each rollout.

Freezing the parameters of $`v_0`$ does **not** remove its state Jacobian. A `no_grad` block is appropriate for the separate frozen-baseline rollout, but not for evaluating $`v_0`$ inside the differentiable corrected rollout.

Separately, a first-order endpoint perturbation about the frozen trajectory $`x_k^0`$ is

```math
F_\phi^N(z,c)-F_0^N(z,c)
\approx
\sum_{k=0}^{N-1}
h
J^0_{k+1\to N}
c_\phi(x_k^0,t_k,c,N).
```

Here $`J^0_{k+1\to N}`$ is the Jacobian of the remaining frozen Euler steps. This is an approximation in correction amplitude, unlike the discrete adjoint above. A small local correction can produce a large terminal effect when downstream sensitivity is high.

---

## 5. Relation to a Doob $`h`$-transform: interpretation, not equivalence

There is a useful conceptual connection to terminally conditioned stochastic control.

For a baseline stochastic path law $`P_0`$ and terminal reward $`R(X_T)`$, consider

```math
\max_Q
\left\{
\mathbb{E}_Q[R(X_T)]
-
\frac{1}{\beta}
D_{\rm KL}(Q\|P_0)
\right\}.
```

The corresponding tilted path law has the form

```math
dQ^\star
\propto
e^{\beta R(X_T)}
dP_0.
```

The Doob transform uses the desirability function

```math
h(x,t)
=
\mathbb{E}_{P_0}
\left[
e^{\beta R(X_T)}
\mid
X_t=x
\right].
```

The local controlled dynamics therefore depend on the downstream desirability of future terminal states.

Our learned correction has a related interpretation: every velocity correction is trained through terminal loss propagated over all subsequent dynamics. It therefore internalizes both:

1. the state produced by preceding velocity decisions; and
2. the downstream consequence of the current intervention for terminal utility.

However, the present method is **not formally a Doob $`h`$-transform**. We optimize a deterministic finite-NFE Euler operator with a parametric paired terminal objective, without explicitly constructing a KL-regularized tilted stochastic path law.

The safe interpretation is:

> **Doob-style downstream credit assignment / terminally conditioned control view**

rather than exact $`h`$-transform equivalence.

---

## 6. Conditional mean displacement vs distribution deformation

To identify how the unrestricted corrective transport improves the task, we use a post-hoc counterfactual diagnostic.

For a fixed context and NFE:

```math
X_0 = F_0^N(z,c),
\qquad
X_\phi = F_\phi^N(z,c).
```

Define the conditional means

```math
\mu_0
=
\mathbb{E}_z[X_0],
\qquad
\mu_\phi
=
\mathbb{E}_z[X_\phi].
```

Write the two endpoint distributions as

```math
X_0
=
\mu_0+A_0,
\qquad
X_\phi
=
\mu_\phi+A_\phi,
```

where $`A_0`$ and $`A_\phi`$ are centered anomalies.

We construct the counterfactual endpoint

```math
X_{\rm mean}
=
X_0
+
(\mu_\phi-\mu_0)
=
\mu_\phi+A_0.
```

This is **not another trained model**. It combines:

- the conditional mean learned by the full corrective transport; and
- the centered stochastic geometry of the frozen bridge.

Because

```math
X_{\rm mean}-\mu_\phi
=
X_0-\mu_0,
```

pairwise differences are preserved exactly:

```math
X_{\rm mean}^{(i)}
-
X_{\rm mean}^{(j)}
=
X_0^{(i)}
-
X_0^{(j)}.
```

Therefore covariance, effective rank, pairwise diversity, and every translation-invariant diversity statistic are preserved exactly in the modeled state space.

Estimating $`\mu_\phi-\mu_0`$ requires both frozen and corrected ensembles, so this is primarily a **mechanism diagnostic**, not the preferred deployment method.

---

## 7. Current held-out evidence: original objective

Normalized composite $`(\mathrm{CRPS}+\mathrm{MAE}+\mathrm{RMSE})/3`$, lower is better:

| NFE regime | Base | Trained translation | Full transport | Full mean + base shape* |
|---|---:|---:|---:|---:|
| Seen: 2/4/6/8 | 0.67312 | 0.63896 (5.07%) | **0.41157 (38.86%)** | 0.47086 (30.05%) |
| Interpolation: 3/5/7 | 0.63209 | 0.60327 (4.56%) | **0.43351 (31.42%)** | 0.48599 (23.11%) |
| Extrapolation: 12/16 | 0.56934 | 0.55226 (3.00%) | **0.43380 (23.81%)** | 0.45745 (19.65%) |

\*Counterfactual diagnostic, not a trained model.

The counterfactual mean displacement recovers approximately:

- **77%** of the full gain at seen NFEs;
- **74%** under interpolation;
- **83%** under extrapolation;

while preserving the frozen bridge's centered distribution exactly.

The explicitly trained translation-only model is much weaker. Therefore the final method does not need a separate translation model: the unrestricted corrective field learns a substantially better task-directed conditional mean displacement, and the post-hoc mean counterfactual is sufficient to identify its contribution.

---

## 8. Fidelity-diversity mechanism

The empirical mechanism can be summarized as

```math
X_\phi
=
\mu_\phi+A_\phi.
```

There are two effects:

```math
\mu_0
\longrightarrow
\mu_\phi
```

for conditional mean displacement, and

```math
A_0
\longrightarrow
A_\phi
```

for centered distribution deformation.

The counterfactual

```math
X_{\rm mean}
=
\mu_\phi+A_0
```

shows that a large fraction of task improvement can be retained **without changing the centered distribution**.

The unrestricted full transport

```math
X_\phi
=
\mu_\phi+A_\phi
```

improves fidelity further by allowing

```math
A_\phi \neq A_0.
```

Empirically, this extra freedom is associated with:

- lower pairwise diversity;
- lower covariance / stochastic spread;
- worse coverage;
- reduced effective rank in aggressive settings.

Thus part of the additional fidelity obtained by unrestricted correction is purchased through **centered distribution deformation / contraction**.

Because the task metrics are nonlinear, mean and shape effects should not be interpreted as strictly additive contributions.

---

## 9. Original method: conceptual picture

The method is not best understood as a numerical corrector.

It is an **NFE-conditioned, terminally supervised corrective transport**:

```math
{\rm frozen\ bridge}
\;\longrightarrow\;
{\rm task\mbox{-}optimal\ finite\mbox{-}NFE\ flow\ map}.
```

Its defining properties are:

1. **Paired objective** — learn incremental terminal utility relative to the same-noise frozen bridge.
2. **Discrete-then-optimize** — optimize the actual uniform-Euler finite-NFE operator used at inference.
3. **Implicit flow-map learning** — parameterize local velocity / score corrections, but supervise them at the terminal finite-time-operator level.
4. **Downstream credit assignment** — optimize each local correction according to its propagated effect on terminal reward; this admits a Doob-style control interpretation without claiming exact $`h`$-transform equivalence.
5. **Mechanism decomposition** — use the post-hoc counterfactual

```math
X_{\rm mean}
=
X_0+(\mu_\phi-\mu_0)
```

to separate learned conditional mean displacement from centered distribution deformation.
6. **Fidelity-diversity trade-off** — most of the measured gain is associated with mean displacement; unrestricted deformation supplies further fidelity but can suppress diversity and coverage.

---

## Original-method summary

> We post-train a frozen generative bridge by optimizing its deployed finite-NFE flow map through a residual velocity / score field. The inference clock and uniform-Euler solver remain fixed. A same-noise paired terminal objective directly optimizes the realized discrete finite-time operator rather than an ideal continuous-time path. Dense-rollout diagnostics show that the correction primarily learns task-optimal transport rather than numerical-error compensation. A simple counterfactual construction, $`X_{\rm mean}=X_0+(\mu_\phi-\mu_0)`$, preserves the frozen bridge's centered distribution exactly while retaining most of the full model's downstream improvement, indicating that conditional mean displacement is the dominant mechanism. The unrestricted corrective transport gains additional fidelity by deforming the centered distribution, producing the observed fidelity-diversity / coverage trade-off.

---

## 10. Extension: one-stage joint-distribution regularization

The extension retains the frozen bridge, unrestricted corrective field, same-noise pairing, and prescribed uniform-Euler NFE. It changes only terminal supervision:

```math
\boxed{
\mathcal J(\phi)
=
\mathcal L_{\rm pair}(\phi)
+
\frac{\lambda}{2}
\mathbb E_{c,N}
\left[
\operatorname{MMD}_{k_\sigma}^2
\left(q_{\phi,c,N},p_c\right)
\right],
\qquad \lambda\geq0.
}
```

The conditional distributions are

```math
\rho_{\phi,c,N}
=
\operatorname{Law}\left(F_\phi^N(Z,c)\mid c,N\right),
\qquad
q_{\phi,c,N}=\Psi_\#\rho_{\phi,c,N},
\qquad
p_c=\Psi_\#P_{\rm data}(\cdot\mid c).
```

Here $`\Psi`$ is a fixed feature map applied identically to generated and real outputs; $`\#`$ denotes the induced distribution. Noise and the sampled NFE are independent of the observed future conditional on the context. The NFE therefore does not change the data target $`p_c`$.

The paired term rewards incremental task utility relative to the frozen bridge. The kernel term compares the **conditional joint distribution of forecast features**, rather than their scalarized mean or separately matched marginals. MMD is the kernel discrepancy defined in Section 12 [1].

**One scalar loss, one backward pass, one optimizer.** The coefficient $`\lambda`$ is fixed during a training run, not a dual variable. There is no separate particle-transport stage, frozen transport-target regression, critic, or outer–inner optimization loop. Setting $`\lambda=0`$ recovers the original objective for the same sampling and reduction conventions.

The observed contraction in Sections 7–8 motivates this extension; improved coverage or preserved task performance is a hypothesis to test, not an established result.

---

## 11. Fixed trajectory features and Laplace kernel

### 11.1 Concrete forecasting feature map

For this implementation, assume the terminal forecast is $`x\in\mathbb R^{H\times D}`$, with $`H\geq2`$ forecast positions and $`D`$ channels. These are forecast coordinates, **not Euler solver steps**. The original summary does not specify a tensor layout; this layout is an explicit implementation assumption.

Estimate channel-wise standard deviations of levels and signed increments from the training split, apply positive floors, and freeze the resulting scales $`s^{\rm lev},s^{\rm inc}\in\mathbb R_{>0}^D`$.

```math
\boxed{
\Psi(x)
=
\begin{bmatrix}
\displaystyle
\frac{\operatorname{vec}(x/s^{\rm lev})}{\sqrt{HD}}
\\[6pt]
\displaystyle
\frac{\operatorname{vec}(\Delta x/s^{\rm inc})}
{\sqrt{(H-1)D}}
\end{bmatrix},
\qquad
(\Delta x)_{t,d}=x_{t+1,d}-x_{t,d}.
}
```

Division is channel-wise. Ordered levels retain amplitude, timing, and cross-channel information; signed increments increase sensitivity to temporal changes. The block denominators normalize coordinate counts. Because the full level block is retained, this feature map is injective: increments change the comparison geometry, not the available information.

For example, $`(0,1,0)`$ and $`(0,0,1)`$ have the same temporal mean and variance, but different ordered features. Comparing only summary moments would miss that distinction.

Do not normalize each forecast by its own mean or variance. Do not use CRPS/MAE/RMSE against the target as $`\Psi(x)`$: those are reference-dependent task losses, not identically defined features of generated and real trajectories. Keep the existing task-loss implementation, including any ensemble-based CRPS reduction, inside $`\mathcal L_{\rm pair}`$.

### 11.2 One kernel on the whole vector

Use the **Euclidean Laplace kernel**:

```math
\boxed{
k_\sigma(u,v)
=
\exp\left(-\frac{\|u-v\|_2}{\sigma}\right),
\qquad \sigma>0.
}
```

The Euclidean distance is not squared. This kernel is characteristic; zero population MMD identifies the joint feature distribution [3]. With the injective feature map above, equality of feature distributions also identifies the full forecast distribution. This population statement does not guarantee finite-sample sensitivity to every dependence pattern.

Use a single kernel on the concatenated vector, rather than a sum of independent coordinate-wise kernels. Feature scales and bandwidth still define a fixed geometry; the method is not free of weighting choices.

**Suggested initialization.** Set $`\sigma`$ to the median nonzero generated–real feature distance on a fixed training calibration batch from the frozen bridge, always pairing forecasts with their own context's target. Then freeze it. This is a proposed heuristic, not a derived optimum. If all distances vanish, select and record a positive fallback bandwidth rather than using zero.

---

## 12. Conditional kernel score and the implemented objective

For each training pair $`(c_b,y_b)`$ and sampled $`N_b`$, draw $`K\geq2`$ independent noises. Define

```math
X_{bi}=F_\phi^{N_b}(z_{bi},c_b),
\qquad
X^0_{bi}=F_0^{N_b}(z_{bi},c_b),
\qquad
u_{bi}=\Psi(X_{bi}),
\qquad
v_b=\Psi(y_b).
```

The frozen and corrected ensembles use the same noises. The $`K`$ generated samples within a kernel-score group share the same context and NFE.

The population discrepancy has the expansion [1]

```math
\operatorname{MMD}_{k_\sigma}^2(q,p)
=
\mathbb E_{U,U'\sim q}k_\sigma(U,U')
-2\mathbb E_{U\sim q,V\sim p}k_\sigma(U,V)
+\mathbb E_{V,V'\sim p}k_\sigma(V,V'),
```

where the draws in each expectation are independent. With one observed future per context, compute the half-scaled kernel score [2]:

```math
\boxed{
\widehat S_b
=
\frac{1}{2K(K-1)}
\sum_{i\ne j}k_\sigma(u_{bi},u_{bj})
-
\frac{1}{K}\sum_i k_\sigma(u_{bi},v_b).
}
```

The generated–real term encourages similarity to observations. The generated–generated term penalizes excessive similarity within the predictive ensemble. Dropping that term leaves an attraction objective, not distribution matching.

By expanding the expectations,

```math
\mathbb E_{Y_b,Z_{b1:K}\mid c_b,N_b}[\widehat S_b]
=
\frac12\operatorname{MMD}_{k_\sigma}^2
(q_{\phi,c_b,N_b},p_{c_b})
-
C(p_{c_b}),
```

```math
C(p_c)
=
\frac12\mathbb E_{V,V'\sim p_c}k_\sigma(V,V').
```

The omitted term is independent of $`\phi`$. Therefore, the actual minibatch loss is

```math
\boxed{
\widehat{\mathcal J}
=
\widehat{\mathcal L}_{\rm pair}
+
\frac{\lambda}{B}\sum_{b=1}^B\widehat S_b.
}
```

Keep the existing paired penalty, metric normalizations, and sample/ensemble reductions unchanged. Add the kernel score **outside** the paired softplus. Do not put it inside $`\ell`$, transform it through another softplus, or differentiate through only one generated argument of the pairwise kernel.

One real future per context suffices to estimate the population scoring objective across context–outcome pairs [2]; it does not identify each conditional law from one observation. Generalization to held-out contexts remains necessary.

**Estimator details.** Exclude generated self-pairs and use $`K(K-1)`$, not $`K^2`$. Compute scores within context and NFE, then average. Do not pool unrelated contexts or NFEs into one predictive ensemble. The score may be negative because the real–real constant is omitted; do not clamp it to zero or report it as an absolute MMD value.

---

## 13. Terminal credit assignment and the WGF interpretation

### 13.1 The combined terminal gradient

For the batch loss above, initialize each endpoint adjoint as

```math
\boxed{
a_{N_b,bi}
=
\nabla_{X_{bi}}\widehat{\mathcal L}_{\rm pair}
+
\frac{\lambda}{B}\nabla_{X_{bi}}\widehat S_b.
}
```

Both contributions backpropagate through the same Euler recursion from Section 4. With $`h_b=1/N_b`$,

```math
\nabla_\phi\widehat{\mathcal J}
=
\sum_{b,i}\sum_{k=0}^{N_b-1}
h_b
\left(
\frac{\partial c_\phi(x_{k,bi},t_k,c_b,N_b)}{\partial\phi}
\right)^\top a_{k+1,bi}.
```

All batch and ensemble weights are already included in the terminal adjoints. There is no additional gradient normalization after this sum.

### 13.2 What is Wasserstein-inspired

Fix $`c,N`$ and write $`q=\Psi_\#\rho`$. For the kernel component alone, define

```math
\mathcal E(\rho)
=
\frac12\operatorname{MMD}_{k_\sigma}^2(\Psi_\#\rho,p),
\qquad
g_q(u)
=
\mathbb E_{U\sim q}k_\sigma(u,U)
-
\mathbb E_{V\sim p}k_\sigma(u,V).
```

Where the derivatives are defined, its formal state-space transport direction is

```math
V_\rho^{\rm ker}(x)
=
-\nabla_x\frac{\delta\mathcal E}{\delta\rho}(x)
=
-J_\Psi(x)^\top\nabla_u g_q(\Psi(x)).
```

For sufficiently regular kernels and measures, the corresponding continuity equation in **optimization time** $`s`$ is the MMD Wasserstein gradient flow [4]:

```math
\partial_s\rho_s
+
\nabla_x\!\cdot
\left(\rho_s V_{\rho_s}^{\rm ker}\right)
=0.
```

This geometry is in terminal state space. Feature motion is preconditioned by $`J_\Psi J_\Psi^\top`$; it is generally not Euclidean Wasserstein flow in feature space. Optimization time $`s`$ is distinct from inference time $`t_k=k/N`$.

The empirical terminal direction is already present in the single loss:

```math
\widehat V_{bi}^{\rm ker}
=
-K\nabla_{X_{bi}}\widehat S_b,
\qquad
a_{N_b,bi}
=
\nabla_{X_{bi}}\widehat{\mathcal L}_{\rm pair}
-
\frac{\lambda}{BK}\widehat V_{bi}^{\rm ker}.
```

These equations explain the gradient; **the implementation does not explicitly construct transport targets**.

### 13.3 Limits of the interpretation

**Parameter optimization is not exact WGF.** The endpoint Jacobian and optimizer determine which terminal displacements the corrective field realizes. Ordinary AdamW is not a Wasserstein natural-gradient method, and the inference drift need not equal $`V_\rho^{\rm ker}`$.

**The full paired objective is coupling-dependent.** Because the same-noise baseline appears inside a nonlinear paired penalty, two operators with the same corrected terminal distribution can have different paired losses. Thus the full $`\mathcal J`$ is not generally a functional of the corrected terminal marginal alone. The density-level WGF interpretation applies to the kernel component, not automatically to the combined objective.

**Laplace is nonsmooth at coincidence.** The reference implementation selects zero norm gradient when feature vectors coincide. Exact duplicates therefore do not automatically separate. The smooth-kernel assumptions of [4] do not hold globally for the exact Laplace kernel; no classical-flow existence or convergence theorem is claimed here.

**The local generator is not the finite-time map.** The drift $`b_\phi=v_0+c_\phi`$ induces the differential operator $`f\mapsto b_\phi\cdot\nabla f`$ for the formal continuous dynamics. The deployed $`F_\phi^N`$ remains the finite composition of Euler steps. We optimize that composition, not an exact exponential of a time-independent generator.

The precise description is:

> **Single-objective, joint-distribution-regularized finite-NFE corrective transport, with an MMD-Wasserstein terminal-gradient interpretation.**

---

## 14. Reference implementation: one loss and one update

The loss function below is self-contained. `rollout_base`, `rollout_corrected`, and `existing_paired_objective` are integration points: their implementations were not included in the supplied summary.

```python
import math
import torch
from torch import Tensor


def laplace_scores(
    x: Tensor,
    y: Tensor,
    s_level: Tensor,
    s_inc: Tensor,
    sigma: float,
) -> Tensor:
    """x [B,K,H,D], y [B,H,D], frozen scales [D]; returns scores [B]."""
    if x.ndim != 4:
        raise ValueError("Expected x with shape [B,K,H,D].")
    B, K, H, D = x.shape
    if B < 1 or K < 2 or H < 2 or D < 1 or y.shape != (B, H, D):
        raise ValueError("Require B,D >= 1, K,H >= 2 and y [B,H,D].")
    if not x.is_floating_point() or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Require floating-point x and finite sigma > 0.")

    # Use float32 for low-precision rollouts; preserve float64 for testing.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x = x.to(dtype=dtype)  # This cast preserves gradients to the rollout.
    y = y.detach().to(device=x.device, dtype=dtype)
    scales = []
    for scale in (s_level, s_inc):
        scale = scale.detach().to(device=x.device, dtype=dtype)
        if scale.shape != (D,) or not bool(
            (torch.isfinite(scale) & (scale > 0)).all()
        ):
            raise ValueError("Scales must be finite, positive tensors [D].")
        scales.append(scale)
    sl, si = scales

    def features(a: Tensor) -> Tensor:
        level = (a / sl).flatten(-2) / math.sqrt(H * D)
        inc = ((a[..., 1:, :] - a[..., :-1, :]) / si).flatten(-2)
        inc = inc / math.sqrt((H - 1) * D)
        return torch.cat((level, inc), dim=-1)

    u, v = features(x), features(y)
    i, j = torch.triu_indices(K, K, offset=1, device=x.device)
    d_xx = torch.linalg.vector_norm(u[:, i] - u[:, j], dim=-1)
    d_xy = torch.linalg.vector_norm(u - v[:, None], dim=-1)

    # Unique off-diagonal pairs: 0.5 * their mean equals the stated U-statistic.
    generated = 0.5 * torch.exp(-d_xx / sigma).mean(dim=-1)
    observed = torch.exp(-d_xy / sigma).mean(dim=-1)
    return generated - observed
```

For one NFE per minibatch, integrate it as follows:

```python
# z contains K independent noises per context; x and x0 have shape [B,K,H,D].
# Freeze v0's parameters once, but retain input derivatives in corrected rollouts.
with torch.no_grad():
    x0 = rollout_base(z, c, N)

x = rollout_corrected(z, c, N)
loss_pair = existing_paired_objective(x, x0, y)  # Preserve existing reductions.
loss_joint = laplace_scores(x, y, s_level, s_inc, sigma).mean()
loss = loss_pair + lambda_joint * loss_joint  # Fixed lambda_joint >= 0.

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

For mixed NFEs, compute each context–NFE score within its own group and average using the intended training weights before the same backward pass. Do not detach corrected endpoints or either generated argument of the pairwise kernel.

This code uses the exact stated Laplace kernel; it does not silently square or smooth the distance. Pairwise feature computation costs $`O(BK^2M)`$ for $`M=D(2H-1)`$. For large forecasts, compute pair distances in blocks while retaining the same sums and denominators. Increasing $`K`$ adds training rollouts; **one optimization process does not mean unchanged training cost**. Deployment retains the prescribed $`N`$ and uses no reference targets or kernel evaluation.

---

## 15. Validation and claims to retain

Choose $`\lambda`$ on a validation split and keep it fixed within each run. Do not equate raw loss magnitudes: the kernel score includes an omitted constant, and its gradient scale depends on the feature scales and bandwidth. Keep those fixed across compared models.

The minimum evaluation compares the original paired-only objective ($`\lambda=0`$) with the combined objective under matched architecture, NFE sampling, generated-trajectory budget, and checkpoint-selection rules. Retain the original mean-displacement counterfactual. Re-evaluate the existing task metrics, coverage, spread, pairwise diversity, and effective rank, alongside held-out kernel-score differences. Report the seen, interpolation, and extrapolation NFE regimes separately; check sensitivity to $`K`$, $`\lambda`$, and $`\sigma`$.

A held-out comparison against the frozen bridge can use

```math
\widehat{\Delta S}
=
\frac1B\sum_b
\left[
\widehat S_b(F_\phi^{N_b})
-
\widehat S_b(F_0^{N_b})
\right].
```

With the same fixed kernel and features, its expectation is half the difference between the corresponding conditional squared MMDs: the real–real constants cancel. Negative values favor the corrected model. Reusing contexts, targets, and noise improves comparability without changing this expectation. Subtracting a detached reference score is optional for logging and is not required in the training loss.

**Interpretation limits.** The kernel score alone is a proper distributional objective for the fixed characteristic kernel [2,3]. Adding the paired task term does not make the full objective proper, impose a hard performance constraint, or guarantee exact data matching. The scalar $`\lambda`$ trades task utility against joint-distribution agreement. It does not remove conflicts among task metrics. Population identifiability does not rule out finite-data overfitting, restricted model capacity, or optimization failure.

The included loss implementation passed explicit pairwise value/gradient comparisons, numerical gradient checks away from coincidences, duplicate-sample and low-precision checks, and input-validation tests. A synthetic linear Euler system also verified the combined gradient, the discrete adjoint, and the $`\lambda=0`$ reduction. These checks validate the reference implementation and differentiation, not the untested empirical benefits of the extension.

## Extension summary

> We extend the paired finite-NFE corrective bridge with a conditional Laplace kernel score on fixed, ordered forecast levels and signed increments. Its expectation equals half the squared MMD between generated and real joint-feature distributions, up to a parameter-independent constant. Adding this score outside the existing paired penalty produces one differentiable objective and one optimizer update. Both terminal signals train the same local corrective field through the deployed Euler rollout. The kernel term provides an MMD-Wasserstein transport interpretation, while the complete method remains parameter-space optimization of a coupling-dependent finite-NFE objective. Improved calibration and retained task gains require new held-out experiments.

## References and provenance

**Original source.** `finite_nfe_corrective_bridge_summary_github_ultrasafe(2).md`, supplied by the user. All reported empirical values and original mechanism claims are retained from that source; they were not independently re-evaluated. The feature design, combined objective, implementation, and evaluation protocol in Sections 10–15 are the proposed extension developed in this discussion.

[1] Gretton et al. (2012). *A Kernel Two-Sample Test*. Journal of Machine Learning Research, 13:723–773. https://www.jmlr.org/papers/v13/gretton12a.html

[2] Pacchiardi et al. (2024). *Probabilistic Forecasting with Generative Networks via Scoring Rule Minimization*. Journal of Machine Learning Research, 25(45):1–64. Sections 2.2 and Appendix C.1.2 support conditional kernel-score training and its unbiased estimator. https://jmlr.org/papers/v25/23-0038.html

[3] Sriperumbudur, Fukumizu, and Lanckriet (2011). *Universality, Characteristic Kernels and RKHS Embedding of Measures*. Journal of Machine Learning Research, 12:2389–2410. https://www.jmlr.org/papers/v12/sriperumbudur11a.html

[4] Arbel, Korba, Salim, and Gretton (2019). *Maximum Mean Discrepancy Gradient Flow*. NeurIPS. The smooth-kernel theory motivates the transport interpretation but is not a convergence theorem for the exact Laplace implementation above. https://arxiv.org/abs/1906.04370

