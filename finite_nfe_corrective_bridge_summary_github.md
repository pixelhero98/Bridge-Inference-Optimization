# Finite-NFE Corrective Bridge: Method Summary

## 1. Core idea

We start from a frozen generative bridge / probability-flow field

$$
\dot{x}_t = v_0(x_t,t;c),
$$

and keep the deployed inference procedure fixed: **uniform-grid Euler with a prescribed NFE \(N\)**.

We learn an NFE- and context-conditioned residual field

$$
c_\phi(x_t,t,c,N),
$$

so that the deployed dynamics are

$$
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
$$

The final method uses only this **unrestricted corrective field**. We do not learn a clock, solver, or separate translation-only model.

The key empirical finding is that the correction is **not primarily compensating numerical discretization error**. It learns a task-optimal transport for the downstream objective. Most of its gain can be explained by moving the conditional mean, while the remaining gain comes from deforming the centered distribution, which trades diversity / coverage for additional fidelity.

---

## 2. Paired terminal objective

For the same context $c$, target $y$, initial noise $z$, and NFE $N$, let

$$
X_0 = F_0^N(z,c)
$$

be the frozen-bridge endpoint and

$$
X_\phi = F_\phi^N(z,c)
$$

be the corrected endpoint.

For a lower-is-better terminal task loss $\ell$, define the paired excess loss

$$
\Delta_\phi
=
\ell(X_\phi,y)
-
\operatorname{sg}\!\left[\ell(X_0,y)\right].
$$

We optimize a smooth paired improvement objective

$$
\boxed{
\mathcal{L}_{\mathrm{pair}}(\phi)
=
\mathbb{E}_{c,y,z,N}
\left[
\tau\,
\operatorname{softplus}
\left(
\frac{\Delta_\phi+m}{\tau}
\right)
\right]
}
$$

where $\tau$ is a temperature and $m$ is an optional improvement margin.

The frozen endpoint is used as a **same-noise counterfactual baseline**. This gives lower-variance and more stable supervision than directly optimizing the corrected terminal loss.

Empirically:

- paired training learns substantially smaller corrections than direct training;
- its advantage remains after reducing the learning rate of direct training;
- it generalizes better across unseen NFEs and contexts;
- therefore the effect is objective-level, not simply an effective step-size difference.

---

## 3. Discrete-then-optimize

The important modeling choice is the order of operations.

We do **not** optimize a continuous-time bridge first and then ask how well a numerical solver approximates it.

Instead, for every target NFE $N$, we first define the actual deployed discrete operator

$$
F_\phi^N
=
G_{\phi,N-1}
\circ
\cdots
\circ
G_{\phi,0},
$$

with

$$
G_{\phi,k}(x)
=
x
+
h\left[
v_0(x,t_k;c)
+
c_\phi(x,t_k,c,N)
\right].
$$

We then optimize this realized finite-time operator directly:

$$
\boxed{
\phi^\star
=
\arg\min_\phi
\mathcal{L}_{\mathrm{pair}}
\left(
F_\phi^N,
F_0^N
\right).
}
$$

Hence the method is **discretize first, then optimize the deployed finite-NFE transport**.

This differs from:

- timestep / clock optimization;
- solver design;
- distillation toward a dense numerical trajectory;
- pure numerical-error compensation.

Dense-rollout ablations show that the dominant improvement remains when the corrected field is integrated densely. Therefore the learned correction mainly changes the **task-optimal transport itself**, rather than merely repairing Euler truncation error.

---

## 4. Implicit flow-map / finite-time operator model

Although the correction is parameterized locally at the velocity / score level, supervision is applied only through the completed rollout.

The model is therefore best viewed as

$$
\boxed{
\text{field-level parameterization}
\quad+\quad
\text{flow-map-level objective}.
}
$$

The residual field implicitly parameterizes the finite-time map

$$
F_\phi^N : z \mapsto X_\phi.
$$

A local correction at step $k$ is useful only through its effect on the terminal objective. With terminal adjoint

$$
a_N
=
\nabla_{x_N}\ell,
$$

the discrete backward recursion is approximately

$$
a_k
=
\left[
I
+
hJ_x\!\left(v_0+c_\phi\right)
\right]^\top
a_{k+1},
$$

and

$$
\nabla_\phi \mathcal{L}
=
\sum_{k=0}^{N-1}
h
\left(
\frac{\partial c_{\phi,k}}{\partial \phi}
\right)^\top
a_{k+1}.
$$

Thus terminal supervision automatically assigns credit over the entire trajectory.

A first-order endpoint perturbation also has the form

$$
F_\phi^N(z)
-
F_0^N(z)
\approx
\sum_{k=0}^{N-1}
h\,
J^{0}_{k+1\rightarrow N}
\,
c_\phi(x_k,t_k,c,N),
$$

so a small local correction can have a large terminal effect when applied in a direction with high downstream sensitivity.

---

## 5. Relation to a Doob $h$-transform: interpretation, not equivalence

There is a useful conceptual connection to terminally conditioned stochastic control.

For a baseline stochastic path law $P_0$ and terminal reward $R(X_T)$, the KL-regularized control problem

$$
\max_Q
\left\{
\mathbb{E}_Q[R(X_T)]
-
\frac{1}{\beta}
D_{\mathrm{KL}}(Q\|P_0)
\right\}
$$

has the tilted path law

$$
dQ^\star
\propto
e^{\beta R(X_T)}
\,dP_0.
$$

Its Doob $h$-transform uses

$$
h(x,t)
=
\mathbb{E}_{P_0}
\left[
e^{\beta R(X_T)}
\mid
X_t=x
\right],
$$

so the local controlled dynamics depend on the **downstream desirability of future terminal states**.

Our learned correction has a related interpretation: every velocity correction is trained using terminal reward / loss propagated through all subsequent dynamics. It therefore internalizes both:

1. the state produced by preceding velocity decisions; and
2. the downstream effect of the current intervention on terminal utility.

However, our method is **not formally a Doob $h$-transform** as currently defined. We optimize a deterministic finite-NFE Euler operator with a parametric paired terminal objective, without explicitly constructing a KL-regularized tilted stochastic path measure.

The safe interpretation is

$$
\boxed{
\text{Doob-style downstream credit assignment / terminally conditioned control view}
}
$$

rather than exact $h$-transform equivalence.

---

## 6. Conditional mean displacement vs distribution deformation

To understand *how* the unrestricted corrective transport improves the task, we use a post-hoc counterfactual diagnostic.

For a fixed context and NFE, define

$$
X_0
=
F_0^N(z,c),
\qquad
X_\phi
=
F_\phi^N(z,c),
$$

with conditional means

$$
\mu_0
=
\mathbb{E}_z[X_0],
\qquad
\mu_\phi
=
\mathbb{E}_z[X_\phi].
$$

Write

$$
X_0
=
\mu_0+A_0,
\qquad
X_\phi
=
\mu_\phi+A_\phi,
$$

where $A_0$ and $A_\phi$ are centered anomalies.

We then construct

$$
\boxed{
X_{\mathrm{mean}}
=
X_0
+
(\mu_\phi-\mu_0)
=
\mu_\phi+A_0.
}
$$

This is **not another trained model**. It combines:

- the **conditional mean learned by the full corrective transport**; and
- the **centered stochastic geometry of the frozen bridge**.

Because

$$
X_{\mathrm{mean}}
-
\mu_\phi
=
X_0
-
\mu_0,
$$

all pairwise differences are preserved:

$$
X_{\mathrm{mean}}^{(i)}
-
X_{\mathrm{mean}}^{(j)}
=
X_0^{(i)}
-
X_0^{(j)}.
$$

Therefore covariance, effective rank, pairwise diversity, and any other translation-invariant diversity statistic are preserved exactly in the modeled state space.

Operationally, estimating $\mu_\phi-\mu_0$ requires both frozen and corrected ensembles, so this is primarily a **mechanism diagnostic**, not the preferred deployment method.

---

## 7. Current held-out evidence

Normalized composite $(\mathrm{CRPS}+\mathrm{MAE}+\mathrm{RMSE})/3$, lower is better:

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

The explicitly trained translation-only model is much weaker. Therefore we do not need a separate translation model in the final method: the unrestricted corrective field learns a substantially better task-directed conditional mean displacement, and the post-hoc mean counterfactual is sufficient to identify its contribution.

---

## 8. Fidelity-diversity mechanism

The current evidence suggests the following decomposition:

$$
\boxed{
\text{full task gain}
=
\text{gain associated with conditional mean displacement}
+
\text{additional gain enabled by distribution deformation}
}
$$

with an important caveat: nonlinear task metrics make the two effects non-additive in general.

The mean counterfactual

$$
X_{\mathrm{mean}}
=
\mu_\phi+A_0
$$

shows that a large fraction of task improvement can be obtained **without changing the centered distribution**.

The unrestricted full transport

$$
X_\phi
=
\mu_\phi+A_\phi
$$

improves fidelity further by allowing

$$
A_\phi \neq A_0.
$$

Empirically, this extra freedom often causes:

- lower pairwise diversity;
- lower covariance / stochastic spread;
- worse coverage;
- reduced effective rank in aggressive settings.

Therefore the additional fidelity obtained by the unrestricted correction is partly purchased through **centered distribution deformation / contraction**.

This gives two useful operating interpretations.

### Diversity-sensitive analysis

Use the learned full transport, but form the diagnostic distribution

$$
\mu_\phi + A_0
$$

to measure how much of the gain can be retained when the frozen bridge's centered stochastic geometry is preserved.

### Fidelity-dominant deployment

Deploy the unrestricted corrected transport

$$
\mu_\phi+A_\phi
$$

when maximum task fidelity is preferred over preserving the original conditional diversity.

---

## 9. Final conceptual picture

The method is not best understood as a numerical corrector.

It is an **NFE-conditioned, terminally supervised corrective transport**:

$$
\boxed{
\text{frozen bridge}
\;\xrightarrow{\;\text{paired terminal post-training}\;}
\text{task-optimal finite-NFE flow map}.
}
$$

Its defining properties are:

1. **Paired objective**  
   Learn incremental terminal utility relative to the same-noise frozen bridge.

2. **Discrete-then-optimize**  
   Optimize the actual uniform-Euler finite-NFE operator used at inference.

3. **Implicit flow-map learning**  
   Parameterize local velocity / score corrections, but supervise them at the terminal finite-time-operator level.

4. **Downstream credit assignment**  
   Each local correction is optimized according to its propagated effect on terminal reward; this admits a Doob-style control interpretation without claiming exact $h$-transform equivalence.

5. **Mechanism decomposition**  
   The post-hoc mean transport

   $$
   X_0+(\mu_\phi-\mu_0)
   $$

   separates the learned conditional mean displacement from centered distribution deformation.

6. **Fidelity-diversity trade-off**  
   Most of the measured gain is associated with learned mean displacement; unrestricted deformation supplies further fidelity but can suppress diversity and coverage.

---

## One-paragraph summary

> We post-train a frozen generative bridge by optimizing its deployed finite-NFE flow map through a residual velocity / score field. The model keeps the inference clock and uniform-Euler solver fixed and is trained with a same-noise paired terminal objective, so the learned field is optimized for the realized discrete finite-time operator rather than for an ideal continuous-time path. Dense-rollout diagnostics show that the correction primarily learns task-optimal transport rather than numerical-error compensation. A simple counterfactual construction, \(X_{\mathrm{mean}}=X_0+(\mu_\phi-\mu_0)\), preserves the frozen bridge's centered distribution exactly while retaining most of the full model's downstream improvement, showing that conditional mean displacement is the dominant mechanism. The unrestricted corrective transport gains additional fidelity by deforming the centered distribution, producing the observed fidelity-diversity / coverage trade-off.
