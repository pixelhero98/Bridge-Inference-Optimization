# Equation-to-code audit

Source: user-supplied *Finite-Time Corrective Transport for Generative ODE
Post-Training*, ICML2027.pdf. SHA256:
`88992974b0cfd45019100c1a67bc6eea748f5ef72c81b74df4976b66081f8d0c`.
Equation numbers refer to that document; the unpublished PDF is not redistributed here.

| Paper item | Canonical implementation | Independent evidence |
|---|---|---|
| Eqs. 2–4: finite corrected operator | `rollout.euler_step`, `uniform_euler` | Scalar closed form; explicit current-state and time-grid test |
| Eq. 8: frozen teacher | `rollout.frozen_rollout`; task target provider | Separate teacher stream; zero-correction equality |
| Eq. 9: utility plus matching | `objectives.conditional_loss` | Utility/matching/joint fixtures and gradient additivity |
| Eq. 15: frozen input Jacobians | Full generated rollout; trainer freezes parameters only | Backbone-detach control; checkpoint/offload equivalence |
| Eq. 21: conditional kernel score | `objectives.kernel_score` | Independent scalar pair loops, K/M normalization tests |
| Algorithm 1: both generated arguments | Ordered off-diagonal interaction | One-sided-kernel control |
| Eqs. 24–25: discrete adjoint | `verification_cases.discrete_adjoint` | Separate reverse recursion and full-autograd/FD comparison |
| Eqs. 41–43: particle covectors | Analytic terminal covector in `verification_cases` | Endpoint gradient and extra-ensemble-mean control |
| Fixed feature geometry | `FeatureKernel` branches | Latent + decoded perceptual gradient test; frozen-parameter assertions |

M=1 reproduces Eq. 21. M>1 averages independent teacher attractions with factor
1/(KM); generated repulsion remains 1/(2K(K-1)). The terminal covector used in E0
is an analytic Gaussian/tanh implementation, independent of autograd on the loss.
The NumPy objective uses separate scalar loops. The adjoint shares the canonical
local Euler step but performs a distinct backward recursion.

Controls are confined to `verification_cases`; the training API exposes no gradient
omission flags. Full E0 endpoints and losses invoke the installed canonical sampler
and objective. Historical post-step/standalone operators and official-value/proxy-gradient
helpers are absent from the active API.

The review motivating these changes was conducted against commit `35c0e1b`.
Its archived `rollout.py:95-100` used a post-step displacement;
`training.py:240-248` reused student noise for targets;
`README.md:145` acknowledged missing stage continuation;
`precision.py:38-39` rejected FP64; and `pyproject.toml:13-14` lacked an importable
package beneath the source discovery root. See the archive inventory for original hashes.
