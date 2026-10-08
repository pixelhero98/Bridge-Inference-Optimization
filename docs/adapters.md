# Task and backbone adapter contract

The JSON training configuration names an explicit `module:factory`. The factory
receives task options plus seed, dtype, device, student steps, and teacher steps.
It returns the `TaskAdapter` protocol in `fct.interfaces`:

- `correction`: a trainable module returning velocity with the current state's shape.
- `dynamics`: deterministic frozen velocity in normalized time.
- `batch(cursor)`: separate `ContextBatch` instances, each with student initial
  draws, independently sampled matching endpoints, and role-qualified identities.
- `utility(endpoint, context, baseline)`: one scalar loss per student draw.
- `branches`: `FeatureKernel` objects; each feature callable returns [draws, features].
- `frozen_modules`: all frozen modules, including modules hidden inside callables.
- `identity()`: weights/checkpoint hashes, dataset split, preprocessing, kernels,
  feature weights/bandwidths, utility definition, and sampling convention.
- `state_dict()/load_state_dict()`: all mutable sampler state, including custom RNGs.

IDs expose accidental reuse but cannot prove independence. The adapter is responsible
for actual independent streams; do not copy student noise and assign a new teacher ID.
Optional paired utility baselines use student IDs and stay distinct from matching targets.
Sampling must be reproducible from the committed cursor and restored adapter state.

## Latent plus perceptual example

```python
latent = FeatureKernel(
    "latent", lambda z, c: z.flatten(1), GaussianKernel(sigma_latent), beta_latent)
perceptual = FeatureKernel(
    "dino", lambda z, c: dino_features(preprocess(decoder(z))),
    GaussianKernel(sigma_dino), beta_dino)
adapter.branches = (latent, perceptual)
adapter.frozen_modules = (backbone, teacher, decoder, dino_encoder, reward_model)
```

This is an interface example, not a completed DINO integration. A task specification
must choose DINO version/weights, token selection/pooling, resizing, normalization,
bandwidth calibration and the decoded range. Generated preprocessing must remain
differentiable; PIL/NumPy conversions and `no_grad` on generated images break it.
Verify frozen weights and nonzero decoder/feature input derivatives independently.

Keep task choices out of the core: latent layout/scaling, text encoders, CFG,
reward transforms, feature selection, batch size, learning rate, stage lengths,
teacher budget, evaluation metrics, and train/validation/test splits. A new adapter
must pass frozen-sampler equivalence, full-rollout finite differences on a small
fixture, target-stream tests, and stage/resume tests before large experiments.

Both FP32 and FP64 are explicit policies in this reference. AMP, proxy gradients,
truncated backpropagation, stochastic generated layers, distributed training and
large-model orchestration are not implemented by this first core release.
