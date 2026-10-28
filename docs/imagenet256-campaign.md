# ImageNet-256 single-seed FID-10K campaign

This is the frozen first stage of the [OGO experiment plan](experiment-plan.md).
The planning catalog does not change this executable contract or expand its
budget. Existing deployed source releases remain immutable; newly published
source requires a fresh identity and verification, not an in-place resume.

This production campaign compares DRaFT-K, DDPO-IS, Flow-GRPO, lambda-Controlled GRPO and offline trajectory-DPO based on D3PO. Each method has a separate SiT-XL/2 policy for 16 and 32 uniform Euler steps, trained for exactly 512 optimizer updates. Seed 0 and all 1,000 classes are fixed. OGO and ImageNet-64 are subsequent stages.

`configs/imagenet256-baselines-512.json` freezes the native baseline contract from `ogo-profile.json`. The production runner imports its verified update equations; it does not change clipping, learning rates, adapter scope/rank, FP32 precision or stochastic training transitions. These architecture adaptations are not claims of exact paper reproduction. Deterministic deployment has no Gaussian transition likelihood.

Every class has eight paired training references, two independent development references and ten independent final references. References use the pinned initial SiT weights, Euler-256, CFG 1.5 on the verified first three channels and the pinned VAE scale. Reference generation reproduces the original teacher's math-attention runtime; policy training and deployment retain the verified policy runtime. Compatible old records are read without changing the teacher pool. A full original 32-case block is reserved exclusively for exact reference reconstruction and excluded from every scientific role. Profiling/calibration/reserved IDs are excluded. A frozen seed-0 class permutation and cyclic case traversal determine all training inputs without reward screening.

DRaFT presents 16,384 fresh trajectories. Each online RL method presents 8,192 fresh trajectories with two optimizer replay passes per batch. D3PO collects 8,000 immutable pairs per horizon from the initial policy and presents 8,192 pairs during training. Pair members share class, initial noise and paired reference, with independent transition noise. Shards retain actual pair/input identities, labels and reference likelihoods. Equal optimizer counts do not imply equal sample exposure or compute.

The primary endpoint is update 512; no checkpoint selection is allowed. Development at updates 128/256/384 uses the same 2,000 independent cases. Final evaluations and both frozen controls use 10,000 images (ten per class), compared with the official 50,000-image validation reference through pinned ADM features. Report FID-10K, paired LPIPS, top-1/top-5 classification and mean within-class squared distance of unit DINO features. Retain per-case raw measurements and features. Image quantization for ADM is explicit; LPIPS uses continuous decoded RGB as in training.

The executable interface takes explicit paths:

```bash
ogo-campaign verify --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign prepare --part references --partition 0 --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign prepare --part references --partition 1 --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign prepare --part seal --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign prepare --part real --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign collect --steps 16 --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign train --method ddpo_is --steps 16 --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign evaluate --method ddpo_is --steps 16 --role final --update 512 --config CONFIG --assets ASSETS --output OUTPUT --deadline UNIX_TIME
ogo-campaign audit --config CONFIG --assets ASSETS --output OUTPUT
ogo-campaign report --config CONFIG --assets ASSETS --output OUTPUT
```

Numerical commands require an allocated GPU node. The private scheduler driver permits at most two campaign GPUs, reserves full pending/running job limits, and charges every terminal allocation including failures and retries against 110 GH200-hours. All new campaign storage is limited to 64 GB. A failed preflight or insufficient remaining budget preserves evidence and stops; class/update/evaluation counts and thresholds are never reduced.

Atomic checkpoint pointers bind source/configuration/asset/reference identities, trainable parameters, Adam state, RNG and cursor. Rolling checkpoints occur every 32 optimizer updates; 128/256/384/512 snapshots remain. An interruption between two RL replay passes resumes from the last committed complete round and repeats uncommitted work exactly. State-restored warmup probes consume allocation but no scientific updates. Separate data-shard and optimizer interruption tests compare the actual training recurrence, including RNG. Deployment merges LoRA only after output parity; old profiling releases remain immutable.

The first production verification profiles reference reconstruction, the real training recurrence, preference-shard I/O and the complete evaluator. The allocation forecast includes all ten policies, three development readouts each, both controls, preferences, missing references and ADM extraction. Completion requires all ten terminal checkpoints, twelve final FID evaluations, 32 development readouts, passed audits and final scheduler accounting. A launched or scheduler-completed job alone is not evidence of completed scientific work.
