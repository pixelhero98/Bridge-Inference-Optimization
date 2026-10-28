# Profiling protocol

configs/ogo-profile.json fixes 100 classes, all 24 combinations, three warmup
rounds and five measured rounds per combination. Batch size is 32; microbatches
and conditional groups contain four samples. Policies and likelihoods use FP32.
Historical references retain their pinned precision. Profiling cases are
excluded from later scientific training and model selection.

Asset loading, calibration, reference reads, warmup, resume checks, preference
collection and independent development are separated from steady-state work.
CUDA synchronization brackets phases. Backbone counts include rematerialization
and CFG duplication. Committed optimizer states carry immutable provenance.

After an external restart, three unmeasured update probes warm the new process.
Each probe reuses a previously visited batch and restores the committed weights,
optimizer, RNG and cursor before the next probe or measured update. Probe costs
enter resource accounting, and no probe advances the scientific trajectory.

Before accepted timings: verify densities/scores, drift conversion, initial
Euler equivalence, reward pairing, DRaFT truncation, baseline loss/gradient
parity, frozen references, finite updates, on-policy replay and exact AdamW
resume. Existing canonical finite-difference fixtures remain available.

Reference reuse checks provenance and hashes and reconstructs initial seeds and
classes. Paired, matching and development roles are disjoint. SiT flattened
terminal-latent features are reshaped without decoding or quantization loss.

The ceiling is eight aggregate GH200-hours including setup, cleanup,
verification and failures. Every allocation enters the ledger. Uncovered cases
must be reported; the matrix cannot be reduced silently.

Price training by rollout rounds, optimizer updates and replay passes. Class
count alone does not determine training cost. Price shared pools separately at
8/16/32 paired cases per class, plus matching and development. Add offline
preference collection, final generation and metric extraction.

Timing does not establish quality improvements or justify selecting easy or
high-reward examples. Later sampling must be fixed before inspecting outcomes.

The original training-profile development records time unmerged LoRA endpoints;
SiT endpoints are latent tensors. A separate deployment audit materializes LoRA,
checks endpoint parity, and times Euler plus RGB decoding on the same independent
development inputs. Use these complete RGB timings for image-generation costs.
The 16M corrective field remains separate in OGO deployment. Image encoding,
filesystem writes and quality-metric extraction are additional costs.

The accepted cost report and measurement-selection manifest are in the project's
results/ogo-cost-assessment-20261026 directory. They preserve original records,
select the fresh warmed EDM DRaFT-32 repeat after exact external-resume proof,
and include the complete allocation ledger. The basic CLI report exposes the
original endpoint timings; it labels them separately from full RGB generation.

The profiling pool is disposable. Its short schedule visits a fixed portion of
the 100-class subset; it is not itself the 100-class scientific screen. Final
scientific sampling must cover all declared classes regardless of reward.
