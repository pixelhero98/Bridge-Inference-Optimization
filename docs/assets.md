# Explicit asset contract

The repository contains source and compact numerical evidence, not model weights,
datasets, credentials or machine-specific launchers. Paths are supplied through
`--assets`; loaders validate source revisions and content hashes and fail on
missing or incompatible inputs. Use trusted upstream weights and pinned sources.

## ImageNet profiling

The asset JSON consumed by `ogo-study` supplies these fields:

| Field | Meaning |
|---|---|
| `shared_manifest` | JSON describing SiT, VAE, DINO, VGG/LPIPS and evaluator/data assets |
| `edm_manifest` | JSON with `sources.edm` (path/revision) and `assets.edm` (path/SHA256) |
| `edm_cache` | Directory of pinned tensor shards and matching JSON descriptors |
| `edm_weight_sha256`, `edm_sampler_sha256` | Exact reference model and sampler identities |
| `sit_pool`, `sit_pool_identity` | SiT latent reference shards and their immutable identity |

Source records use `path` and `revision`; weight records use `path` and `sha256`.
The shared manifest provides `sources.sit`, `sources.dino`, `assets.sit`,
`assets.dino`, `assets.vgg16`, `lpips.sha256`, and `vae.path`/`vae.files`.
VAE file hashes are checked before local-only loading. The LPIPS record pins
the calibration weights in the installed package, separately from VGG weights.
See `backbones.load` and `reference_pool.ReferencePool` for executable checks.

Reference-cache IDs may retain historical `teacher-training` names. They are
immutable data identities, not an active teacher-training objective. Never
rename a cache ID or fabricate provenance to make a new source accept it.

## ImageNet-256 production

`ogo-campaign` additionally requires:

| Field | Meaning |
|---|---|
| `reuse_audit` | Reserved profiling/calibration/reference IDs to exclude |
| `protected_root`, `protection_receipt` | Root and relative-path hashes of protected source/input dependencies |
| `evaluation_python` | Python executable for the isolated ADM evaluator environment |
| `evaluation_overlay` | Evaluator support directory with its hashed `ready.json` |
| `cost_assessment` | Accepted measured baseline/deployment cost report used by preflight |
| `campaign_ledger` | Allocation accounting record, when supplied to the audit |

The shared manifest also supplies `assets.resnet50`, `assets.adm_inception`,
`sources.adm` and `imagenet.path`/`imagenet.files`. The evaluator verifies the
official validation population and hashes the Parquet inputs. The separate
evaluator environment must satisfy the pinned ADM implementation; installing the
package's image extra alone does not create it.

Provision these manifests and an allocation with project-local caches before
numerical work. Torch's VGG cache uses a hardlink to the supplied VGG file, so
place that cache on the same filesystem as the weight. Run `verify` first;
`prepare`, `collect`, `train` and `evaluate` require the passing identity-bound
preflight receipt. Checkpoints and outputs must remain tied to their original
source and input manifests. These specialized commands do not initialize a
generic T2I or T2V campaign.
