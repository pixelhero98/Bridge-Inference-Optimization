"""Run a declared task adapter and two-regime training configuration."""
import argparse
import json
from pathlib import Path

import torch

from .training import Trainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int, help="Absolute update boundary for an intentional stop")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    trainer = Trainer(config, args.output, resume=args.resume)
    summary = trainer.run(args.stop_after)
    print(json.dumps({k: summary[k] for k in ("status", "updates", "cursor")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
