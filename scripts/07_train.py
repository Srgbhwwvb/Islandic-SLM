"""Обучение модели: python scripts/07_train.py --config configs/large.yaml [--max-steps N]."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isllm.config import load_config  # noqa: E402
from isllm.training import set_seed, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--resume", default=None, help="путь к чекпоинту Trainer (dir) или 'true'")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.grad_accum:
        config.train.grad_accum = args.grad_accum
    if args.seed is not None:
        config.train.seed = args.seed
    if args.outdir:
        config.train.outdir = args.outdir
    set_seed(config.train.seed)

    result = train(config, max_steps_override=args.max_steps,
                   log_every=args.log_every, device=args.device,
                   resume_from_checkpoint=args.resume)
    print("\nРезультат:", json.dumps(result, indent=2))
    print("Логи: tensorboard --logdir", config.train.log_dir)


if __name__ == "__main__":
    main()