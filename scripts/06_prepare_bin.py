"""Токенизация корпуса в uint16 memmap train.bin / val.bin + meta.json."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isllm.tokenizer import prepare_bins  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="tokenizer/final")
    parser.add_argument("--out", default="data/bins")
    parser.add_argument("--train-file", default="data/prepared/train.txt")
    parser.add_argument("--val-file", default="data/prepared/val.txt")
    args = parser.parse_args()
    prepare_bins(args.tokenizer, outdir=args.out,
                 train_file=args.train_file, val_file=args.val_file)


if __name__ == "__main__":
    main()