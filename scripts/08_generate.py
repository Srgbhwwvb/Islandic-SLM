"""Генерация текста: python scripts/08_generate.py --ckpt checkpoints/large --prompt ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from isllm.generation import generate, load_model_from_dir  # noqa: E402
from isllm.tokenizer import load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="директория чекпоинта (Trainer output_dir)")
    parser.add_argument("--tokenizer", default="tokenizer/final")
    parser.add_argument("--prompt", default="Höfuðborg Íslands er —")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(args.tokenizer)
    # Если передан файл (best.pt) — берём его родителя как output_dir Trainer
    model_dir = Path(args.ckpt)
    if model_dir.suffix in (".pt", ".bin"):
        model_dir = model_dir.parent
    model = load_model_from_dir(str(model_dir), device=device)
    print(f"loaded model from {model_dir}")

    results = []
    for i in range(args.num_samples):
        text = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                        do_sample=True, seed=args.seed + i, device=device)
        results.append({"seed": args.seed + i, "prompt": args.prompt, "text": text})
        print(f"\n--- sample {i+1} (seed {args.seed+i}) ---\n{text}")

    Path("artifacts").mkdir(exist_ok=True)
    (Path("artifacts") / "generated.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()