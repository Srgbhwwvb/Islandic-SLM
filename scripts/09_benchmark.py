"""Бенчмарк: python scripts/09_benchmark.py --ckpt checkpoints/large [--corpus data/prepared/val.txt]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from isllm.benchmark import format_report, load_facts, perplexity, run_cloze_benchmark  # noqa: E402
from isllm.generation import load_model_from_dir  # noqa: E402
from isllm.tokenizer import load_tokenizer  # noqa: E402


def load_val_texts(corpus: str, max_chars: int = 500_000) -> list[str]:
    """Список предложений для ppl-оценки (если указан .bin — берём data/prepared/val.txt)."""
    path = Path(corpus)
    if path.suffix == ".bin":
        fallback = Path("data/prepared/val.txt")
        print(f"[bench] {path} — бинарный; для ppl используем {fallback}")
        path = fallback
    if not path.exists():
        return []
    texts = []
    chars = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 20:
                texts.append(line)
                chars += len(line)
            if chars > max_chars:
                break
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="директория чекпоинта (Trainer output_dir)")
    parser.add_argument("--tokenizer", default="tokenizer/final")
    parser.add_argument("--corpus", default="data/prepared/val.txt")
    parser.add_argument("--facts", default="isllm/facts_is.json")
    parser.add_argument("--max-texts", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(args.tokenizer)
    model_dir = Path(args.ckpt)
    if model_dir.suffix in (".pt", ".bin"):
        model_dir = model_dir.parent
    model = load_model_from_dir(str(model_dir), device=device)
    print(f"loaded model from {model_dir}")

    texts = load_val_texts(args.corpus)[: args.max_texts]
    ppl = perplexity(model, tokenizer, texts, device=device)
    print(f"Perplexity on {len(texts)} texts: {ppl:.2f}")

    facts = load_facts(args.facts)
    results = run_cloze_benchmark(model, tokenizer, facts, device=device)
    results["perplexity"] = ppl
    report = format_report(results)

    Path("artifacts").mkdir(exist_ok=True)
    (Path("artifacts") / "benchmark.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path("artifacts") / "benchmark_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()