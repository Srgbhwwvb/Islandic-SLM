"""Сводный отчёт: artifacts/report.md из всех накопленных результатов."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load(path: str) -> dict:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    stats = _load("data/prepared/stats.json")
    grid = _load("tokenizer/grid_results.json")
    bench = _load("artifacts/benchmark.json")

    lines = ["# Сводный отчёт — Icelandic LLM", ""]

    lines.append("## 1. Корпус")
    train = stats.get("train", {})
    lines.append(f"- документов train: {train.get('n_docs', '?')}, val: {stats.get('val', {}).get('n_docs', '?')}")
    lines.append(f"- символов train: {train.get('n_chars', '?')}, слов: {train.get('n_words', '?')}")
    lines.append(f"- уникальных слов: {train.get('unique_words', '?')}")
    lines.append(f"- причины отсева: {stats.get('reject_reasons', '?')}")

    lines.append("")
    lines.append("## 2. Токенизатор")
    if grid:
        lines.append("| вариант | chars/token | tokens/word | byte-fb | rare | roundtrip |")
        lines.append("|---|---|---|---|---|---|")
        for r in grid:
            bf = r.get("byte_fallback_rate")
            bf_s = "n/a" if bf is None else f"{bf:.3f}"
            lines.append(
                f"| {r.get('alg')}_{r.get('vocab_size')} | {r.get('chars_per_token', 0):.2f} | "
                f"{r.get('tokens_per_word', 0):.2f} | {bf_s} | "
                f"{r.get('rare_token_rate', 0):.3f} | {r.get('roundtrip_accuracy', 0):.3f} |"
            )

    lines.append("")
    lines.append("## 3. Обучение")
    lines.append("Кривые loss/ppl — в TensorBoard (`logs/`) и `notebooks/report.ipynb`.")

    lines.append("")
    lines.append("## 4. Бенчмарк")
    if bench:
        lines.append(f"- perplexity: {bench.get('perplexity', '?')}")
        lines.append(f"- accuracy@1: {bench.get('accuracy@1', '?')}, accuracy@5: {bench.get('accuracy@5', '?')}")
        lines.append(f"- MRR: {bench.get('MRR', '?')}")
        lines.append(f"- согласованность парафраз: {bench.get('consistency', '?')}")
        for f in bench.get("per_fact", []):
            lines.append(f"  - {f['fact']}: ranks={f['ranks']} {'OK' if f['consistent'] else 'X'}")
    else:
        lines.append("Бенчмарк ещё не запускался (после обучения: scripts/09_benchmark.py).")

    lines.append("")
    lines.append("## 5. Эксперименты")
    lines.append("См. configs/exp_*.yaml; результаты — в artifacts/ после прогонов.")

    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()