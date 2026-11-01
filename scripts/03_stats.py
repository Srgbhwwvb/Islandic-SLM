"""Анализ корпуса: объём, распределения, Zipf, частотность — в data/prepared/stats.json + графики."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from isllm.data import corpus_stats, read_docs  # noqa: E402


def plot_stats(stats: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    train = stats["train"]

    # Гистограмма длины документов
    hist = train.pop("doc_len_hist", [])
    if hist:
        labels = [h["range"] for h in hist]
        counts = [h["count"] for h in hist]
        plt.figure(figsize=(8, 4))
        plt.bar(range(len(counts)), counts)
        plt.xticks(range(len(counts)), labels, rotation=45)
        plt.ylabel("документов")
        plt.title("Распределение длины документов (символы)")
        plt.tight_layout()
        plt.savefig(outdir / "doc_len_hist.png", dpi=120)
        plt.close()

    top = train.get("top_words", [])
    if top:
        words = [w for w, _ in top]
        freqs = [f for _, f in top]
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(freqs)), freqs)
        plt.xticks(range(len(words)), words, rotation=45, ha="right")
        plt.ylabel("частота")
        plt.title(f"Топ-{len(words)} слов (выборка: {train.get('word_stats_sample', '?')} док.)")
        plt.tight_layout()
        plt.savefig(outdir / "top_words.png", dpi=120)
        plt.close()


def main() -> None:
    prepared = Path("data/prepared")
    train_docs = read_docs(prepared / "train.txt")
    val_docs = read_docs(prepared / "val.txt")
    stats_path = prepared / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    print(f"train docs: {len(train_docs)}, val docs: {len(val_docs)}")
    print("train stats:", json.dumps(corpus_stats(train_docs), ensure_ascii=False)[:2000])
    print("val stats:", json.dumps(corpus_stats(val_docs), ensure_ascii=False)[:800])

    stats.update({"train": corpus_stats(train_docs), "val": corpus_stats(val_docs)})
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    out = Path("artifacts/plots")
    plot_stats(stats, out)
    print(f"Графики сохранены в {out}/")


if __name__ == "__main__":
    main()