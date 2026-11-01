"""Сравнение токенизаторов: читает grid_results.json и строит таблицу/рекомендацию."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    path = Path("tokenizer/grid_results.json")
    if not path.exists():
        print("Нет tokenizer/grid_results.json — сначала scripts/04_train_tokenizer.py --grid")
        return
    results = json.loads(path.read_text(encoding="utf-8"))
    print(f"{'name':<18} {'fert':>6} {'tok/w':>6} {'bytefb':>7} {'rare':>6} {'roundtrip':>9} {'speed':>8}")
    for r in results:
        bf = r.get("byte_fallback_rate")
        print(f"{r['alg']}_{r['vocab_size']:<14} {r.get('chars_per_token',0):6.2f} "
              f"{r.get('tokens_per_word',0):6.2f} "
              f"{bf if bf is not None else 'n/a':>7} {r.get('rare_token_rate',0):6.3f} "
              f"{r.get('roundtrip_accuracy',0):9.3f} "
              f"{r.get('chars_per_sec',0):8.0f}")

    best = max(results, key=lambda r: (r.get("chars_per_token", 0), -r.get("rare_token_rate", 0)))
    print(f"\nЛучший по metrics: {best['alg']}_{best['vocab_size']} "
          f"(fert {best.get('chars_per_token',0):.2f} chars/token)")
    # BPE на исландском обычно даёт лучшую fertility; 32768 — компромисс
    # компактность vs размер эмбеддингов (для 110M-модели).
    bpe = [r for r in results if r.get("alg") == "bpe"]
    if bpe:
        rec = max(bpe, key=lambda r: (r.get("chars_per_token", 0) if r.get("vocab_size", 0) <= 32768 else 0,
                                      -abs(r.get("vocab_size", 0) - 32768)))
        print(f"Рекомендуемый (BPE, 16384..32768): {rec['alg']}_{rec['vocab_size']}")
    print("Затем: python scripts/04_train_tokenizer.py --final 32768 --alg bpe")


if __name__ == "__main__":
    main()