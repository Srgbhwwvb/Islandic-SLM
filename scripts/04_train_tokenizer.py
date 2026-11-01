"""Обучение токенизаторов: --grid (сетка 2 алгоритма x 4 размера) и --final (один вариант)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isllm.tokenizer import (  # noqa: E402
    DEFAULT_ALGS,
    DEFAULT_VOCABS,
    evaluate_tokenizer,
    sample_texts,
    train_single,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", action="store_true", help="обучить всю сетку")
    parser.add_argument("--final", type=int, default=0,
                        help="обучить один вариант с этим vocab_size (алгоритм из --alg)")
    parser.add_argument("--alg", default="unigram", choices=["bpe", "unigram"])
    parser.add_argument("--corpus", default="data/prepared/train.txt")
    parser.add_argument("--outdir", default="tokenizer")
    parser.add_argument("--sample-chars", type=int, default=1_000_000,
                        help="сколько символов использовать для быстрой сетки")
    args = parser.parse_args()

    # Фикстура tests/data/ — НЕ корпус: предупредить, чтобы не обучать токенизатор на ней
    if "tests/data" in str(args.corpus).replace("\\", "/"):
        print("[warning] Вы используете тестовый фикстур tests/data/* — это НЕ корпус! "
              "Обучайте токенизатор на data/prepared/train.txt.")

    # Для сетки берём подвыборку для скорости; для финального — весь корпус.
    if args.grid:
        sample = sample_texts(args.corpus, max_chars=args.sample_chars)
        sample_file = Path(args.outdir) / "_grid_sample.txt"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        sample_file.write_text("\n".join(sample), encoding="utf-8")
        inputs = [str(sample_file)]
        print(f"[grid] сэмпл: {len(sample)} строк, {sum(len(s) for s in sample)/1e6:.1f}M символов")
        vocab_sizes = DEFAULT_VOCABS
        algs = DEFAULT_ALGS
    elif args.final:
        inputs = [args.corpus]
        vocab_sizes = [args.final]
        algs = [args.alg]
    else:
        parser.error("Укажите --grid или --final N")

    results = []
    for alg in algs:
        for vs in vocab_sizes:
            name = f"{alg}_{vs}"
            path = Path(args.outdir) / name
            print(f"\n=== {name} ===")
            tok = train_single(str(path), input_files=inputs, alg=alg, vocab_size=vs,
                               min_frequency=2, byte_fallback=(alg == "bpe"))
            eval_sample = sample_texts(args.corpus, max_chars=200_000)
            metrics = evaluate_tokenizer(tok, eval_sample[:2000], alg=alg)
            metrics.update({"alg": alg, "vocab_size": vs})
            results.append(metrics)
            with open(Path(args.outdir) / f"{name}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

    if args.final:
        # скопировать выбранный в tokenizer/final
        import shutil
        src = Path(args.outdir) / f"{args.alg}_{args.final}"
        dst = Path(args.outdir) / "final"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"\nФинальный токенизатор: {dst}")

    with open(Path(args.outdir) / "grid_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nСводка:")
    for r in results:
        bf = r.get("byte_fallback_rate")
        print(f"  {r['alg']}_{r['vocab_size']}: fert={r.get('chars_per_token',0):.2f} "
              f"tokens/word={r.get('tokens_per_word',0):.2f} "
              f"byte_fb={bf if bf is not None else 'n/a'} "
              f"rare={r.get('rare_token_rate',0):.3f} roundtrip={r.get('roundtrip_accuracy',0):.2f}")


if __name__ == "__main__":
    main()