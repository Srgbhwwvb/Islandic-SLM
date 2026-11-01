"""Подготовка корпуса: IGC2024 Filtered-1 -> чистые документы, языковой фильтр,
дедуп, train/val split.

```
python scripts/02_clean.py --corpus-path data/raw/igc/igc2024_filtered_1.jsonl.gz
```

Вся логика в isllm.data.prepare_corpus (её же использует notebook train.ipynb).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isllm.data import prepare_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-path",
                        default="data/raw/igc/igc2024_filtered_1.jsonl.gz",
                        help="файл/архив IGC (jsonl/tsv/txt/xml/zip/tar.gz)")
    parser.add_argument("--outdir", default="data/prepared")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="ограничить число документов (0 = все)")
    parser.add_argument("--min-doc-chars", type=int, default=80)
    parser.add_argument("--val-frac", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-dedup", action="store_true",
                        help="пропустить дедуп (если IGC уже дедуплицирован)")
    args = parser.parse_args()

    prepare_corpus(
        corpus_path=args.corpus_path,
        outdir=args.outdir,
        max_docs=args.max_docs,
        min_doc_chars=args.min_doc_chars,
        val_frac=args.val_frac,
        seed=args.seed,
        no_dedup=args.no_dedup,
    )


if __name__ == "__main__":
    main()