"""Скачивание корпуса Icelandic Gigaword Corpus (IGC2024 Filtered-1).

IGC2024 доступен через CLARIN-IS (https://igc-clarin.hi.is/) — обычно требуется
авторизация (CLARIN login). Поэтому скрипт:
  - печатает ссылку и подсказку;
  - после ручного скачивания архива в data/raw/igc/ его подхватывает 02_clean.py
    (поддерживаются .jsonl/.tsv/.txt/.xml/.zip/.tar.gz);
  - дополнительно пытается скачать открытые исландские корпуса (Risamálheild /
    Leipzig) как fallback, если указать флаги.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isllm.data import download_file  # noqa: E402

IGC_SITE = "https://igc-clarin.hi.is/"
IGC_HINT_DIR = "data/raw/igc"

LEIPZIG_URLS = [
    # Открытые исландские корпуса Leipzig (лист по isl): прямая ссылка меняется.
    "https://downloads.wortschatz-leipzig.de/corpora/isl_news_2023.tar.gz",
]
TATOEBA_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/is.txt.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leipzig", action="store_true", help="попробовать Leipzig (isp/isl)")
    parser.add_argument("--tatoeba", action="store_true", help="попробовать Tatoeba (is)")
    parser.add_argument("--outdir", default="data/raw")
    args = parser.parse_args()

    out = Path(args.outdir) / "igc"
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("Icelandic Gigaword Corpus (IGC2024 Filtered-1)")
    print("Сайт:", IGC_SITE)
    print("Доступ: обычно требуется логин CLARIN. Скачайте архив вручную в:")
    print(f"  {IGC_HINT_DIR}/  (например igc2024_filtered_1.jsonl.gz или .tar.gz)")
    print("Затем запустите scripts/02_clean.py --corpus-path <путь>.")
    print("=" * 72)

    if args.leipzig:
        for i, url in enumerate(LEIPZIG_URLS):
            try:
                download_file(url, str(Path(args.outdir) / "leipzig" / f"corpus_{i}.tar.gz"))
            except Exception as e:  # noqa: BLE001
                print(f"[leipzig] не удалось: {e}")

    if args.tatoeba:
        try:
            download_file(TATOEBA_URL, str(Path(args.outdir) / "tatoeba" / "is.txt.gz"))
        except Exception as e:  # noqa: BLE001
            print(f"[tatoeba] не удалось: {e}")

    print("\nГотово. Если скачали IGC вручную — пропустите 01_download и сразу 02_clean.")


if __name__ == "__main__":
    main()