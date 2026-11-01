"""Подготовка корпуса Icelandic Gigaword Corpus (IGC2024 Filtered-1).
"""
from __future__ import annotations

import bz2
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Iterator
from tqdm.auto import tqdm
import numpy as np
import requests

IGC_URL = "https://igc-clarin.hi.is/"

# Символы, характерные для исландского
ICELANDIC_SPECIFIC = set("þðæöáéíóúýÞÐÆÖÁÉÍÓÚÝ")
LATIN = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_RE = re.compile(r"&[a-zA-Z#0-9]+;")
_URL_RE = re.compile(r"(?:https?|ftp)://\S+|//[^\s,\)\]]+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\u200c\u200d\ufeff]")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_EMPTY_PAREN_RE = re.compile(r"\(\s*[,;:]?\s*\)")
_EMPTY_COMMA_RE = re.compile(r"(?:[,;:]\s*){2,}")
_PUNCT_JUNK_RE = re.compile(r"[,;:]\s*([.!?])")
_DASH_RUN_RE = re.compile(r"(?:[—–]\s*){2,}|(?:-\s*){2,}")
_DASH_JUNK_RE = re.compile(r"[—–]\s*(?:[.,:;!?]+\s*)+")
_APOS_RE = re.compile(r"'{2,5}")
_SKIP_PATTERNS = [
    re.compile(r"^={2,}.*={2,}\s*$"),        # вики-заголовки 
    re.compile(r"^\[\[.*\]\]$"),             # внутренние ссылки
    re.compile(r"^<[^>]+>.*</[^>]+>$"),      # остатки XML-строк
]


def download_file(url: str, dest: str, chunk_size: int = 4 * 1024 * 1024,
                  timeout: int = 60) -> None:
    """Скачать файл. Для IGC обычно нужно авторизоваться на CLARIN —
    тогда скачайте архив вручную в data/raw/igc/ и вызовите prepare_corpus."""
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        print(f"already exists: {dest} ({dest_path.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"downloading {url} -> {dest}")
    resume_from = dest_path.stat().st_size if dest_path.exists() else 0
    headers = {"User-Agent": "IcelandicLLM-data-prep/1.0 (educational)"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + resume_from
        down = resume_from
        mode = "ab" if resume_from and r.status_code == 206 else "wb"
        if mode == "wb":
            down = 0
            total = int(r.headers.get("content-length", 0))
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    down += len(chunk)
                    if total:
                        print(f"\r{down / 1e6:.1f} / {total / 1e6:.1f} MB", end="")
    print()



def _open_maybe_compressed(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix in (".bz2",):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _iter_text_lines(path: Path) -> Iterator[str]:
    with _open_maybe_compressed(path) as f:
        for line in f:
            yield line.rstrip("\n")


def _extract_text_field(record: dict) -> str | None:
    for key in ("text", "txt", "body", "content", "sentence", "doc", "article"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v:
            return " ".join(str(x) for x in v)
    return None


def iter_igc_docs(path: str | Path) -> Iterator[str]:
    """Потоково читать документы из IGC (jsonl/tsv/txt/xml/zip/tar)."""
    p = Path(path)
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_file():
                yield from iter_igc_docs(child)
        return

    if p.suffix in (".zip",):
        with zipfile.ZipFile(p) as zf:
            for name in zf.namelist():
                if name.endswith("/") or "__MACOSX" in name:
                    continue
                with zf.open(name) as fh:
                    raw = fh.read()
                suffix = Path(name).suffix
                if suffix == ".gz":
                    raw = gzip.decompress(raw)
                if suffix in (".jsonl", ".json"):
                    for line in raw.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            yield line
                            continue
                        if isinstance(rec, dict) and (t := _extract_text_field(rec)):
                            yield t
                        elif isinstance(rec, str):
                            yield rec
                elif suffix in (".tsv", ".csv", ".tab"):
                    text = raw.decode("utf-8", errors="replace")
                    for row in csv.reader(text.splitlines(), delimiter="\t"):
                        if row and row[0].strip():
                            yield "\t".join(c for c in row).strip()
                else:
                    text = raw.decode("utf-8", errors="replace")
                    yield from split_blocks(text)
        return

    if p.suffix in (".tar.gz", ".tgz", ".tar.bz2", ".tar"):
        mode = "r:gz" if p.suffix in (".tar.gz", ".tgz") else (
            "r:bz2" if p.suffix == ".tar.bz2" else "r:")
        with tarfile.open(p, mode) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                fh = tf.extractfile(m)
                if fh is None:
                    continue
                raw = fh.read()
                suffix = Path(m.name).suffix
                if suffix in (".jsonl", ".json"):
                    for line in raw.decode("utf-8", errors="replace").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            yield line
                            continue
                        if isinstance(rec, dict) and (t := _extract_text_field(rec)):
                            yield t
                        elif isinstance(rec, str):
                            yield rec
                elif suffix in (".tsv", ".csv", ".tab"):
                    text = raw.decode("utf-8", errors="replace")
                    for row in csv.reader(text.splitlines(), delimiter="\t"):
                        if row and row[0].strip():
                            yield "\t".join(c for c in row).strip()
                else:
                    yield from split_blocks(raw.decode("utf-8", errors="replace"))
        return

    suffix = p.suffix.lower()
    if suffix in (".jsonl", ".json"):
        for line in _iter_text_lines(p):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                yield line
                continue
            if isinstance(rec, dict) and (t := _extract_text_field(rec)):
                yield t
            elif isinstance(rec, str):
                yield rec
        return
    if suffix in (".tsv", ".csv", ".tab"):
        for row in csv.reader(_iter_text_lines(p), delimiter="\t"):
            if row and row[0].strip():
                yield "\t".join(c for c in row).strip()
        return
    if suffix in (".xml", ".html", ".htm"):
        text = "".join(_iter_text_lines(p))
        text = _TAG_RE.sub(" ", text)
        text = _ENTITY_RE.sub(" ", text)
        yield from split_blocks(text)
        return
    if suffix in (".txt", ".text") or suffix == "":
        # Блоки, разделённые пустой строкой (как в старом формате train.txt)
        yield from split_blocks("".join(_iter_text_lines(p)))
        return
    # Неизвестное расширение — считаем текстом
    yield from split_blocks("".join(_iter_text_lines(p)))


def split_blocks(text: str) -> list[str]:
    """Разбить сплошной текст на блоки по пустой строке"""
    return [d.strip() for d in re.split(r"\n\s*\n", text) if d.strip()]



def clean_text(text: str) -> str:
    """Очистка: теги, сущности, URL, управляющие символы,
    нормализация пробелов/пунктуации"""
    text = _TAG_RE.sub(" ", text)
    text = _ENTITY_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _CTRL_RE.sub(" ", text)
    text = _APOS_RE.sub("", text)
    text = _EMPTY_PAREN_RE.sub(" ", text)
    text = _DASH_RUN_RE.sub(" ", text)
    text = _DASH_JUNK_RE.sub(" ", text)
    text = _EMPTY_COMMA_RE.sub("", text)
    text = _PUNCT_JUNK_RE.sub(r"\1", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def is_icelandic_doc(text: str, min_len: int = 80,
                     min_ice_spec: float = 0.002,
                     min_lat: float = 0.8,
                     max_digits: float = 0.15) -> tuple[bool, str]:
    stripped = text.strip()
    if len(stripped) < min_len:
        return False, "too_short"
    
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False, "no_letters"
    
    # Буквы должны составлять минимум 65% текста
    if len(letters) / len(stripped) < 0.65:
        return False, "too_much_punctuation"

    # Убиваем википедийные сноски и таблицы
    if stripped.count("[") > 5 or stripped.count("]") > 5 or stripped.count("|") > 3:
        return False, "wiki_or_table_format"
        
    lat = sum(1 for c in letters if c in LATIN) / len(letters)
    ice_spec = sum(1 for c in letters if c in ICELANDIC_SPECIFIC) / len(letters)
    if lat < min_lat:
        return False, "not_latin"
    if ice_spec < min_ice_spec:
        return False, "not_icelandic"
    digits = sum(1 for c in stripped if c.isdigit()) / max(len(stripped), 1)
    if digits > max_digits:
        return False, "numeric_heavy"
    return True, "ok"



# Дедупликация (sha1 + SimHash) 
def simhash(text: str, ngram: int = 8, bits: int = 64, stride: int = 1) -> int:
    """SimHash по символьным n-граммам."""
    if bits != 64:
        raise ValueError("simhash currently supports only bits=64")
    if len(text) < ngram:
        return 0
    if len(text) > 100_000:
        text = text[:100_000]
    data = np.frombuffer(text.encode("utf-32-le", errors="ignore"), dtype=np.uint32).copy()
    if len(data) < ngram:
        return 0
    windows = np.lib.stride_tricks.sliding_window_view(data, ngram)[::stride]
    if windows.shape[0] == 0:
        return 0
    with np.errstate(over="ignore"):
        base = np.uint64(1099511628211)
        powers = np.ones(ngram, dtype=np.uint64)
        for k in range(1, ngram):
            powers[k] = powers[k - 1] * base
        hashes = (windows.astype(np.uint64) * powers).sum(axis=1, dtype=np.uint64)
    bit_rows = np.unpackbits(
        np.ascontiguousarray(hashes).view(np.uint8).reshape(-1, 8),
        axis=1, bitorder="little",
    )
    ones = bit_rows.sum(axis=0, dtype=np.int64)
    half = len(hashes) / 2
    out = 0
    for b in range(bits):
        if ones[b] > half:
            out |= 1 << b
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _hash_many(texts: list[str], stride: int = 1) -> list[int]:
    return [simhash(t, stride=stride) for t in texts]


def _hash_all(docs: list[str], workers: int | None,
              hash_stride: int, chunk_size: int = 2000) -> list[int]:
    chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
    if workers is None:
        workers = max(1, (os.cpu_count() or 4) - 1)
    results: list[list[int]]
    if workers <= 1 or len(chunks) <= 1:
        results = [_hash_many(c, hash_stride) for c in chunks]
    else:
        try:
            from multiprocessing import Pool
            hasher = partial(_hash_many, stride=hash_stride)
            with Pool(workers) as pool:
                results = pool.map(hasher, chunks)
        except Exception as e:
            print(f"[dedup] multiprocessing недоступен ({type(e).__name__}: {e}); "
                  f"считаю последовательно", flush=True)
            results = [_hash_many(c, hash_stride) for c in chunks]
    return [h for r in results for h in r]


def dedup_docs(docs: list[str], hamming_threshold: int = 3,
               workers: int | None = None, hash_stride: int = 2,
               bucket_cap: int = 2000) -> tuple[list[str], int]:
    """Убрать точные и near-дубли (SimHash + LSH, корзина по длине)."""
    n = len(docs)
    print(f"[dedup] хеширую {n} док (stride={hash_stride}, workers={workers or 'auto'})...",
          flush=True)
    hashes = _hash_all(docs, workers, hash_stride)
    print("[dedup] хеши готовы, строю LSH...", flush=True)

    seen = set()
    bands: list[dict[tuple[int, int], list[tuple[int, int]]]] = [{} for _ in range(8)]
    out: list[str] = []
    exact_removed = 0
    near_removed = 0
    skipped_cand = 0
    oversized_buckets = 0

    pbar = tqdm(total=n, desc="dedup", unit="doc", dynamic_ncols=True, mininterval=1.0)
    try:
        for d, h in zip(docs, hashes):
            key = hashlib.sha1(d.encode("utf-8", errors="ignore")).digest()
            if key in seen:
                exact_removed += 1
                pbar.update(1)
                continue
            L = len(d)
            L_bucket = L // 20
            parts = [(h >> (16 * b)) & 0xFFFF for b in range(8)]
            is_dup = False
            for b in range(8):
                bucket = bands[b].get((L_bucket, parts[b]))
                if not bucket:
                    continue
                if len(bucket) > bucket_cap:
                    skipped_cand += 1
                    continue
                for xL, x in bucket:
                    if 0.9 * L <= xL <= 1.1 * L and hamming(h, x) <= hamming_threshold:
                        is_dup = True
                        break
                if is_dup:
                    break
            if is_dup:
                near_removed += 1
                pbar.update(1)
                continue
            seen.add(key)
            for b in range(8):
                bk = (L_bucket, parts[b])
                bucket = bands[b].setdefault(bk, [])
                if len(bucket) > bucket_cap:
                    if len(bucket) == bucket_cap + 1:
                        oversized_buckets += 1
                    continue
                bucket.append((L, h))
            out.append(d)
            pbar.update(1)
    finally:
        pbar.close()

    print(f"[dedup] точных дублей: {exact_removed}, near-дублей: {near_removed}, "
          f"осталось: {len(out)}", flush=True)
    if oversized_buckets or skipped_cand:
        print(f"[dedup] переполненных бакетов: {oversized_buckets}, "
              f"пропущено проверок: {skipped_cand}", flush=True)
    return out, exact_removed + near_removed



def split_docs(docs: list[str], val_frac: float = 0.03,
               seed: int = 42) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    n_val = max(1, int(len(docs) * val_frac))
    val_idx = set(idx[:n_val])
    train = [d for i, d in enumerate(docs) if i not in val_idx]
    val = [d for i, d in enumerate(docs) if i in val_idx]
    return train, val


def write_docs(docs: list[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d + "\n\n")


def read_docs(path: str, max_docs: int | None = None) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    if max_docs:
        docs = docs[:max_docs]
    return docs


def _histogram(values: list[int], bins: list[int]) -> list[dict]:
    out = []
    for lo, hi in zip(bins, bins[1:]):
        n = sum(1 for v in values if lo <= v < hi)
        out.append({"range": f"{lo}-{hi}", "count": n})
    return out


def corpus_stats(docs: list[str], top_n: int = 50,
                 sample_for_words: int = 20_000, seed: int = 42) -> dict:
    """Статистики корпуса: символы, слова, предложения, топ-слова, гистограмма длин."""
    n_docs = len(docs)
    total_chars = 0
    total_words = 0
    total_sents = 0
    doc_lens: list[int] = []
    for d in docs:
        L = len(d)
        total_chars += L
        doc_lens.append(L)
        total_words += (d.count(" ") + 1) if d else 0
        total_sents += d.count(".") + d.count("!") + d.count("?") + d.count("…")

    rng = random.Random(seed)
    sample = docs if n_docs <= sample_for_words else rng.sample(docs, sample_for_words)
    word_freq: Counter = Counter()
    for d in sample:
        word_freq.update(d.lower().split())
    scale = n_docs / max(len(sample), 1)
    unique_est = int(len(word_freq) * scale)

    return {
        "n_docs": n_docs,
        "n_chars": total_chars,
        "n_words_estimate": total_words,
        "n_sentences": total_sents,
        "avg_doc_chars": total_chars / max(n_docs, 1),
        "unique_words_est": unique_est,
        "n_words": total_words,
        "unique_words": unique_est,
        "top_words": word_freq.most_common(top_n),
        "doc_len_hist": _histogram(doc_lens, bins=[0, 100, 500, 1000, 2000, 4000, 8000, 1_000_000]),
        "sampled_for_word_stats": len(sample) < n_docs,
        "word_stats_sample": len(sample),
    }



def prepare_corpus(corpus_path: str, outdir: str = "data/prepared",
                   max_docs: int = 0, min_doc_chars: int = 80,
                   val_frac: float = 0.03, seed: int = 42,
                   no_dedup: bool = False) -> dict:
    """IGC2024 Filtered-1 -> data/prepared/{train,val}.txt + stats.json.

    Шаги: чтение IGC -> очистка (clean_text) -> языковой
    фильтр -> дедуп (sha1 + SimHash) -> train/val split -> статистики.
    """
    src = Path(corpus_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Нет корпуса {src}. Скачайте IGC2024 Filtered-1 (см. scripts/01_download.py)"
            f" или укажите --corpus-path."
        )

    if any(marker in str(src) for marker in ("tests/data", "test_", "sample")):
        print("[warning] Похоже, вход — тестовый фикстур, а не настоящий IGC!",
              flush=True)

    print(f"[0] Читаю IGC из {src} ...", flush=True)
    raw_docs: list[str] = []
    reject_reasons: Counter = Counter()
    n_scanned = 0
    for i, text in enumerate(iter_igc_docs(str(src))):
        n_scanned = i + 1
        if max_docs and n_scanned > max_docs:
            break
        clean = clean_text(text)
        ok, reason = is_icelandic_doc(clean, min_len=min_doc_chars)
        if ok:
            raw_docs.append(clean)
        else:
            reject_reasons[reason] += 1
        if n_scanned % 100_000 == 0:
            print(f"    {n_scanned} документов, принято {len(raw_docs)}", flush=True)

    print(f"[1] Документов: {n_scanned}, принято: {len(raw_docs)}", flush=True)
    print("    Причины отсева:", dict(reject_reasons), flush=True)

    if no_dedup:
        deduped, removed = raw_docs, 0
    else:
        print("[2] Дедупликация (точные + SimHash)...", flush=True)
        deduped, removed = dedup_docs(raw_docs)
    print(f"    удалено дублей: {removed} (осталось {len(deduped)})", flush=True)

    train, val = split_docs(deduped, val_frac=val_frac, seed=seed)
    write_docs(train, Path(outdir) / "train.txt")
    write_docs(val, Path(outdir) / "val.txt")

    print("[3] Считаю статистики train...", flush=True)
    train_stats = corpus_stats(train)
    print("[4] Считаю статистики val...", flush=True)
    val_stats = corpus_stats(val)

    stats = {
        "corpus": str(src),
        "raw_docs": n_scanned,
        "accepted": len(deduped),
        "removed_duplicates": removed,
        "reject_reasons": dict(reject_reasons),
        "train": train_stats,
        "val": val_stats,
        "icelandic_spec_chars": "þ ð æ ö á é í ó ú ý",
        "pipeline": "symbolic clean_text + language guard + sha1/simhash dedup",
    }
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    (outdir_path / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[5] Готово: {outdir_path/'train.txt'}, {outdir_path/'val.txt'}, stats.json")
    return stats

clean_wikipedia = clean_text