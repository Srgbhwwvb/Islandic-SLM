"""Обучение и оценка токенизаторов (BPE / Unigram) на исландском корпусе.

Используем библиотеку `tokenizers`, каждый словарь обучается с нуля на нашем подготовленном корпусе.

Метрики сравнения:
- chars_per_token (fertility): сколько символов покрывает один токен;
- tokens_per_word;
- byte_fallback_rate: доля токенов вида <0xXX> (редкие/неизвестные байты);
- rare_token_rate: доля токенов c частотой <= 2 на валидационном сэмпле;
- roundtrip: доля текстов, точно восстановленных encode->decode;
- speed: chars/sec на сэмпле.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE, Unigram
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import BpeTrainer, UnigramTrainer

SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]

DEFAULT_VOCABS = [8192, 16384, 32768, 65536]
DEFAULT_ALGS = ["bpe", "unigram"]


def _sample_paths() -> list[str]:
    """Пути к подготовленным текстам (по умолчанию data/prepared)."""
    return [Path("data/prepared/train.txt").as_posix()]


def train_single(
    tokenizer_path: str,
    input_files: Iterable[str] | None = None,
    alg: str = "unigram",
    vocab_size: int = 16384,
    min_frequency: int = 2,
    byte_fallback: bool = True,
    lower: bool = False,
    seed: int = 42,
) -> Tokenizer:
    """Обучить один токенизатор и сохранить его в tokenizer_path.tar?  (в json-папку)."""
    input_files = list(input_files) if input_files else _sample_paths()
    out = Path(tokenizer_path)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        BPE(unk_token="<unk>", byte_fallback=byte_fallback)
        if alg == "bpe"
        else Unigram(vocab=[("<unk>", 0.0)], unk_id=0, byte_fallback=byte_fallback)
    )
    if lower:
        tokenizer.normalizer = Sequence([NFKC()])
    else:
        tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    if alg == "bpe":
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=True,
        )
    else:
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

    tokenizer.train(files=list(input_files), trainer=trainer)
    tokenizer.save(str(out / "tokenizer.json"))
    with open(out / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=1)
    (out / "config.json").write_text(
        json.dumps(
            {
                "alg": alg,
                "vocab_size": vocab_size,
                "byte_fallback": byte_fallback,
                "lower": lower,
                "special_tokens": SPECIAL_TOKENS,
                "files": [os.path.basename(p) for p in input_files],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return tokenizer


def load_tokenizer(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))


def encode_text_file(path: str, tokenizer: Tokenizer, out_path: str) -> int:
    import numpy as np

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    total_tokens = 0
    eos_id = tokenizer.token_to_id("<eos>")
    buffer_ids = []
    buffer_limit = 1_000_000

    with open(out, "wb") as f_out:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    buffer_ids.append(eos_id)
                else:
                    ids = tokenizer.encode(line + " ").ids
                    if ids:
                        buffer_ids.extend(ids)
                
                if len(buffer_ids) >= buffer_limit:
                    np.asarray(buffer_ids, dtype=np.uint16).tofile(f_out)
                    total_tokens += len(buffer_ids)
                    buffer_ids.clear()
            
            if buffer_ids:
                np.asarray(buffer_ids, dtype=np.uint16).tofile(f_out)
                total_tokens += len(buffer_ids)

    print(f"  {path}: {total_tokens:,} токенов -> {out}")
    return total_tokens


def prepare_bins(tokenizer_path: str, outdir: str = "data/bins",
                 train_file: str = "data/prepared/train.txt",
                 val_file: str = "data/prepared/val.txt") -> dict:
    """Токенизировать train/val в uint16 memmap и сохранить meta.json."""
    tokenizer = load_tokenizer(tokenizer_path)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    n_train = encode_text_file(train_file, tokenizer, str(out / "train.bin"))
    n_val = encode_text_file(val_file, tokenizer, str(out / "val.bin"))

    meta = {
        "vocab_size": len(tokenizer.get_vocab()),
        "vocab": tokenizer.get_vocab(),
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "tokenizer": tokenizer_path,
        "special_eos": tokenizer.token_to_id("<eos>"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"meta -> {out/'meta.json'} (vocab_size={meta['vocab_size']})")
    return meta


def evaluate_tokenizer(tokenizer: Tokenizer, sample_texts: list[str],
                       alg: str | None = None) -> dict:
    """Посчитать метрики на готовом токенизаторе.
    """
    if not sample_texts:
        return {}
    n_chars = sum(len(t) for t in sample_texts)
    n_tokens = 0
    n_words = sum(len(t.split()) for t in sample_texts)
    byte_count = 0
    vocab = list(tokenizer.get_vocab().keys())
    rare_tokens = {t for t in vocab if t.startswith("<0x")}

    t0 = time.time()
    for text in sample_texts:
        enc = tokenizer.encode(text)
        ids = enc.ids
        n_tokens += len(ids)
        if alg == "bpe":
            for tid in ids:
                tok = tokenizer.id_to_token(tid)
                if tok in rare_tokens or (tok is not None and tok.startswith("<0x")):
                    byte_count += 1
    elapsed = max(time.time() - t0, 1e-9)

    freqs: dict[str, int] = {}
    for text in sample_texts[: min(len(sample_texts), 5000)]:
        for tid in tokenizer.encode(text).ids:
            tok = tokenizer.id_to_token(tid) or ""
            freqs[tok] = freqs.get(tok, 0) + 1
    rare_rate = sum(1 for v in freqs.values() if v <= 2) / max(len(freqs), 1)

    def _norm(s: str) -> str:
        return "\n".join(line.strip() for line in s.splitlines()).strip()

    round_final = sum(
        1 for t in sample_texts
        if tokenizer.decode(tokenizer.encode(t).ids) == t
        or _norm(tokenizer.decode(tokenizer.encode(t).ids)) == _norm(t)
    )
    round_final = round_final / max(len(sample_texts), 1)

    return {
        "chars_per_token": n_chars / max(n_tokens, 1),
        "tokens_per_word": n_tokens / max(n_words, 1),
        "byte_fallback_rate": (byte_count / max(n_tokens, 1)) if alg == "bpe" else None,
        "rare_token_rate": rare_rate,
        "roundtrip_accuracy": round_final,
        "chars_per_sec": n_chars / elapsed,
        "vocab_size": len(vocab),
        "alg": alg,
    }


def sample_texts(path: str | Path, max_chars: int = 3_000_000, max_docs: int = 20000) -> list[str]:
    """Прочитать кусок корпуса для быстрой оценки"""
    p = Path(path)
    if not p.exists():
        return []
    texts: list[str] = []
    chars = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() and len(line) >= 20:
                texts.append(line)
                chars += len(line)
            if chars >= max_chars or len(texts) >= max_docs:
                break
    return texts