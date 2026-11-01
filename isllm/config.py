"""Загрузка и валидация конфигураций из YAML."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class ModelConfig:
    # Архитектура: mistral | mixtral | llama | gpt2 (transformers, случайная инициализация)
    arch: str = "mistral"
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    seq_len: int = 512
    vocab_size: int = 32772
    # Для Mixtral:
    use_moe: bool = False
    n_experts: int = 8
    n_experts_per_tok: int = 2
    moe_aux_loss_coef: float = 0.01
    # Для Mistral: sliding window; None = полное внимание
    use_swa: bool = False
    swa_window: int | None = 512
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TrainConfig:
    data_dir: str = "data/bins"
    tokenizer: str = "tokenizer/final"
    outdir: str = "checkpoints/model"
    log_dir: str = "logs/model"
    batch_size: int = 16
    grad_accum: int = 1
    grad_clip: float = 1.0
    lr: float = 6e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_steps: int = 10000
    eval_every: int = 500
    save_every: int = 1000
    gen_every: int = 1000
    save_total_limit: int = 2
    eval_iters: int = 100
    seed: int = 42
    amp: str = "bf16"            # bf16 | fp16 | none
    compile_model: bool = False
    grad_checkpointing: bool = False
    num_workers: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {"name": self.name, "model": self.model.to_dict(), "train": self.train.to_dict()}


def _coerce_float(v):
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _parse_model(d: dict) -> ModelConfig:
    allowed = set(ModelConfig.__dataclass_fields__)
    out = {k: v for k, v in d.items() if k in allowed}
    for f in ("dropout", "moe_aux_loss_coef"):
        if f in out:
            out[f] = _coerce_float(out[f])
    return ModelConfig(**out)


def _parse_train(d: dict) -> TrainConfig:
    allowed = set(TrainConfig.__dataclass_fields__)
    out = {k: v for k, v in d.items() if k in allowed}
    for f in ("grad_clip", "lr", "weight_decay"):
        if f in out:
            out[f] = _coerce_float(out[f])
    for f in ("warmup_steps", "max_steps", "eval_every", "save_every", "gen_every", "eval_iters",
              "batch_size", "grad_accum", "seed", "num_workers", "save_total_limit"):
        if f in out and isinstance(out[f], str):
            try:
                out[f] = int(out[f])
            except ValueError:
                pass
    return TrainConfig(**out)


def load_config(path: str | Path, overrides: dict | None = None) -> ExperimentConfig:
    """Загрузить YAML-конфиг; overrides — словарь с CLI-переопределениями (top-level)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _deep_update(raw, overrides)

    model_raw = raw.get("model", {}) or {}
    train_raw = raw.get("train", {}) or {}
    model = _parse_model(model_raw)
    train = _parse_train(train_raw)
    name = raw.get("name", Path(path).stem)
    return ExperimentConfig(name=name, model=model, train=train)


def save_config_json(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)


def load_meta(data_dir: str | Path) -> dict:
    with open(Path(data_dir) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)