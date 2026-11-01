"""Построение моделей transformers со случайной инициализации из конфига.
"""
from __future__ import annotations

import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    MixtralConfig,
    MixtralForCausalLM,
)

from .config import ModelConfig

ARCH_BY_NAME = {
    "gpt2": (GPT2Config, GPT2LMHeadModel),
    "llama": (LlamaConfig, LlamaForCausalLM),
    "mistral": (MistralConfig, MistralForCausalLM),
    "mixtral": (MixtralConfig, MixtralForCausalLM),
}


def _base_kwargs(cfg: ModelConfig) -> dict:
    return {
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.n_embd,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.n_head,
        "tie_word_embeddings": cfg.tie_word_embeddings,
    }


def build_hf_config(cfg: ModelConfig) -> object:
    """Создать конфиг transformers из наших параметров"""
    cfg_cls, _ = ARCH_BY_NAME[cfg.arch]
    if cfg.arch == "gpt2":
        return cfg_cls(
            vocab_size=cfg.vocab_size,
            n_embd=cfg.n_embd,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_positions=cfg.seq_len,
            resid_pdrop=cfg.dropout,
            attn_pdrop=cfg.dropout,
            embd_pdrop=cfg.dropout,
            tie_word_embeddings=cfg.tie_word_embeddings,
        )
    if cfg.arch == "llama":
        return cfg_cls(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.n_embd,
            intermediate_size=int(8 * cfg.n_embd / 3),
            num_hidden_layers=cfg.n_layer,
            num_attention_heads=cfg.n_head,
            num_key_value_heads=cfg.n_head,
            max_position_embeddings=cfg.seq_len,
            tie_word_embeddings=cfg.tie_word_embeddings,
            attention_dropout=cfg.dropout,
        )
    if cfg.arch == "mistral":
        return cfg_cls(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.n_embd,
            intermediate_size=int(8 * cfg.n_embd / 3),
            num_hidden_layers=cfg.n_layer,
            num_attention_heads=cfg.n_head,
            num_key_value_heads=cfg.n_head,
            max_position_embeddings=cfg.seq_len,
            sliding_window=cfg.swa_window if cfg.use_swa else None,
            tie_word_embeddings=cfg.tie_word_embeddings,
            attention_dropout=cfg.dropout,
        )
    if cfg.arch == "mixtral":
        return cfg_cls(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.n_embd,
            intermediate_size=int(8 * cfg.n_embd / 3),
            num_hidden_layers=cfg.n_layer,
            num_attention_heads=cfg.n_head,
            num_key_value_heads=cfg.n_head,
            max_position_embeddings=cfg.seq_len,
            num_local_experts=cfg.n_experts,
            num_experts_per_tok=cfg.n_experts_per_tok,
            router_aux_loss_coef=cfg.moe_aux_loss_coef,
            tie_word_embeddings=cfg.tie_word_embeddings,
        )
    raise ValueError(f"Неизвестная архитектура: {cfg.arch}. Доступны: {sorted(ARCH_BY_NAME)}")


def build_model(cfg: ModelConfig, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """Случайно инициализированная модель transformers"""
    _, model_cls = ARCH_BY_NAME[cfg.arch]
    hf_cfg = build_hf_config(cfg)
    model = model_cls(hf_cfg)
    return model.to(device=device, dtype=dtype)


def num_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_m(model: torch.nn.Module) -> float:
    return num_parameters(model) / 1e6