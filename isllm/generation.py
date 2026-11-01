"""Инференс для transformers-моделей: загрузка чекпоинта и генерация.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .config import ModelConfig
from .model import build_hf_config, ARCH_BY_NAME


def load_model_from_dir(model_dir: str, device: str = "cpu",
                        dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """Загрузить модель из директории, сохранённой Trainer.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_model(ckpt: str, device: str = "cpu") -> torch.nn.Module:
    return load_model_from_dir(ckpt, device=device)


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k: int | None = 40, top_p: float | None = 0.95,
             do_sample: bool = True, seed: int | None = None,
             device: str = "cpu", eos_token_id: int | None = None) -> str:
    """Greedy / top-k / top-p генерация"""
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
    enc = tokenizer.encode(prompt)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    max_len = getattr(model.config, "max_position_embeddings", 1024)
    if ids.shape[1] > max_len:
        ids = ids[:, -max_len:]
    for _ in range(max_new_tokens):
        logits = model(ids).logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs, dim=-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            sorted_logits[remove] = -float("inf")
            logits = torch.scatter(logits, -1, sorted_idx, sorted_logits)
        if do_sample:
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        else:
            nxt = logits.argmax(dim=-1, keepdim=True)
        if eos_token_id is not None and nxt.item() == eos_token_id:
            break
        ids = torch.cat([ids, nxt], dim=1)
    return tokenizer.decode(ids[0].tolist())


@torch.no_grad()
def logprob_of_tokens(model, tokenizer, text: str, device: str = "cpu") -> float:
    """Средний logprob токенов."""
    enc = tokenizer.encode(text)
    ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    max_len = getattr(model.config, "max_position_embeddings", 1024)
    if ids.shape[1] > max_len:
        ids = ids[:, -max_len:]
    logits = model(ids).logits[:, :-1]
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return chosen.mean().item()