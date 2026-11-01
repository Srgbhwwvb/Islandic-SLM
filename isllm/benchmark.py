"""Бенчмарк: perplexity, cloze-ранжирование фактов, MRR, согласованность парафраз."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .generation import logprob_of_tokens
from .tokenizer import load_tokenizer


@torch.no_grad()
def score_continuations(model, tokenizer, prefix: str, candidates: list[str],
                        device: str = "cpu") -> list[tuple[str, float, int]]:
    """Для каждого кандидата: средний logprob только токенов ответа.
    """
    model.eval()
    results = []
    prefix_ids = tokenizer.encode(prefix).ids
    for cand in candidates:
        cand_ids = tokenizer.encode(cand).ids
        if not prefix_ids or not cand_ids:
            results.append((cand, -1e9, 0))
            continue
        # сохраняем конец последовательности, если она длиннее окна
        full = (prefix_ids + cand_ids)[-getattr(model.config, "max_position_embeddings", 1024):]
        off = max(len(full) - len(cand_ids), 0)  # начало кандидата в окне
        start = max(off + 1, 1)  # первому токену кандидата нет предыдущего контекста
        if start >= len(full):
            results.append((cand, -1e9, 0))
            continue
        ids = torch.tensor([full], dtype=torch.long, device=device)
        logits = model(ids).logits[:, :-1]
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp[:, start - 1:len(full) - 1].gather(-1, ids[:, start:].unsqueeze(-1)).squeeze(-1)
        mean_logp = chosen.mean().item()
        results.append((cand, mean_logp, 0))
    results.sort(key=lambda r: r[1], reverse=True)
    results = [(c, lp, i + 1) for i, (c, lp, _) in enumerate(results)]
    return results


@torch.no_grad()
def perplexity(model, tokenizer, texts: list[str], device: str = "cpu") -> float:
    model.eval()
    total_ll = 0.0
    n_toks = 0
    for text in texts:
        enc = tokenizer.encode(text)
        ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
        max_len = getattr(model.config, "max_position_embeddings", 1024)
        if ids.shape[1] > max_len:
            ids = ids[:, -max_len:]
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits[:, :-1]
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        total_ll += chosen.sum().item()
        n_toks += ids.shape[1] - 1
    if n_toks == 0:
        return float("nan")
    return math.exp(-total_ll / n_toks)


def load_facts(path: str | Path = "isllm/facts_is.json") -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["facts"]


def run_cloze_benchmark(model, tokenizer, facts: list[dict], device: str = "cpu") -> dict:
    """accuracy@1/@5, MRR, согласованность между парафразами."""
    acc1 = 0
    acc5 = 0
    mrr = 0.0
    n_items = 0
    consistency = 0
    per_fact = []
    for fact in facts:
        correct = fact["answer"]
        candidates = fact["candidates"]
        formulations = fact["formulations"]
        answers = []
        for f in formulations:
            scores = score_continuations(model, tokenizer, f["prefix"], candidates, device)
            rank = next((r for c, _, r in scores if c == correct), 99)
            answers.append(rank)
            if rank == 1:
                acc1 += 1
            if rank <= 5:
                acc5 += 1
            mrr += 1.0 / rank
            n_items += 1
        all_correct = all(a == 1 for a in answers)
        if all_correct:
            consistency += 1
        per_fact.append({
            "fact": fact["fact"],
            "correct": correct,
            "ranks": answers,
            "consistent": all_correct,
        })
    return {
        "n_items": n_items,
        "accuracy@1": acc1 / max(n_items, 1),
        "accuracy@5": acc5 / max(n_items, 1),
        "MRR": mrr / max(n_items, 1),
        "consistency": consistency / max(len(facts), 1),
        "per_fact": per_fact,
    }


def format_report(results: dict) -> str:
    lines = [
        "# Benchmark report",
        "",
        f"- Items: {results['n_items']}",
        f"- accuracy@1: {results['accuracy@1']:.3f}",
        f"- accuracy@5: {results['accuracy@5']:.3f}",
        f"- MRR: {results['MRR']:.3f}",
        f"- consistency (все парафразы верны): {results['consistency']:.3f}",
        "",
    ]
    for item in results["per_fact"]:
        lines.append(
            f"- {item['fact']}: {item['correct']} | ranks={item['ranks']} | "
            f"{'СОГЛАСОВАНО' if item['consistent'] else 'нет'}"
        )
    return "\n".join(lines)