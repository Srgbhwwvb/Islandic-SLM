"""Обучение transformers-моделей через стандартный Trainer + свои колбэки.

Данные: memmap uint16 (уже подготовлены scripts/06_prepare_bin.py).
Модель: случайная инициализация (from_config), НЕ предобученная.
Контроль: TensorBoard, валидация, few-shot пробы, чекпоинты, resume.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import Trainer, TrainerCallback, TrainingArguments

from .config import ExperimentConfig, save_config_json
from .model import build_model, num_parameters
from .tokenizer import load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TokenDataset(Dataset):
    def __init__(self, path: str | Path, seq_len: int, max_samples: int | None = None, max_tokens: int = 1_000_000_000):
        self.path = Path(path)
        self.seq_len = seq_len
        
        self.data = np.fromfile(self.path, dtype=np.uint16, count=max_tokens)
        self.n_tokens = len(self.data)
        
        max_possible_starts = max(self.n_tokens - self.seq_len - 1, 1)
        
        if max_samples is not None:
            self.max_start = min(max_possible_starts, max_samples)
        else:
            self.max_start = max_possible_starts

    def __len__(self) -> int:
        return self.max_start

    def __getitem__(self, idx: int) -> dict:
        start = torch.randint(0, self.max_start, (1,)).item()
        x = torch.from_numpy(self.data[start : start + self.seq_len].astype(np.int64))
        return {"input_ids": x, "labels": x.clone()}


from torch.utils.data import Dataset, IterableDataset

class StreamingTokenDataset(IterableDataset):
    """Читает случайные окна с диска, не загружая файл в RAM."""
    def __init__(self, path: str | Path, seq_len: int, num_samples: int = 500_000):
        self.path = Path(path)
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # Вычисляем общее количество токенов (размер файла в байтах / 2)
        self.file_size = self.path.stat().st_size
        self.total_tokens = self.file_size // 2
        
        if self.total_tokens <= self.seq_len:
            raise ValueError(f"Файл {path} слишком мал для seq_len={seq_len}")

    def __iter__(self):
        # Внутри каждого воркера DataLoader открываем файл заново
        worker_info = torch.utils.data.get_worker_info()
        
        # Если воркеров несколько, делим количество сэмплов между ними
        if worker_info is None:
            num_samples = self.num_samples
        else:
            num_samples = self.num_samples // worker_info.num_workers

        with open(self.path, "rb") as f:
            for _ in range(num_samples):
                start_token = random.randint(0, self.total_tokens - self.seq_len - 1)
                start_byte = start_token * 2

                f.seek(start_byte)
                chunk = f.read((self.seq_len + 1) * 2)
                
                tokens = np.frombuffer(chunk, dtype=np.uint16).astype(np.int64)
                
                if len(tokens) < self.seq_len + 1:
                    continue
                    
                x = torch.from_numpy(tokens)

                yield {"input_ids": x, "labels": x.clone()}

def load_prompts(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return ["Höfuðborg Íslands er ", "Hæsti tindur Íslands er ", "Vatnajökull er stærsti jökull "]
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.no_grad()
def generate_simple(model, tokenizer, prompt: str, max_new_tokens: int = 60, temperature=0.3, device: str = "cpu") -> str:
    model.eval()
    enc = tokenizer.encode(prompt)
    input_ids = torch.tensor([enc.ids], dtype=torch.long, device=device)
    
    vocab = tokenizer.get_vocab()
    bad_ids = [v for k, v in vocab.items() if k.startswith("<0x") or k in ["<unk>", "<pad>", "<bos>", "<mask>"]]
    
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        min_new_tokens=10,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.3,      
        no_repeat_ngram_size=2,      
        bad_words_ids=[[i] for i in bad_ids],
        pad_token_id=tokenizer.token_to_id("<pad>"),
        eos_token_id=tokenizer.token_to_id("<eos>"),
    )
    model.train()
    
    decoded = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    decoded = decoded.replace("�", "").replace("\ufffd", "")
    decoded = re.sub(r'([.,;:!?()"«»\-—]\s*){2,}', ' ', decoded)
    decoded = re.sub(r'\s+([.,;:!?])', r'\1', decoded)
    
    return decoded.strip()


@torch.no_grad()
def estimate_ppl(model, tokenizer, texts: list[str], device: str = "cpu") -> float:
    """Perplexity на списке текстов."""
    import torch.nn.functional as F

    model.eval()
    total_ll = 0.0
    n_toks = 0
    for text in texts:
        enc = tokenizer.encode(text)
        ids = torch.tensor([enc.ids], dtype=torch.long, device=device)[:, :2048]
        if ids.shape[1] < 2:
            continue
        logits = model(ids).logits[:, :-1]
        logp = F.log_softmax(logits, dim=-1)
        chosen = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        total_ll += chosen.sum().item()
        n_toks += ids.shape[1] - 1
    model.train()
    if n_toks == 0:
        return float("nan")
    return math.exp(-total_ll / n_toks)


class TrainingMonitorCallback(TrainerCallback):
    """Красивое и информативное логирование процесса обучения и генерации."""
    def __init__(self, tokenizer, model, prompts: list[str], gen_every: int = 1000,
                 max_new_tokens: int = 50, device: str = "cpu"):
        self.tokenizer = tokenizer
        self.model = model
        self.prompts = prompts
        self.gen_every = gen_every
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            if "loss" in logs:
                loss = logs.get("loss", 0.0)
                lr = logs.get("learning_rate", 0.0)
                grad_norm = logs.get("grad_norm", 0.0)
                
                elapsed = time.time() - self.start_time
                hours, rem = divmod(elapsed, 3600)
                minutes, seconds = divmod(rem, 60)
                
                print(f"\n[Step {state.global_step:>6}] "
                      f"Loss: {loss:.4f} | LR: {lr:.2e} | GradNorm: {grad_norm:.2f} | "
                      f"Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}", 
                      flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero and metrics is not None:
            eval_loss = metrics.get("eval_loss", 0.0)
            ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            print(f"[Eval  {state.global_step:>6}] Validation Loss: {eval_loss:.4f} | Perplexity: {ppl:.2f}", flush=True)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step and state.global_step % self.gen_every == 0:
            m = model or self.model
            m.eval()
            print("\n" + "="*60)
            print(f" ГЕНЕРАЦИЯ ТЕКСТА (Шаг {state.global_step})")
            print("="*60)
            for prompt in self.prompts:
                out = generate_simple(m, self.tokenizer, prompt,
                                      max_new_tokens=self.max_new_tokens, device=self.device)
                print(f"Промпт: {prompt!r}")
                print(f"Ответ : {out[:150]!r}\n")
            print("="*60 + "\n", flush=True)
            m.train()
        return control


def build_trainer(config: ExperimentConfig, device: str | None = "cpu",
                  max_steps_override: int | None = None, resume_from_checkpoint: str | None = None):
    """Собрать всё для Trainer: model, datasets, args, callbacks."""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    mcfg, tcfg = config.model, config.train
    if max_steps_override:
        tcfg.max_steps = max_steps_override
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_path = Path(tcfg.data_dir) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("vocab_size"):
            mcfg.vocab_size = int(meta["vocab_size"])
            print(f"[train] vocab_size из meta.json: {mcfg.vocab_size}")

    train_path = Path(tcfg.data_dir) / "train.bin"
    val_path = Path(tcfg.data_dir) / "val.bin"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Нет данных {train_path} / {val_path}. Сначала запустите скрипт подготовки бинов."
        )

    model = build_model(mcfg, device=device, dtype=torch.float32)
    print(f"[train] arch={mcfg.arch}, параметров: {num_parameters(model) / 1e6:.2f}M")
    if tcfg.grad_checkpointing:
        model.gradient_checkpointing_enable()

    train_ds = StreamingTokenDataset(train_path, mcfg.seq_len, num_samples=500_000)
    val_ds = TokenDataset(val_path, mcfg.seq_len, max_samples=500, max_tokens=20_000_000)

    use_bf16 = bool(tcfg.amp == "bf16" and device.startswith("cuda"))
    use_fp16 = bool(tcfg.amp == "fp16" and device.startswith("cuda"))

    try:
        import tensorboard
        report_to = "tensorboard"
    except ImportError:
        report_to = "none"

    save_steps = tcfg.save_every
    if tcfg.eval_every != save_steps:
        save_steps = tcfg.eval_every

    args = TrainingArguments(
        output_dir=tcfg.outdir,
        logging_dir=tcfg.log_dir,
        max_steps=tcfg.max_steps,
        per_device_train_batch_size=tcfg.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=tcfg.grad_accum,
        learning_rate=tcfg.lr,
        weight_decay=tcfg.weight_decay,
        warmup_steps=tcfg.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=tcfg.grad_clip,
        eval_strategy="steps",
        eval_steps=tcfg.eval_every,
        save_steps=save_steps,
        save_total_limit=tcfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(tcfg.eval_every, 1),
        report_to=report_to,
        fp16=use_fp16,
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        seed=tcfg.seed,
        disable_tqdm=True,
        log_level="info",
        do_eval=True,
        prediction_loss_only=True,
        include_num_input_tokens_seen=True,
    )

    tokenizer = load_tokenizer(tcfg.tokenizer)
    prompts = load_prompts(Path("isllm/prompts_is.txt"))
    callback = TrainingMonitorCallback(tokenizer=tokenizer, model=model, prompts=prompts,
                                       gen_every=tcfg.gen_every, device=device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[callback],
    )
    return trainer, model, config, resume_from_checkpoint


def train(config: ExperimentConfig, max_steps_override: int | None = None,
          log_every: int = 20, device: str | None = None,
          resume_from_checkpoint: str | None = None) -> tuple[dict, Trainer]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer, model, config, resume = build_trainer(
        config, device=device, max_steps_override=max_steps_override,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    Path(config.train.outdir).mkdir(parents=True, exist_ok=True)
    save_config_json(config, Path(config.train.outdir) / "experiment_config.json")

    t0 = time.time()
    result = None
    try:
        result = trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        print("\n[!] Обучение прервано пользователем. Сохранение текущего состояния...")
        interrupt_dir = Path(config.train.outdir) / "checkpoint_interrupted"
        interrupt_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(interrupt_dir)
        trainer.save_state()
        print(f"[!] Состояние сохранено в {interrupt_dir}")
    elapsed = time.time() - t0

    final_loss = None
    if result is not None:
        final_loss = getattr(result, "training_loss", None)
        
    metrics = {
        "best_val": None,
        "final_train_loss": float(final_loss) if final_loss is not None else None,
        "steps": trainer.state.global_step,
        "model_size_m": num_parameters(model) / 1e6,
        "device": device,
        "elapsed": elapsed,
    }
    return metrics, trainer