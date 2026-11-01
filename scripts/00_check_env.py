"""Проверка окружения: GPU, VRAM, bf16, ключевые пакеты."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("Environment check")
    print("=" * 60)
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"capability: {torch.cuda.get_device_capability(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"bf16_supported: {torch.cuda.is_bf16_supported()}")
        print(f"flash_attn_2 (через SDPA): {torch.backends.cuda.flash_sdp_enabled()}")
    else:
        print("GPU не найден — обучение пойдёт на CPU (медленно).")

    for mod in ["tokenizers", "numpy", "matplotlib", "tensorboard", "yaml"]:
        try:
            m = __import__(mod)
            print(f"{mod}: {getattr(m, '__version__', 'ok')}")
        except ImportError:
            print(f"{mod}: НЕ УСТАНОВЛЕН")
    print("=" * 60)


if __name__ == "__main__":
    main()