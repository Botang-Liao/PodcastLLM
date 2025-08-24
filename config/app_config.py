from __future__ import annotations
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    vector_dir: str = os.getenv("PODCAST_VECTOR_DIR", "./vector_store")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")
    use_cpu: bool = os.getenv("USE_CPU", "false").lower() == "true"
    k: int = int(os.getenv("TOP_K", "5"))
    fetch_k: int = int(os.getenv("FETCH_K", "100"))
    request_timeout_sec: int = int(os.getenv("OLLAMA_TIMEOUT", "120"))


def apply_overrides(cfg: AppConfig) -> AppConfig:
    return cfg