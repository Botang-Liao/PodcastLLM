from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AppSettings:
    parent_vector_dir: str = "/home/sylvia2004/projects/PodcastLLM/vector_store"
    model_name: str = "BAAI/bge-m3"
    use_cpu: bool = False
    top_k: int = 5
    fetch_k: int = 100
    ollama_model: str = "deepseek-r1:14b"
    ollama_url: str = "http://163.14.137.59:11434/api/generate"
    ollama_stream: bool = False
    ollama_timeout_sec: int = 120