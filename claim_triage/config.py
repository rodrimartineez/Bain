"""Centralised configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines) so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    # Text model: coordinator decisions, fact extraction, assessment narrative.
    # qwen2.5:3b fits fully in a 4 GB GPU next to other apps; it is used for all text-only calls.
    text_model: str = field(default_factory=lambda: _env("TEXT_MODEL", "qwen2.5:3b"))
    # Vision model: must accept images (gemma3 does; qwen2.5 does not).
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "gemma3:4b"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "nomic-embed-text"))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "180")))
    # Vision calls are much slower when the model does not fit entirely in GPU memory.
    vision_timeout_s: float = field(default_factory=lambda: float(_env("VISION_TIMEOUT_S", "600")))
    # GPU layers for the vision model. Empty = Ollama decides. "0" = CPU only, which is faster than a
    # partial GPU split when the GPU is shared with other apps (measured: 66 s vs >10 min per photo).
    vision_num_gpu: str = field(default_factory=lambda: _env("VISION_NUM_GPU", ""))
    llm_keep_alive: str = field(default_factory=lambda: _env("LLM_KEEP_ALIVE", "30m"))

    # "llm": the model picks the next tool; "rules": deterministic planner (no LLM for decisions).
    coordinator_mode: str = field(default_factory=lambda: _env("COORDINATOR_MODE", "llm"))
    max_react_steps: int = field(default_factory=lambda: int(_env("MAX_REACT_STEPS", "12")))
    max_searches: int = 2

    # Retrieval
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "5")))
    # Cosine similarity below which a historical claim is not considered relevant.
    min_similarity: float = field(default_factory=lambda: float(_env("MIN_SIMILARITY", "0.63")))

    # Intake limits
    max_images: int = 5
    max_documents: int = 5
    max_file_mb: float = 10.0
    min_description_chars: int = 20
    max_document_chars: int = 6000

    # Paths
    kb_path: Path = ROOT_DIR / "data" / "knowledge_base" / "historical_claims.json"
    index_dir: Path = ROOT_DIR / "data" / "index"
    demo_dir: Path = ROOT_DIR / "data" / "demo"
    runs_dir: Path = ROOT_DIR / "output" / "runs"
    upload_dir: Path = ROOT_DIR / "output" / "uploads"


settings = Settings()
