"""Application settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

NOT_FOUND_MESSAGE = "I don't know. That information is not available in the handbook."


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    handbook_path: Path = field(
        default_factory=lambda: Path(_env("HANDBOOK_PATH", str(PROJECT_ROOT / "data" / "handbook.pdf")))
    )
    website_url: str = field(default_factory=lambda: _env("WEBSITE_URL", "https://www.zaio.io/"))
    website_max_pages: int = field(default_factory=lambda: int(_env("WEBSITE_MAX_PAGES", "80")))
    website_snapshot: Path = field(
        default_factory=lambda: Path(
            _env("WEBSITE_SNAPSHOT", str(PROJECT_ROOT / "data" / "website_pages.json"))
        )
    )
    chroma_dir: Path = field(
        default_factory=lambda: Path(_env("CHROMA_DIR", str(PROJECT_ROOT / "chroma_db")))
    )
    collection_name: str = field(default_factory=lambda: _env("COLLECTION_NAME", "handbook"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    min_similarity: float = field(default_factory=lambda: float(_env("MIN_SIMILARITY", "0.25")))
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-3.5-flash-lite"))


def get_settings() -> Settings:
    return Settings()
