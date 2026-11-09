"""Configuration management for Smriti Agent Memory."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """Lightweight .env loader without external dependency."""
    for candidate in [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]:
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k and k not in os.environ and v:
                    os.environ[k] = v


_load_dotenv()


@dataclass
class SmritiConfig:
    """Runtime and architectural hyperparameters for Smriti."""

    db_path: str = field(
        default_factory=lambda: os.environ.get("SMRITI_DB_PATH", "artifacts/smriti_memory.db")
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.environ.get(
            "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")
        )
    )
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    openai_base_url: str = field(
        default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    llm_model: str = field(
        default_factory=lambda: os.environ.get("SMRITI_LLM_MODEL", "gemini-3.1-flash-lite")
    )
    embedding_dim: int = 384
    token_budget: int = field(
        default_factory=lambda: int(os.environ.get("SMRITI_TOKEN_BUDGET", "1500"))
    )
    global_header_max_tokens: int = 220
    wiki_budget_fraction: float = 0.25
    claims_budget_fraction: float = 0.40
    episodic_turns_budget_fraction: float = 0.35
    enable_dreaming: bool = True
    enable_temporal_anchoring: bool = True
    enable_supersession: bool = True
    enable_hybrid_bm25: bool = True
    enable_entity_graph: bool = True
    rrf_k: int = 60
    temperature: float = 0.0
