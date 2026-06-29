from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KALEIDO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Backend: "stub" runs entirely on CPU with no model downloads.
    backend: Literal["stub", "vllm"] = "stub"

    # Database (async SQLAlchemy URL).
    database_url: str = "postgresql+asyncpg://kaleido:kaleido@localhost:5432/kaleido"

    # vLLM inference server.
    vllm_base_url: str = "http://localhost:8000/v1"
    scorer_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Embedding model (sentence-transformers hub name).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Sparse Facet Activation parameters.
    gate_top_k: int = 64
    gate_threshold: float = 0.35  # cosine similarity floor for "observable" facets

    # Confidence / abstention.
    # Stub backend produces confidence ~0.20 (uniform logprobs); set tau below that
    # so scores are returned in CPU-demo mode. vLLM mode will see higher confidence.
    abstain_tau: float = 0.10  # scores below this confidence are flagged for review
    self_consistency_samples: int = 3  # extra stochastic samples for consistency signal
    self_consistency_temperature: float = 0.7

    # Registry.
    registry_version: str = "2026.06.0"
    facets_csv_path: str = "data/processed/facets_enriched.csv"

    # Logging.
    log_level: str = "INFO"

    # Redis (optional; disabled by default so docker-compose.stub works).
    use_redis: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # Confidence fusion weights (must sum to ~1).
    conf_weight_logprob: float = Field(default=0.5, ge=0.0, le=1.0)
    conf_weight_consistency: float = Field(default=0.3, ge=0.0, le=1.0)
    conf_weight_ordinal_var: float = Field(default=0.2, ge=0.0, le=1.0)


# Module-level singleton; override by passing Settings() explicitly in tests.
settings = Settings()
