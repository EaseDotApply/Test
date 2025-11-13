"""Application configuration powered by Pydantic settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Runtime application settings.

    The settings object centralises configuration for service components, favouring
    environment variables with sensible defaults so the service is plug-and-play.
    """

    model_config = SettingsConfigDict(env_prefix="QA_", env_file=(".env", ".env.local"))

    environment: Literal["local", "staging", "production"] = Field(
        "local", description="Deployment environment label used for logging and metrics.",
    )
    api_host: str = Field("0.0.0.0", description="Hostname FastAPI should bind to.")
    api_port: int = Field(8000, description="Port FastAPI should listen on.")
    api_workers: int = Field(1, description="Number of Uvicorn workers for production runs.")

    messages_api_url: AnyHttpUrl = Field(  # type: ignore[assignment]
        "https://november7-730026606190.europe-west1.run.app/messages",
        description="Public endpoint serving member messages.",
    )
    request_page_size: int = Field(
        200,
        description="Page size to request from the upstream API; tuned to minimise round trips.",
        ge=1,
        le=500,
    )
    request_timeout_seconds: float = Field(10.0, description="HTTPX timeout when calling upstream.")
    request_retry_attempts: int = Field(5, description="Maximum retry attempts for transient errors.")

    data_dir: Path = Field(Path("data"), description="Directory for raw and processed data caches.")
    vectorstore_dir: Path = Field(Path("vectorstore"), description="Directory for persisted vectorstores.")
    reports_dir: Path = Field(Path("reports"), description="Directory holding evaluation reports.")

    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model identifier for embeddings.",
    )
    llm_model: str = Field(
        "mistral:instruct",
        description=(
            "Identifier for the open-source generation model served via Ollama or llama.cpp compatible server.",
        ),
    )
    llm_api_base: str | None = Field(
        default=None,
        description="Optional base URL for an HTTP server exposing the OSS LLM (e.g. Ollama REST).",
    )
    llm_temperature: float = Field(0.2, description="Sampling temperature for the answer model.")

    enable_tracing: bool = Field(
        False,
        description="Enable OpenTelemetry tracing exporters when configured.",
    )
    hallucination_model: str = Field(
        "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        description="Sequence classification model used for hallucination detection.",
    )
    hallucination_threshold: float = Field(
        0.55,
        ge=0.0,
        le=1.0,
        description="Minimum entailment probability required to consider an answer supported.",
    )

    @property
    def raw_messages_path(self) -> Path:
        """Absolute path to cached raw messages."""

        return self.data_dir / "raw" / "messages.parquet"

    @property
    def processed_messages_path(self) -> Path:
        """Absolute path to processed message dataset."""

        return self.data_dir / "processed" / "messages.parquet"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return a cached settings instance.

    `lru_cache` ensures FastAPI dependency injection produces a singleton instance while
    keeping the object testable by clearing the cache during fixtures.
    """

    settings = AppSettings()
    for path in (settings.data_dir, settings.vectorstore_dir, settings.reports_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = ["AppSettings", "get_settings"]
