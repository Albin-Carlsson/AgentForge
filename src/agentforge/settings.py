"""Environment settings, as distinct from experiment configuration.

`RunConfig` in config.py holds the knobs that affect measured outputs and get
hashed. This module holds where things live: DSNs, credentials, cache paths.

Nothing here has a silent default that could mask a misconfiguration. An absent
DSN raises at startup rather than quietly connecting to localhost and producing
numbers against the wrong database.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentforge.errors import ConfigError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="", extra="ignore", frozen=True
    )

    pg_dsn: str = Field(alias="AF_PG_DSN")
    model_dir: Path = Field(default=Path("./models"), alias="AF_MODEL_DIR")

    # Required from M3 onward. M1 and M2 run without it, so it is optional here
    # and the LLM client raises if it is reached while unset -- rather than this
    # module blocking the key-free retrieval tier.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Groq serves the open-weight models (see config.MODEL_PROVIDERS). Optional
    # for the same reason: only a call to a Groq-served model needs it.
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    langfuse_public_key: str | None = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    def require_anthropic_key(self) -> str:
        if not self.anthropic_api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is unset. Milestones M1 and M2 (ingestion and the "
                "retrieval tier) need no key; M3 onward do."
            )
        return self.anthropic_api_key

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache(maxsize=1)
def settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        raise ConfigError(
            f"invalid environment: {exc}. Copy .env.example to .env and fill it in."
        ) from exc
