from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 80
    # 800/80 800/150 1000/150
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    CHROMA_PATH: str = "chroma_db"

    OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
