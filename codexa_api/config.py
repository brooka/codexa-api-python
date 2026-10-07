"""Settings, from environment variables or api-python/.env."""

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODEXA_", extra="ignore")

    # packs.json and mobile/*.db
    data_dir: Path = ROOT / "data"
    # the EmbeddingGemma ONNX files the packs were embedded with
    model_dir: Path = ROOT / "models" / "embeddinggemma-300m-ONNX"
    # load only these packs (comma-separated work ids); empty loads every pack in packs.json
    packs: str = ""
    llm_model: str = "gemini-3.1-flash-lite"
    ask_sources: int = 8
    gemini_api_key: str | None = Field(default=None, validation_alias="GEMINI_API_KEY")
    # when set, /ask requires it in an X-API-Key header (it spends Gemini credit; /search doesn't)
    ask_key: str | None = None

    @property
    def pack_ids(self) -> list[str] | None:
        return [p.strip() for p in self.packs.split(",") if p.strip()] or None


@lru_cache
def get_settings() -> Settings:
    load_dotenv()  # also exposes LANGFUSE_* to the Langfuse SDK, which reads os.environ
    return Settings()
