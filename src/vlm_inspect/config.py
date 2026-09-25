"""Runtime settings, overridable with VLM_INSPECT_* environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLM_INSPECT_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    results_dir: Path = Path("results")

    # Vision-language model
    vlm_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    # Longest image side fed to the VLM. Visual tokens grow with the square of this, and so does
    # CPU latency; small defects vanish if it is too low. Measured trade-off in DESIGN_NOTES.
    vlm_max_side: int = 768
    vlm_threshold: float = 0.5
    vlm_max_new_tokens: int = 160

    # Inspection service
    database_url: str = "sqlite:///./vlm_inspect.db"
    db_auto_create: bool = False  # create tables at startup (tests/dev); production uses Alembic
    api_methods: list[str] = ["stub"]  # first entry is the default method
    calibration_dir: Path = Path("results")  # <method>/<part>/calibration.json from the benchmark
    max_upload_bytes: int = 25 * 1024 * 1024

    # Trained detector
    yolo_weights: Path = Path("models/yolo/best.pt")
    yolo_imgsz: int = 1024
    yolo_threshold: float = 0.25


def get_settings() -> Settings:
    return Settings()
