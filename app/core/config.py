"""
La Morph ML Service — Core configuration.

Reads settings from environment variables with sensible development defaults.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # lamorph-ml-service/


class Settings(BaseSettings):
    """Application-wide settings, loaded from env vars / .env file."""

    # ── Server ──────────────────────────────────────────────────────────
    app_name: str = "La Morph ML Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=True)
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Model paths ─────────────────────────────────────────────────────
    models_dir: Path = Field(default=_ROOT_DIR / "models")

    # Wrinkle U-Net segmentation model (input: 4×1024×1024, output: 2×1024×1024)
    wrinkle_model_file: str = "wrinkle_unet.int8.onnx"
    wrinkle_model_hash: str = "c60324480bd40e31bbcf6c81ce3d501e8367a4d50d69eb883e76d9dc7557b61f"

    # Eye-bag classification model (input: 3×160×256, output: presence + severity)
    eyebag_model_file: str = "model.fp16.onnx"
    eyebag_model_hash: str = "69c1d1a960c53c59d7f0a30443246dd9e09f18af551cf455b1336fb7d00d63a1"

    # ── ONNX Runtime ────────────────────────────────────────────────────
    ort_intra_op_threads: int = 4
    ort_inter_op_threads: int = 1

    # ── Security ────────────────────────────────────────────────────────
    # Shared secret the NestJS backend sends in the X-Internal-Token header.
    internal_api_token: str = Field(default="dev-internal-token")

    # ── Scanner metadata ────────────────────────────────────────────────
    scanner_version: str = "scanner-pipeline-1.0.0"
    model_bundle_version: str = "lamorph-scanner-bundle-1.0.0"
    observation_schema_version: str = "skin-observation-schema-1.0.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def wrinkle_model_path(self) -> Path:
        return self.models_dir / self.wrinkle_model_file

    @property
    def eyebag_model_path(self) -> Path:
        return self.models_dir / self.eyebag_model_file


settings = Settings()
