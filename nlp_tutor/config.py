# nlp_tutor/config.py
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

PATHS = Paths(
    root_dir=ROOT_DIR,
    data_dir=DATA_DIR,
    raw_dir=DATA_DIR / "raw",
    processed_dir=DATA_DIR / "processed",
    models_dir=DATA_DIR / "models",
)

RESOURCES_DIR = ROOT_DIR / "nlp_tutor" / "resources"
RESOURCE_MODELS_DIR = RESOURCES_DIR / "models"
RESOURCE_LM_DIR = RESOURCES_DIR / "lm"


def resolve_lesson_bank_path() -> Path:
    env_path = os.environ.get("NLP_TUTOR_LESSON_BANK")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    resource_path = RESOURCES_DIR / "lesson_bank.csv"
    if resource_path.exists():
        return resource_path

    return PATHS.processed_dir / "lesson_bank.csv"


def resolve_model_path(filename: str) -> Path:
    env_dir = os.environ.get("NLP_TUTOR_MODELS_DIR")
    if env_dir:
        candidate = Path(env_dir) / filename
        if candidate.exists():
            return candidate

    resource_candidate = RESOURCE_MODELS_DIR / filename
    if resource_candidate.exists():
        return resource_candidate

    return PATHS.models_dir / filename

