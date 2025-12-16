from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    root_dir: Path
    data_dir: Path
    raw_dir: Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

PATHS = Paths(
    root_dir=ROOT_DIR,
    data_dir=DATA_DIR,
    raw_dir=DATA_DIR / "raw",
)
