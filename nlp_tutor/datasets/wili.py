from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from nlp_tutor.corpora import TextLabelDataset
from nlp_tutor.config import PATHS


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def load_wili_2018(split: str = "train") -> TextLabelDataset:
    base = PATHS.raw_dir / "wili-2018" / "versions" / "1"
    if not base.exists():
        raise FileNotFoundError(f"WiLI base folder not found: {base}")

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")

    x_path = base / f"x_{split}.txt"
    y_path = base / f"y_{split}.txt"
    labels_path = base / "labels.csv"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing WiLI files: {x_path} / {y_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing WiLI labels.csv: {labels_path}")

    texts = _read_lines(x_path)
    y_ids = _read_lines(y_path)

    # Build id -> label mapping
    df = pd.read_csv(labels_path, dtype=str)

    # If it parsed as a single column, it's almost certainly semicolon-delimited.
    if len(df.columns) < 2:
        df = pd.read_csv(labels_path, sep=";", dtype=str, engine="python")

    if len(df.columns) < 2:
        raise ValueError(
            f"labels.csv should have >=2 columns. Found: {list(df.columns)}"
        )

    # WiLI labels.csv columns typically include: "Label", "English", ...
    # Use "Label" -> "English" if available; otherwise fallback to first two columns.
    if "Label" in df.columns and "English" in df.columns:
        id_col, name_col = "Label", "English"
    else:
        id_col, name_col = df.columns[0], df.columns[1]
    id_to_label: Dict[str, str] = {
        str(row[id_col]): str(row[name_col]) for _, row in df.iterrows()
    }

    labels = [id_to_label.get(str(i), str(i)) for i in y_ids]

    if len(texts) != len(labels):
        raise ValueError(f"WiLI split mismatch: len(texts)={len(texts)} len(labels)={len(labels)}")

    return TextLabelDataset(texts=texts, labels=labels)
