from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET_REF = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT_DIR / "data" / "raw" / "imdb" / "IMDB Dataset.csv"


def _find_imdb_csv(download_dir: Path) -> Path:
    preferred = sorted(download_dir.rglob("IMDB Dataset.csv"))
    if preferred:
        return preferred[0]

    csv_files = sorted(download_dir.rglob("*.csv"))
    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        f"Could not locate the IMDB CSV inside {download_dir}. "
        f"Found {len(csv_files)} CSV files."
    )


def main() -> None:
    download_dir = Path(kagglehub.dataset_download(DATASET_REF))
    source_csv = _find_imdb_csv(download_dir)

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, TARGET_PATH)

    print(f"Downloaded {DATASET_REF}")
    print(f"Source: {source_csv}")
    print(f"Saved to: {TARGET_PATH}")


if __name__ == "__main__":
    main()
