from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


DATASET_REF = "mexwell/wili-2018"
ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_ROOT = ROOT_DIR / "data" / "raw" / "wili-2018"
REQUIRED_FILES = [
    Path("versions/1/x_train.txt"),
    Path("versions/1/y_train.txt"),
    Path("versions/1/x_test.txt"),
    Path("versions/1/y_test.txt"),
    Path("versions/1/labels.csv"),
]


def _has_required_files(root: Path) -> bool:
    return all((root / rel_path).exists() for rel_path in REQUIRED_FILES)


def _find_wili_root(download_dir: Path) -> Path:
    if _has_required_files(download_dir):
        return download_dir

    for path in sorted(download_dir.rglob("*")):
        if path.is_dir() and _has_required_files(path):
            return path

    raise FileNotFoundError(
        "Could not locate the WiLI-2018 dataset structure containing "
        "`versions/1/x_train.txt` inside the Kaggle download directory."
    )


def _copy_tree(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    for item in source_root.iterdir():
        target = target_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main() -> None:
    download_dir = Path(kagglehub.dataset_download(DATASET_REF))
    source_root = _find_wili_root(download_dir)

    _copy_tree(source_root, TARGET_ROOT)

    print(f"Downloaded {DATASET_REF}")
    print(f"Source: {source_root}")
    print(f"Saved to: {TARGET_ROOT}")


if __name__ == "__main__":
    main()
