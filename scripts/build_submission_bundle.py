from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Set

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INCLUDE = [
    "api",
    "nlp_tutor",
    "scripts",
    "tests",
    "web",
    "reports",
    "nlp_tutor/resources",
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "Makefile",
]

EXCLUDE_DIR_NAMES: Set[str] = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".idea",
    "node_modules",
    "dist",
    "build",
    "nlp_adaptive_tutor.egg-info",
}

EXCLUDE_FILES: Set[str] = {
    "server.log",
}

EXCLUDE_EXTS: Set[str] = {
    ".pyc",
    ".pyo",
    ".log",
}


def should_exclude(rel_path: Path) -> bool:
    parts = set(rel_path.parts)
    if parts & EXCLUDE_DIR_NAMES:
        return True
    if rel_path.as_posix().startswith("data/raw"):
        return True
    if rel_path.name in EXCLUDE_FILES:
        return True
    if rel_path.suffix in EXCLUDE_EXTS:
        return True
    return False


def iter_source_files(src: Path, base: Path) -> Iterable[Path]:
    if src.is_file():
        rel = src.relative_to(base)
        if not should_exclude(rel):
            yield src
        return

    for path in src.rglob("*"):
        if path.is_file():
            rel = path.relative_to(base)
            if not should_exclude(rel):
                yield path


def copy_tree(include_paths: Iterable[str], out_dir: Path) -> None:
    for inc in include_paths:
        src = ROOT / inc
        if not src.exists():
            continue
        for path in iter_source_files(src, ROOT):
            rel = path.relative_to(ROOT)
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def write_manifest(out_dir: Path, include_paths: Iterable[str]) -> None:
    manifest = out_dir / "SUBMISSION.txt"
    lines = [
        "NLP Adaptive Tutor submission bundle",
        "",
        "Included paths:",
    ]
    for inc in include_paths:
        lines.append(f"- {inc}")
    lines.extend([
        "",
        "Excluded:",
        "- data/raw (training datasets)",
        "- virtualenvs, caches, and build artifacts",
    ])
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_zip(src_dir: Path, zip_path: Path, bundle_root: str) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                arcname = Path(bundle_root) / file_path.relative_to(src_dir)
                zf.write(file_path, arcname.as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a minimal submission bundle without raw datasets.")
    parser.add_argument(
        "--include-backups",
        action="store_true",
        help="Include data/models_backup (optional models).",
    )
    args = parser.parse_args()

    include_paths = list(DEFAULT_INCLUDE)
    if args.include_backups:
        include_paths.append("data/models_backup")

    out_dir = ROOT / "dist" / "submission"
    zip_path = ROOT / "dist" / "nlp-adaptive-tutor-submission.zip"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(include_paths, out_dir)
    write_manifest(out_dir, include_paths)
    build_zip(out_dir, zip_path, "nlp-adaptive-tutor")

    print(f"Bundle created at {zip_path}")


if __name__ == "__main__":
    main()
