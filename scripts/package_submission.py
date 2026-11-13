"""Packaging script to create clean submission and review archives for The Relay.

Strictly excludes:
- .env and all environment/credential files (security hygiene)
- .venv/ virtual environment directories
- .pytest_cache/, __pycache__/, and bytecode artifacts
- Large corpus PDFs and local scratch directories
"""

import fnmatch
from pathlib import Path
import sys
import zipfile

# Patterns strictly excluded from any archive packaging
EXCLUDE_PATTERNS = [
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    ".venv*",
    "venv*",
    "__pycache__*",
    "*.pyc",
    "*.pyo",
    ".pytest_cache*",
    ".mypy_cache*",
    ".ruff_cache*",
    "data/corpus/pdfs/*",
    "data/backup_offline_benchmarks*",
    "scratch*",
    "tmp*",
    "*.zip",
    "dist/*",
    ".git*",
    ".agents*",
]


def should_exclude(rel_path_str: str) -> bool:
    """Return True if relative path matches any exclusion rule."""
    normalized = rel_path_str.replace("\\", "/")
    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(Path(normalized).name, pattern):
            return True
        # Check parent folder matches
        parts = normalized.split("/")
        if any(fnmatch.fnmatch(p, pattern.rstrip("/*")) for p in parts):
            return True
    return False


def create_submission_archive(
    root_dir: Path, output_zip_path: Path
) -> None:
    """Package project files into a zip archive with strict exclusion enforcement."""
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    included_count = 0
    excluded_count = 0

    print(f"Creating clean review archive: {output_zip_path}")
    print(f"Project root: {root_dir.resolve()}")

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(root_dir.rglob("*")):
            if file_path.is_dir():
                continue

            rel_path = file_path.relative_to(root_dir)
            rel_str = str(rel_path)

            if should_exclude(rel_str):
                excluded_count += 1
                continue

            zf.write(file_path, arcname=rel_str)
            included_count += 1

    print(f"Archive created successfully!")
    print(f"Included files: {included_count}")
    print(f"Excluded files / sensitive paths: {excluded_count}")
    print(f"Archive size: {output_zip_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    zip_dest = project_root / "dist" / "The_Relay_Submission.zip"
    create_submission_archive(project_root, zip_dest)

