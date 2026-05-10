"""
Shared path discovery helpers for CLI and Python API entrypoints.
"""
from __future__ import annotations

from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".aac",
    ".m4a",
    ".opus",
    ".aiff",
    ".aif",
    ".mp4",
    ".mkv",
    ".mov",
}


def collect_files(path: Path) -> list[Path]:
    """Collect supported audio/video-container files from a file or directory path."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        raise ValueError(f"Unsupported file type: {path.suffix}")
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Path not found: {path}")


def collect_requested_files(paths: list[Path]) -> list[Path]:
    """
    Resolve one or more user-requested paths into a flat list of supported files.

    Directly requested unsupported files and missing paths raise immediately instead
    of being skipped silently.
    """
    files: list[Path] = []
    for path in paths:
        files.extend(collect_files(path))
    return files
