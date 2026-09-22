"""Persistent reviewer annotations and accept/reject decisions.

State lives in one JSON file keyed by file path. Every mutation re-reads the
file before writing, so two ``ReviewStore`` instances (or processes) pointed at
the same file add to each other's work instead of overwriting it. A store file
that cannot be parsed raises ``ValueError`` rather than being replaced, because
review state is human work that should never be discarded silently.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

from .version import __version__

REVIEW_DECISIONS = ("accept", "reject", "needs-review")
REVIEW_EXPORT_FORMAT = "qualiax-review"
REVIEW_EXPORT_VERSION = 1


class ReviewStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = self._load()

    def annotate(
        self,
        file: str,
        text: str,
        *,
        reviewer: str,
        segment_index: int | None = None,
    ) -> dict:
        """Add a note to a file, or to one of its segments when ``segment_index`` is set."""
        reviewer = _require_text(reviewer, "reviewer")
        text = _require_text(text, "annotation text")
        if segment_index is not None and (isinstance(segment_index, bool) or not isinstance(segment_index, int) or segment_index < 0):
            raise ValueError("segment_index must be a non-negative integer or None")
        annotation = {
            "text": text,
            "reviewer": reviewer,
            "segment_index": segment_index,
            "timestamp": time.time(),
        }

        def mutate(data: dict) -> None:
            _entry(data, file)["annotations"].append(annotation)

        self._update(mutate)
        return dict(annotation)

    def decide(self, file: str, decision: str, *, reviewer: str, note: str | None = None) -> dict:
        """Record accept / reject / needs-review. Earlier decisions stay in ``decision_history``."""
        if decision not in REVIEW_DECISIONS:
            raise ValueError("decision must be accept, reject, or needs-review")
        reviewer = _require_text(reviewer, "reviewer")
        record = {"decision": decision, "reviewer": reviewer, "timestamp": time.time(), "note": note}

        def mutate(data: dict) -> None:
            entry = _entry(data, file)
            entry["decision"] = decision
            entry["decision_reviewer"] = reviewer
            entry["decision_timestamp"] = record["timestamp"]
            entry.setdefault("decision_history", []).append(record)

        self._update(mutate)
        return dict(record)

    def for_file(self, file: str) -> dict:
        return deepcopy(self._data.get(file, {"annotations": [], "decision": None}))

    def segment_annotations(self, file: str, segment_index: int) -> list[dict]:
        return [
            dict(item)
            for item in self._data.get(file, {}).get("annotations", [])
            if item.get("segment_index") == segment_index
        ]

    def files(self, decision: str | None = None) -> list[str]:
        """List reviewed files, optionally only those with ``decision``.

        ``decision="undecided"`` lists files that have annotations but no decision.
        """
        if decision is None:
            return sorted(self._data)
        wanted = None if decision == "undecided" else decision
        if wanted is not None and wanted not in REVIEW_DECISIONS:
            raise ValueError("decision must be accept, reject, needs-review, or undecided")
        return sorted(name for name, entry in self._data.items() if entry.get("decision") == wanted)

    def summary(self) -> dict:
        counts = {name: 0 for name in (*REVIEW_DECISIONS, "undecided")}
        for entry in self._data.values():
            key = entry.get("decision") or "undecided"
            counts[key] = counts.get(key, 0) + 1
        return {"files": len(self._data), "decisions": counts}

    def reload(self) -> None:
        self._data = self._load()

    def export(self) -> dict:
        return json.loads(json.dumps(self._data))

    def export_json(self, output: str | Path) -> Path:
        """Write a self-describing, versioned copy of the review state for other tools."""
        destination = Path(output)
        if destination.expanduser().resolve(strict=False) == self.path.expanduser().resolve(strict=False):
            raise ValueError("export_json must not overwrite the review store itself")
        document = {
            "format": REVIEW_EXPORT_FORMAT,
            "version": REVIEW_EXPORT_VERSION,
            "tool_version": __version__,
            "exported_at": time.time(),
            "summary": self.summary(),
            "files": self.export(),
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    def _update(self, mutate) -> None:
        data = self._load()
        mutate(data)
        self._save(data)
        self._data = data

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Review store {self.path} is unreadable ({exc}); refusing to overwrite it.") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Review store {self.path} must contain a JSON object; refusing to overwrite it.")
        return payload

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)


def _entry(data: dict, file: str) -> dict:
    entry = data.setdefault(file, {"annotations": [], "decision": None})
    entry.setdefault("annotations", [])
    entry.setdefault("decision", None)
    return entry


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()
