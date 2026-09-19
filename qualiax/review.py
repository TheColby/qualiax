"""Persistent reviewer annotations and accept/reject decisions."""
from __future__ import annotations

import json
import time
from pathlib import Path


class ReviewStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = self._load()

    def annotate(self, file: str, text: str, *, reviewer: str, segment_index: int | None = None) -> None:
        entry = self._data.setdefault(file, {"annotations": [], "decision": None})
        entry["annotations"].append(
            {
                "text": text,
                "reviewer": reviewer,
                "segment_index": segment_index,
                "timestamp": time.time(),
            }
        )
        self._save()

    def decide(self, file: str, decision: str, *, reviewer: str) -> None:
        if decision not in {"accept", "reject", "needs-review"}:
            raise ValueError("decision must be accept, reject, or needs-review")
        entry = self._data.setdefault(file, {"annotations": [], "decision": None})
        entry["decision"] = decision
        entry["decision_reviewer"] = reviewer
        entry["decision_timestamp"] = time.time()
        self._save()

    def for_file(self, file: str) -> dict:
        return dict(self._data.get(file, {"annotations": [], "decision": None}))

    def export(self) -> dict:
        return json.loads(json.dumps(self._data))

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)
