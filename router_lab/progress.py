"""Append-only answer log, so an interrupted collection resumes where it stopped.

Each line is one model's answer to one question (or the error it hit). A
record is written and fsynced as soon as the call returns, so a crash or a
quota stop loses at most the call in flight. On resume, questions already
answered by a model are skipped; ones that ended in an error are retried.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class AnswerLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> list[dict[str, Any]]:
        """Every parsable record in file order (a line cut off by a crash is skipped)."""
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def answers(self) -> dict[tuple[str, str], dict[str, Any]]:
        """The latest successful answer per (question id, model alias)."""
        done = {}
        for record in self.records():
            if not record.get("error"):
                done[(record["question_id"], record["model"])] = record
        return done

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
