"""Append-only answer log, so an interrupted collection resumes where it stopped.

Each line is one model's answer to one question (or the error it hit). A
record is written and fsynced as soon as the call returns, so a crash or a
quota stop loses at most the call in flight. On resume, questions already
answered by a model are skipped; ones that ended in an error are retried.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class AnswerLogLocked(RuntimeError):
    """Another collection run is writing to the same answer file."""


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

    def answered_by_all(self) -> set[str]:
        """Question ids that every model in this log answered without an error."""
        by_model: dict[str, set[str]] = {}
        for question_id, model in self.answers():
            by_model.setdefault(model, set()).add(question_id)
        return set.intersection(*by_model.values()) if by_model else set()

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold ``<file>.lock`` while writing, so two runs can't append the same answers twice."""
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise AnswerLogLocked(f"{lock_path} exists: another run is writing {self.path.name}. If none is "
                                  f"running (a crash), delete the lock file and rerun.") from e
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            yield
        finally:
            lock_path.unlink(missing_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
