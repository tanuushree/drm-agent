"""Append-only audit trail.

Records who did what, when: every automated pipeline decision, and every
human approve/edit/reject action on a flagged item. Deliberately a plain
JSONL file rather than a database, matching the footprint of the rest of
this prototype — swap it for a real table before this handles real traffic
with concurrent writers.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_lock = threading.Lock()


class AuditLog:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.touch()

    def log(
        self,
        event_type: str,
        donor_id: Optional[str],
        actor: str,
        details: Optional[dict] = None,
    ) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "donor_id": donor_id,
            "actor": actor,
            "details": details or {},
        }
        with _lock:
            with open(self._path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        return entry

    def list(self, donor_id: Optional[str] = None, limit: int = 200) -> list[dict]:
        if not self._path.exists():
            return []
        with _lock:
            with open(self._path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()

        entries = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if donor_id and entry.get("donor_id") != donor_id:
                continue
            entries.append(entry)

        entries.reverse()  # most recent first
        return entries[:limit]
