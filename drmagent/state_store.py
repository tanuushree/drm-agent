"""Minimal persisted state so the same low-stakes action isn't repeatedly
re-triggered for a donor on every run of the batch job.

This is intentionally a small JSON-file store rather than a database, to
match the rest of this prototype's footprint. Swap it for a real table the
moment this runs anywhere with concurrent writers.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from drmagent.models import Action, DonorActionState

_lock = threading.Lock()

# Actions that are safe to suppress/downgrade to "Wait" when repeated inside
# the cooldown window. Human Review and Follow-Up are deliberately excluded:
# a pending human review or an explicit donor-requested follow-up must never
# be silently swallowed just because a similar action fired recently.
COOLDOWN_GUARDED_ACTIONS: set[str] = {"Thank You", "Outreach"}


class DonorStateStore:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._write({})

    def _read(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _write(self, data: dict) -> None:
        tmp_path = self._path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, self._path)

    def get(self, donor_id: str) -> DonorActionState:
        with _lock:
            data = self._read()
        record = data.get(donor_id)
        if not record:
            return DonorActionState(donor_id=donor_id)
        return DonorActionState(**record)

    def record_action(self, donor_id: str, action: Action) -> None:
        with _lock:
            data = self._read()
            data[donor_id] = {
                "donor_id": donor_id,
                "last_action": action,
                "last_action_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write(data)


def is_in_cooldown(
    state: DonorActionState,
    proposed_action: str,
    cooldown_days: int,
) -> bool:
    """True if `proposed_action` was already taken for this donor within
    the cooldown window and should therefore be suppressed."""

    if proposed_action not in COOLDOWN_GUARDED_ACTIONS:
        return False
    if state.last_action != proposed_action or not state.last_action_at:
        return False

    try:
        last_at = datetime.fromisoformat(state.last_action_at)
    except ValueError:
        return False

    if last_at.tzinfo is None:
        last_at = last_at.replace(tzinfo=timezone.utc)

    return datetime.now(timezone.utc) - last_at < timedelta(days=cooldown_days)
