"""Persisted queue of items awaiting human review, with approve/edit/reject
actions.

Backed by a JSON file, following the same pattern as state_store.py. This
module is a pure data store; the caller (main.py) is responsible for also
writing an audit entry for every mutation, so the two logs can't silently
drift apart.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_lock = threading.Lock()

# Fields on the plan an editor is allowed to change. Deliberately excludes
# `action`, `requires_human_approval`, and `donor_id` — editing a draft
# should never let someone quietly bypass the approval gate or relabel the
# action itself; only the message content and execution details are
# editable. To change the action, reject the item and let the pipeline
# re-run.
EDITABLE_PLAN_FIELDS = {"recommended_action", "next_step", "message_type", "message_context"}

# The email actually sent on approval lives at result.execution.email
# (subject/body), not on the plan — the plan only carries the
# message_type/message_context metadata the LLM used to draft it. These
# two keys, sent by the review editor UI as "email_subject"/"email_body",
# must be written to that execution.email location, or an edit a reviewer
# makes in the UI has no effect on what actually gets sent.
EDITABLE_EMAIL_FIELDS = {"email_subject": "subject", "email_body": "body"}


class ReviewItemNotFound(KeyError):
    pass


class ReviewItemAlreadyResolved(ValueError):
    pass


class ReviewStore:
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

    def upsert_pending(self, donor_id: str, result: dict) -> dict:
        """Called by the pipeline whenever a donor's plan needs human
        review. Overwrites any prior record for this donor: a fresh run's
        plan supersedes an old, unresolved review item rather than piling
        up duplicates."""
        with _lock:
            data = self._read()
            record = {
                "donor_id": donor_id,
                "status": "pending",
                "result": result,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "resolved_at": None,
                "resolved_by": None,
                "resolution_notes": None,
                "edit_history": [],
            }
            data[donor_id] = record
            self._write(data)
            return record

    def get(self, donor_id: str) -> Optional[dict]:
        with _lock:
            return self._read().get(donor_id)

    def list(self, status: Optional[str] = None) -> list[dict]:
        with _lock:
            data = self._read()
        records = list(data.values())
        if status:
            records = [r for r in records if r["status"] == status]
        records.sort(key=lambda r: r["created_at"], reverse=True)
        return records

    def _require_pending(self, data: dict, donor_id: str) -> dict:
        record = data.get(donor_id)
        if not record:
            raise ReviewItemNotFound(f"No review item for donor_id={donor_id}")
        if record["status"] != "pending":
            raise ReviewItemAlreadyResolved(
                f"Review item for {donor_id} is already '{record['status']}'"
            )
        return record

    def approve(self, donor_id: str, actor: str) -> dict:
        with _lock:
            data = self._read()
            record = self._require_pending(data, donor_id)
            record["status"] = "approved"
            record["resolved_at"] = datetime.now(timezone.utc).isoformat()
            record["resolved_by"] = actor
            self._write(data)
            return record

    def reject(self, donor_id: str, actor: str, reason: str) -> dict:
        with _lock:
            data = self._read()
            record = self._require_pending(data, donor_id)
            record["status"] = "rejected"
            record["resolved_at"] = datetime.now(timezone.utc).isoformat()
            record["resolved_by"] = actor
            record["resolution_notes"] = reason
            self._write(data)
            return record

    def edit(self, donor_id: str, updates: dict, actor: str) -> dict:
        """Edit the draft plan's message/execution fields on a pending
        item. This does NOT resolve the item — a human must still
        explicitly approve or reject it afterward, even after editing."""
        applied_plan = {k: v for k, v in updates.items() if k in EDITABLE_PLAN_FIELDS}
        applied_email = {
            EDITABLE_EMAIL_FIELDS[k]: v
            for k, v in updates.items()
            if k in EDITABLE_EMAIL_FIELDS and v is not None
        }
        with _lock:
            data = self._read()
            record = self._require_pending(data, donor_id)
            plan = record["result"].setdefault("plan", {})
            plan.update(applied_plan)

            if applied_email:
                execution = record["result"].setdefault("execution", {}) or {}
                email = execution.setdefault("email", {}) or {}
                email.update(applied_email)
                execution["email"] = email
                record["result"]["execution"] = execution
                # Keep the private review context (used at approval time)
                # in sync too, since _send_reviewed_item reads from there.
                context = record["result"].get("_review_context")
                if context is not None:
                    ctx_execution = context.setdefault("execution", {}) or {}
                    ctx_email = ctx_execution.setdefault("email", {}) or {}
                    ctx_email.update(applied_email)
                    ctx_execution["email"] = ctx_email
                    context["execution"] = ctx_execution

            record["edit_history"].append(
                {
                    "edited_at": datetime.now(timezone.utc).isoformat(),
                    "edited_by": actor,
                    "fields": {**applied_plan, **applied_email},
                }
            )
            self._write(data)
            return record
