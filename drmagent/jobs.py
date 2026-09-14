"""In-process background job runner for batch donor processing.

`/donors/process` used to loop over every donor synchronously and hold the
HTTP request open for the whole batch, with the frontend showing a fixed,
fake phase animation while it waited. This module instead runs the batch
on a dedicated thread and keeps a shared, lock-protected status dict that
`/jobs/{job_id}` can poll — giving the frontend real per-donor progress
(which donor, which pipeline phase) instead of a canned animation, without
pulling in a task queue dependency for a prototype this size.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Callable

_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def create_job(total: int) -> str:
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "running",  # running | done | error
            "total": total,
            "completed": 0,
            "current_donor": None,
            "current_phase": None,
            "current_detail": None,
            "results": [],
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
        }
    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _update(job_id: str, **fields) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


def _run_batch(job_id: str, donors: list, process_one: Callable) -> None:
    """`process_one(donor, on_phase)` must return a JSON-serializable
    result dict, matching orchestrator.process_donor's signature extended
    with a phase-progress callback."""
    results = []
    try:
        for donor in donors:
            label = donor.name or donor.donor_id

            # `orchestrator.process_donor`'s progress callback is called as
            # progress(phase_name, detail) -- two positional args. `_label`
            # is bound once via the default-argument trick (the standard
            # fix for late-binding closures in a loop) and must NOT be a
            # plain positional parameter, or the caller's `detail` argument
            # silently overwrites it on every call, showing the phase
            # sentence twice in the UI instead of "<donor name> (<phase>)".
            def on_phase(phase_name: str, detail: str = "", _label=label) -> None:
                _update(
                    job_id,
                    current_donor=_label,
                    current_phase=phase_name,
                    current_detail=detail,
                )

            result = process_one(donor, on_phase)
            results.append(result)
            with _lock:
                job = _jobs.get(job_id)
                if job:
                    job["completed"] += 1
                    job["results"] = list(results)

        _update(
            job_id,
            status="done",
            current_phase=None,
            current_donor=None,
            current_detail=None,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the poller, not silently dropped
        _update(
            job_id,
            status="error",
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def start_batch(donors: list, process_one: Callable) -> str:
    job_id = create_job(total=len(donors))
    thread = threading.Thread(target=_run_batch, args=(job_id, donors, process_one), daemon=True)
    thread.start()
    return job_id
