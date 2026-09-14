import json
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.oauth2.credentials import Credentials

from drmagent.config import settings
from drmagent.gmail.auth import create_authorization_url, exchange_code
from drmagent.gmail.mock_service import MockGmailService
from drmagent.gmail.service import GmailService
from drmagent.models import DonorRecord, LoginRequest
from drmagent.orchestrator import load_donors_from_csv, process_donor
from drmagent.security import decrypt_credentials, encrypt_credentials


# application treat Gmail as connected without storing OAuth credentials.
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.oauth2.credentials import Credentials

from drmagent.config import settings
from drmagent.gmail.auth import create_authorization_url, exchange_code
from drmagent.gmail.mock_service import MockGmailService
from drmagent.gmail.service import GmailService
from drmagent.models import (
    ActionClassification,
    ActionPlan,
    DonorConversation,
    DonorRecord,
    DonorProfile,
    LoginRequest,
)
from drmagent.audit import AuditLog
from drmagent.jobs import get_job, start_batch
from drmagent.orchestrator import load_donors_from_csv, process_donor
from drmagent.review_store import ReviewItemAlreadyResolved, ReviewItemNotFound, ReviewStore
from drmagent.security import decrypt_credentials, encrypt_credentials
from drmagent.drm.execution_agent import create_execution_agent, execute_plan
from drmagent.drm.approval import determine_human_approval
from drmagent.profile.service import format_conversations


# Sentinel used when USE_MOCK_GMAIL=true. This lets the rest of the
# application treat Gmail as connected without storing OAuth credentials.
_MOCK_CREDENTIALS_MARKER = "mock"

app = FastAPI(title="DRM Agent", version="1.0.0")

# Demo-only in-memory state. Replace with a database/secure session store
# in production.
sessions: dict[str, dict[str, Any]] = {}
oauth_states: dict[str, dict[str, str]] = {}
review_store = ReviewStore(settings.review_store_path)
audit_log = AuditLog(settings.audit_log_path)
donor_records: list[DonorRecord] = []

_session_created_at: dict[str, float] = {}
_oauth_state_created_at: dict[str, float] = {}


def _sweep_expired() -> None:
    now = time.time()
    ttl = settings.session_ttl_seconds

    for token in [
        token
        for token, created_at in _session_created_at.items()
        if now - created_at > ttl
    ]:
        sessions.pop(token, None)
        _session_created_at.pop(token, None)

    # OAuth states are intentionally short-lived.
    for state in [
        state
        for state, created_at in _oauth_state_created_at.items()
        if now - created_at > 600
    ]:
        oauth_states.pop(state, None)
        _oauth_state_created_at.pop(state, None)


def _session(request: Request) -> dict[str, Any]:
    _sweep_expired()

    token = request.cookies.get("session_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return sessions[token]


def _actor(session: dict[str, Any]) -> str:
    return session.get("username") or "unknown"


def _json_with_session_cookie(
    payload: dict[str, Any],
    login_token: str,
) -> JSONResponse:
    response = JSONResponse(payload)
    response.set_cookie(
        "session_token",
        login_token,
        httponly=True,
        max_age=settings.session_ttl_seconds,
        samesite="lax",
    )
    return response


@app.get("/")
def root():
    return {"service": "DRM Agent", "status": "ok"}


@app.get("/session")
def get_session(request: Request):
    """Return session state without raising 401 for logged-out visitors."""
    _sweep_expired()

    token = request.cookies.get("session_token")
    session = sessions.get(token) if token else None

    if not session:
        return {
            "authenticated": False,
            "gmail_connected": False,
        }

    return {
        "authenticated": bool(session.get("authenticated")),
        "gmail_connected": bool(session.get("gmail_credentials")),
    }


@app.post("/login")
def login(payload: LoginRequest):
    if (
        payload.username != settings.ngo_username
        or payload.password != settings.ngo_password
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    _sweep_expired()

    login_token = secrets.token_urlsafe(32)

    if settings.use_mock_gmail:
        sessions[login_token] = {
            "authenticated": True,
            "gmail_credentials": _MOCK_CREDENTIALS_MARKER,
            "username": payload.username,
        }
        _session_created_at[login_token] = time.time()

        return _json_with_session_cookie(
            {
                "message": (
                    "Mock Gmail mode is active (USE_MOCK_GMAIL=true) -- "
                    "skipping real Google authorization."
                ),
                "mock": True,
            },
            login_token,
        )

    auth_url, state, code_verifier = create_authorization_url()

    oauth_states[state] = {
        "login_token": login_token,
        "code_verifier": code_verifier,
    }
    _oauth_state_created_at[state] = time.time()

    sessions[login_token] = {
        "authenticated": True,
        "gmail_credentials": None,
        "username": payload.username,
    }
    _session_created_at[login_token] = time.time()

    return {
        "message": "Credentials accepted. Authorize Gmail next.",
        "google_auth_url": auth_url,
        "mock": False,
    }


@app.get("/auth/google/callback")
def google_callback(code: str, state: str):
    _sweep_expired()

    oauth_data = oauth_states.pop(state, None)
    _oauth_state_created_at.pop(state, None)

    if not oauth_data:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired OAuth state",
        )

    login_token = oauth_data["login_token"]
    code_verifier = oauth_data["code_verifier"]

    if login_token not in sessions:
        raise HTTPException(
            status_code=400,
            detail="Login session expired",
        )

    credentials = exchange_code(
        code,
        state,
        code_verifier,
    )

    # Store encrypted credentials; never store raw OAuth JSON in the session.
    sessions[login_token]["gmail_credentials"] = encrypt_credentials(
        credentials.to_json()
    )

    response = RedirectResponse(url="/app/?connected=1")
    response.set_cookie(
        "session_token",
        login_token,
        httponly=True,
        max_age=settings.session_ttl_seconds,
        samesite="lax",
    )
    return response


@app.post("/logout")
def logout(request: Request):
    token = request.cookies.get("session_token")

    if token:
        sessions.pop(token, None)
        _session_created_at.pop(token, None)

    response = RedirectResponse(url="/app/")
    response.delete_cookie("session_token")
    return response


@app.post("/donors/upload")
async def upload_donors(
    request: Request,
    file: UploadFile = File(...),
):
    _session(request)

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")

    try:
        records = load_donors_from_csv(await file.read())
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    donor_records.clear()
    donor_records.extend(records)

    return {
        "count": len(donor_records),
        "donors": [donor.model_dump() for donor in donor_records],
    }


@app.get("/donors")
def list_donors(request: Request):
    _session(request)

    return {
        "count": len(donor_records),
        "donors": [donor.model_dump() for donor in donor_records],
    }


def _build_gmail_service(
    session: dict[str, Any],
) -> GmailService | MockGmailService:
    """Build the Gmail client for this session."""
    if settings.use_mock_gmail:
        return MockGmailService()

    encrypted_json = session.get("gmail_credentials")
    if not encrypted_json:
        raise HTTPException(
            status_code=400,
            detail="Gmail authorization is required",
        )

    decrypted_json = decrypt_credentials(encrypted_json)
    credentials = Credentials.from_authorized_user_info(
        json.loads(decrypted_json)
    )

    return GmailService(
        credentials,
        max_threads=settings.max_threads_per_donor,
        max_messages_per_thread=settings.max_messages_per_thread,
    )


@app.post("/donors/process")
def process_donors(request: Request):
    session = _session(request)

    if not session.get("gmail_credentials"):
        raise HTTPException(
            status_code=400,
            detail="Gmail authorization is required",
        )

    if not donor_records:
        raise HTTPException(
            status_code=400,
            detail="Upload donor CSV first",
        )

    gmail = _build_gmail_service(session)
    actor = _actor(session)

    job_id = start_batch(
        list(donor_records),
        lambda donor, on_phase: process_donor(
            gmail,
            donor,
            actor=actor,
            progress=on_phase,
        ),
    )

    return {"job_id": job_id, "count": len(donor_records)}


def _public_result(result: dict[str, Any]) -> dict[str, Any]:
    """Remove private review context before data reaches the browser."""
    public = dict(result)
    public.pop("_review_context", None)
    return public


@app.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    _session(request)
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job = dict(job)
    job["results"] = [_public_result(r) for r in job.get("results", [])]
    return job


@app.get("/reviews")
def list_reviews(request: Request, status: str | None = "pending"):
    _session(request)
    records = review_store.list(status=status)
    for record in records:
        record.get("result", {}).pop("_review_context", None)
    return {"reviews": records}


@app.get("/reviews/{donor_id}")
def get_review(donor_id: str, request: Request):
    _session(request)
    record = review_store.get(donor_id)
    if not record:
        raise HTTPException(status_code=404, detail="Review item not found")
    public = dict(record)
    public["result"] = dict(public.get("result", {}))
    public["result"].pop("_review_context", None)
    return public


@app.patch("/reviews/{donor_id}/draft")
def edit_review_draft(donor_id: str, payload: dict[str, Any], request: Request):
    session = _session(request)
    try:
        record = review_store.edit(donor_id, payload, _actor(session))
    except ReviewItemNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReviewItemAlreadyResolved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    audit_log.log(
        event_type="human_review_draft_edited",
        donor_id=donor_id,
        actor=_actor(session),
        details={"fields": list(payload.keys())},
    )
    public = dict(record)
    public["result"] = dict(public.get("result", {}))
    public["result"].pop("_review_context", None)
    return public


def _send_reviewed_item(record: dict, session: dict[str, Any], donor_id: str) -> dict[str, Any]:
    context = record.get("result", {}).get("_review_context") or {}
    donor = DonorRecord(**context.get("donor", {}))
    proposed_plan = ActionPlan(**context.get("proposed_plan", {}))
    execution = record.get("result", {}).get("execution") or {}
    email = execution.get("email") or {}

    if donor.donor_id != donor_id:
        raise HTTPException(status_code=400, detail="Review donor mismatch")
    if not donor.email:
        raise HTTPException(status_code=400, detail="Donor has no email address")
    if not email.get("subject") or not email.get("body"):
        raise HTTPException(
            status_code=409,
            detail="This review item has no editable email draft to approve.",
        )

    # The reviewer approves the exact stored draft. Do not call the LLM again
    # at approval time: approval must not silently replace what the human saw.
    gmail = _build_gmail_service(session)
    gmail_context = context.get("gmail_context")
    thread_id = gmail_context.get("thread_id") if gmail_context else None
    message_id = gmail_context.get("message_id") if gmail_context else None

    if thread_id and message_id:
        gmail_result = gmail.send_reply(
            thread_id=thread_id,
            message_id=message_id,
            to=donor.email,
            subject=email["subject"],
            body=email["body"],
        )
    else:
        send_email = getattr(gmail, "send_email", None)
        if not callable(send_email):
            raise HTTPException(status_code=500, detail="Gmail service cannot send new emails")
        gmail_result = send_email(
            to=donor.email,
            subject=email["subject"],
            body=email["body"],
        )

    gmail_message_id = (
        gmail_result.get("id") if isinstance(gmail_result, dict) else None
    )
    audit_log.log(
        event_type="human_review_approved_and_sent",
        donor_id=donor_id,
        actor=_actor(session),
        details={
            "action": proposed_plan.action,
            "gmail_message_id": gmail_message_id,
        },
    )
    return {
        "donor_id": donor_id,
        "status": "sent",
        "execution": {
            **execution,
            "status": "sent",
            "gmail_message_id": gmail_message_id,
            "email": {
                "to": donor.email,
                "subject": email["subject"],
                "body": email["body"],
            },
        },
    }


@app.post("/reviews/{donor_id}/approve")
def approve_review(donor_id: str, request: Request):
    session = _session(request)
    record = review_store.get(donor_id)
    if not record:
        raise HTTPException(status_code=404, detail="Review item not found")
    if record.get("status") != "pending":
        raise HTTPException(status_code=409, detail=f"Review item is already '{record.get('status')}'")

    try:
        result = _send_reviewed_item(record, session, donor_id)
    except HTTPException:
        raise
    except Exception as exc:
        audit_log.log(
            event_type="human_review_send_failed",
            donor_id=donor_id,
            actor=_actor(session),
            details={"error": str(exc)},
        )
        raise HTTPException(
            status_code=502,
            detail=f"Approved draft could not be sent: {exc.__class__.__name__}: {exc}",
        ) from exc

    try:
        review_store.approve(donor_id, _actor(session))
    except (ReviewItemNotFound, ReviewItemAlreadyResolved) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return result


@app.post("/reviews/{donor_id}/reject")
def reject_review(donor_id: str, request: Request, payload: dict[str, Any] | None = None):
    session = _session(request)
    reason = (payload or {}).get("reason", "Rejected by reviewer")
    try:
        record = review_store.reject(donor_id, _actor(session), reason)
    except ReviewItemNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ReviewItemAlreadyResolved as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    audit_log.log(
        event_type="human_review_rejected",
        donor_id=donor_id,
        actor=_actor(session),
        details={"reason": reason},
    )
    return record


@app.get("/audit")
def list_audit(request: Request, donor_id: str | None = None, limit: int = 200):
    _session(request)
    return {"events": audit_log.list(donor_id=donor_id, limit=max(1, min(limit, 500)))}


# Serve the SPA frontend at /app/.
_STATIC_APP_DIR = Path(__file__).resolve().parent / "web"
app.mount(
    "/app",
    StaticFiles(directory=_STATIC_APP_DIR, html=True),
    name="app",
)
