import json
import secrets
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.oauth2.credentials import Credentials

from drmagent.config import settings
from drmagent.gmail.auth import create_authorization_url, exchange_code
from drmagent.gmail.mock_service import MockGmailService
from drmagent.gmail.service import GmailService
from drmagent.models import DonorRecord, LoginRequest
from drmagent.orchestrator import load_donors_from_csv, process_donor

# Sentinel stored in session["gmail_credentials"] when USE_MOCK_GMAIL=true,
# so the truthiness checks used everywhere else ("has the user connected
# Gmail?") keep working without a real OAuth token ever existing.
_MOCK_CREDENTIALS_MARKER = "mock"

app = FastAPI(title="DRM Agent", version="1.0.0")

# Demo-only in-memory state. Replace with a database/secure session store in production.
sessions: dict[str, dict[str, Any]] = {}
# oauth_states: dict[str, str] = {}
oauth_states: dict[str, dict[str, str]] = {}
donor_records: list[DonorRecord] = []


def _session(request: Request) -> dict[str, Any]:
    token = request.cookies.get("session_token")
    if not token or token not in sessions:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sessions[token]


def _json_with_session_cookie(payload: dict[str, Any], login_token: str):
    from fastapi.responses import JSONResponse

    response = JSONResponse(payload)
    response.set_cookie(
        "session_token", login_token, httponly=True,
        max_age=settings.session_ttl_seconds, samesite="lax",
    )
    return response


@app.get("/")
def root():
    return {"service": "DRM Agent", "status": "ok"}

@app.post("/login")
def login(payload: LoginRequest):
    if (
        payload.username != settings.ngo_username
        or payload.password != settings.ngo_password
    ):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # auth_url, state = create_authorization_url()
    # login_token = secrets.token_urlsafe(32)
    # oauth_states[state] = login_token
    auth_url, state, code_verifier = create_authorization_url()

    login_token = secrets.token_urlsafe(32)

    if settings.use_mock_gmail:
        # Demo/testing mode: skip real Google OAuth entirely. The session
        # is immediately "Gmail connected" using the fixture donor
        # conversations in gmail/mock_service.py.
        sessions[login_token] = {
            "authenticated": True,
            "gmail_credentials": _MOCK_CREDENTIALS_MARKER,
        }
        _session_created_at[login_token] = time.time()
        response = _json_with_session_cookie(
            {
                "message": (
                    "Mock Gmail mode is active (USE_MOCK_GMAIL=true) -- "
                    "skipping real Google authorization."
                ),
                "mock": True,
            },
            login_token,
        )
        return response

    auth_url, state, code_verifier = create_authorization_url()

    oauth_states[state] = {
        "login_token": login_token,
        "code_verifier": code_verifier,
    }
    sessions[login_token] = {"authenticated": True, "gmail_credentials": None}
    _session_created_at[login_token] = time.time()
    return {
        "message": "Credentials accepted. Authorize Gmail next.",
        "google_auth_url": auth_url,
        "mock": False,
    }

@app.get("/auth/google/callback")
def google_callback(code: str, state: str):
    # login_token = oauth_states.pop(state, None)
    # if not login_token:
    #     raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")
    # credentials = exchange_code(code, state)
    oauth_data = oauth_states.pop(state, None)

    if not oauth_data:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired OAuth state"
        )

    login_token = oauth_data["login_token"]
    code_verifier = oauth_data["code_verifier"]

    credentials = exchange_code(
        code,
        state,
        code_verifier,
    )
    sessions[login_token]["gmail_credentials"] = credentials.to_json()
    response = RedirectResponse(url="/docs")
    response.set_cookie("session_token", login_token, httponly=True, max_age=28800, samesite="lax")
    return response


@app.post("/logout")
def logout(request: Request):
    token = request.cookies.get("session_token")
    if token:
        sessions.pop(token, None)
    response = RedirectResponse(url="/")
    response.delete_cookie("session_token")
    return response


@app.post("/donors/upload")
async def upload_donors(request: Request, file: UploadFile = File(...)):
    _session(request)
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file")
    try:
        records = load_donors_from_csv(await file.read())
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    donor_records.clear()
    donor_records.extend(records)
    return {"count": len(donor_records), "donors": [d.model_dump() for d in donor_records]}


@app.get("/donors")
def list_donors(request: Request):
    _session(request)
    return {"count": len(donor_records), "donors": [d.model_dump() for d in donor_records]}


def _build_gmail_service(session: dict[str, Any]) -> GmailService | MockGmailService:
    """Build the Gmail client for this session: the scripted MockGmailService
    when USE_MOCK_GMAIL=true, otherwise a real GmailService from the
    session's encrypted OAuth credentials."""
    if settings.use_mock_gmail:
        return MockGmailService()

    decrypted_json = decrypt_credentials(session["gmail_credentials"])
    credentials = Credentials.from_authorized_user_info(json.loads(decrypted_json))
    return GmailService(
        credentials,
        max_threads=settings.max_threads_per_donor,
        max_messages_per_thread=settings.max_messages_per_thread,
    )


@app.post("/donors/process")
def process_donors(request: Request):
    session = _session(request)
    if not session.get("gmail_credentials"):
        raise HTTPException(status_code=400, detail="Gmail authorization is required")
    if not donor_records:
        raise HTTPException(status_code=400, detail="Upload donor CSV first")

    from google.oauth2.credentials import Credentials
    credentials = Credentials.from_authorized_user_info(
        __import__("json").loads(session["gmail_credentials"])
    )
    gmail = GmailService(
        credentials,
        max_threads=settings.max_threads_per_donor,
        max_messages_per_thread=settings.max_messages_per_thread,
    )
    gmail = _build_gmail_service(session)
    results = [process_donor(gmail, donor) for donor in donor_records]
    return {"count": len(results), "results": results}


@app.post("/test/gmail-reply")
def test_gmail_reply(request: Request):
    session = _session(request)

    if not session.get("gmail_credentials"):
        raise HTTPException(
            status_code=400,
            detail="Gmail authorization is required",
        )

    from google.oauth2.credentials import Credentials
    credentials = Credentials.from_authorized_user_info(
        __import__("json").loads(session["gmail_credentials"])
    )

    gmail = GmailService(
        credentials,
        max_threads=settings.max_threads_per_donor,
        max_messages_per_thread=settings.max_messages_per_thread,
    )

    result = gmail.send_reply(
        thread_id="1a096798792ca6ef",
        message_id="1a0967ab3c430a11",
        to="cairen.in@gmail.com",
        subject="Re: Test Gmail Reply",
        body="""Hi,

        This is a test reply from the DRMAgent Gmail execution layer.

        Regards,
        DRMAgent""",
    )

    return {
        "status": "sent",
        "gmail_response": result,
    }
# Serve the SPA frontend at /app/ (and /web/index.html).
_STATIC_APP_DIR = Path(__file__).resolve().parent / "web"
app.mount("/app", StaticFiles(directory=_STATIC_APP_DIR, html=True), name="app")
