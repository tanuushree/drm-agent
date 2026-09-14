# DRM Agent

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Testing/demoing without real donor email history (recommended first run)

Edit `.env` and set:

```
USE_MOCK_GMAIL=true
GROQ_API_KEY=<your real Groq key>   # LLM calls still hit Groq for real
```

Run the app:

```bash
uvicorn drmagent.main:app --reload
```

Open `http://localhost:8000/app/`, log in with `NGO_USERNAME`/`NGO_PASSWORD`
from `.env` — in mock mode this skips Google OAuth entirely and marks Gmail
as "connected" immediately. Upload `data/mock_donors.csv` and run the
pipeline. You'll see all nine action types and every hard approval rule
fire in one pass — see the docstring at the top of
`drmagent/gmail/mock_service.py` for exactly which donor exercises which
branch, and `drmagent/data/mock_donors.csv` for the matching donor list.

This exercises the real Groq LLM calls end-to-end; only the Gmail I/O is
faked. Emails "sent" during a mock run are recorded in-memory, not
actually delivered anywhere.

## Testing with a real Gmail account

Set `USE_MOCK_GMAIL=false`, fill in `GOOGLE_CLIENT_ID`/`GOOGLE_CLIENT_SECRET`/
`GOOGLE_REDIRECT_URI`, and complete the OAuth flow from the UI. For a first
real test, send yourself 1-2 emails from a second account pretending to be
a donor before running the pipeline, so there's real thread history to
read.

## What changed in this pass

- Added the missing `EmailDraft`, `ExecutionResult`, and `GmailContext`
  models (`models.py`) that the execution agent needed but didn't have.
- Implemented `get_gmail_context()` (`gmail/service.py`) — deterministically
  picks the most recent message across a donor's threads to reply into.
- Wired the previously-missing imports in `orchestrator.py`
  (`format_conversations`, `create_execution_agent`, `execute_plan`,
  `get_gmail_context`), so the send-email path actually runs instead of
  raising `NameError`/`ImportError`.
- The orchestrator's returned dict was missing the `execution` field
  entirely — added it, and surfaced it in the results UI (drafted/sent/error
  states), instead of only being visible via "Raw JSON".
- Wrapped the Gmail send + execution-agent call in their own try/except so
  a failed send degrades to an `error` status for that donor rather than
  looking identical to a successful one.
- Removed `/test/gmail-reply` (a debug endpoint with a hardcoded
  thread/message/recipient that also read session credentials without
  decrypting them).
- Added `gmail/mock_service.py` + `data/mock_donors.csv` + `USE_MOCK_GMAIL`
  so the pipeline can be tested/demoed without real donor history.
- Fixed `requirements.txt` (it was missing `fastapi`, `uvicorn`,
  `python-dotenv`, `pydantic`, `google-auth-oauthlib`, `google-auth`,
  `google-api-python-client`, `beautifulsoup4`, `python-multipart` — all
  of which are actually imported by the app).
- Replaced the committed `.env` (real secrets) with `.env.example` +
  `.gitignore`. **Rotate any credentials that were in the original `.env`
  before sharing this project further.**

## Still open (not fixed in this pass, see prior review doc for detail)

- No approve/edit/reject UI for Human Review items — currently read-only.
- `/donors/process` runs sequentially and blocks the request; no
  concurrency or streaming progress.
- Global in-memory `donor_records`/`sessions` — single-tenant, not
  persisted across restarts.
- `classification_model` setting is defined but unused (classification and
  planning both currently run on `planning_model`).
