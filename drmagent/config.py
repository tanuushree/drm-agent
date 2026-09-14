import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Explicit, deterministic .env resolution instead of relying on
# load_dotenv()'s default frame-introspection search (which can behave
# differently depending on how the process was launched — e.g. uv run's
# --reload subprocess). Checked in order:
#   1. drmagent/.env       (next to this file — the package directory)
#   2. <project root>/.env (one level up, alongside pyproject.toml)
_PACKAGE_DIR = Path(__file__).resolve().parent
_ENV_CANDIDATES = [
    _PACKAGE_DIR / ".env",
    _PACKAGE_DIR.parent / ".env",
]

for _candidate in _ENV_CANDIDATES:
    if _candidate.is_file():
        load_dotenv(_candidate)
        break
else:
    # Neither known location has one; still try the default search as a
    # last resort rather than silently running on defaults only.
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    ngo_username: str = os.getenv("NGO_USERNAME", "ngo_admin")
    ngo_password: str = os.getenv("NGO_PASSWORD", "change-me")

    # When true, the app skips real Google OAuth and Gmail API calls and
    # uses drmagent.gmail.mock_service.MockGmailService instead, which
    # serves a fixed set of scripted donor conversations covering every
    # branch of the pipeline (Thank You / Follow-Up / Outreach / each
    # Human Review trigger / no-history / low-confidence). This exists so
    # the full pipeline — including "sending" a reply — can be exercised
    # and demoed without needing real donor email history or live Google
    # OAuth credentials configured.
    use_mock_gmail: bool = os.getenv("USE_MOCK_GMAIL", "false").strip().lower() in (
        "1", "true", "yes", "on",
    )

    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    google_redirect_uri: str = os.getenv(
        "GOOGLE_REDIRECT_URI",
        "http://localhost:8000/auth/google/callback",
    )

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    )

    consolidation_model: str = os.getenv(
        "CONSOLIDATION_MODEL",
        "openai/gpt-oss-20b",
    )

    extraction_model: str = os.getenv(
        "EXTRACTION_MODEL",
        "openai/gpt-oss-120b",
    )

    classification_model: str = os.getenv(
        "CLASSIFICATION_MODEL",
        "openai/gpt-oss-20b",
    )

    planning_model: str = os.getenv(
        "PLANNING_MODEL",
        "openai/gpt-oss-120b",
    )

    execution_model: str = os.getenv(
        "EXECUTION_MODEL",
        "openai/gpt-oss-120b",
    )

    max_threads_per_donor: int = int(
        os.getenv("MAX_THREADS_PER_DONOR", "20")
    )

    max_messages_per_thread: int = int(
        os.getenv("MAX_MESSAGES_PER_THREAD", "30")
    )

    max_conversation_chars: int = int(
        os.getenv("MAX_CONVERSATION_CHARS", "50000")
    )

    # Hard deterministic approval threshold.
    # Any INR donation/payment amount strictly greater than this value
    # requires human approval.
    donation_approval_threshold: float = float(
        os.getenv("DONATION_APPROVAL_THRESHOLD", "100000")
    )


settings = Settings()