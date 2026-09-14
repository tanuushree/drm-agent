from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

class LoginRequest(BaseModel):
    username: str
    password: str

class DonorRecord(BaseModel):
    donor_id: str
    name: Optional[str] = None
    email: str


class EmailMessage(BaseModel):
    id: str
    thread_id: str
    subject: str = ""
    sender: str = ""
    recipients: list[str] = Field(default_factory=list)
    date: str = ""
    timestamp: int = 0
    body_text: str = ""


class DonorConversation(BaseModel):
    donor_id: str
    donor_email: str
    thread_id: str
    subject: str = ""
    messages: list[EmailMessage] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    donor_id: str
    key_points: list[str] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    commitments: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    relationship_signals: list[str] = Field(default_factory=list)
    summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DonorProfile(BaseModel):
    donor_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    donation_context: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    requests: list[str] = Field(default_factory=list)
    commitments_made_by_ngo: list[str] = Field(default_factory=list)
    unresolved_items: list[str] = Field(default_factory=list)
    communication_summary: str = ""
    sentiment: Optional[str] = None
    relationship_signals: list[str] = Field(default_factory=list)
    last_contact_date: Optional[str] = None
    follow_up_required: bool = False
    human_review_required: bool = False
    notes: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


Action = Literal["Thank You", "Outreach", "Follow-Up", "Wait", "Human Review"]


class ActionClassification(BaseModel):
    donor_id: str
    action: Action
    reason: str
    urgency: Literal["low", "medium", "high"] = "low"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ActionPlan(BaseModel):
    donor_id: str
    action: Action
    objective: str
    recommended_action: str
    next_step: str
    message_type: Optional[str] = None
    message_context: Optional[str] = None
    requires_human_approval: bool = False
    due_date: Optional[str] = None


class DonorActionState(BaseModel):
    """Persisted record of the last action taken for a donor, used to avoid
    re-triggering the same low-stakes outbound action on every run."""

    donor_id: str
    last_action: Optional[Action] = None
    last_action_at: Optional[str] = None  # ISO-8601 timestamp


class GmailContext(BaseModel):
    """Identifies which existing Gmail message/thread a reply should be
    attached to. Produced by `get_gmail_context()` from the donor's fetched
    conversations — never guessed by an LLM."""

    thread_id: str
    message_id: str


class EmailDraft(BaseModel):
    """The email the execution agent produces. Deliberately minimal: only
    what's needed to actually send a reply. The agent must not invent the
    recipient — `to` should always resolve to the donor's known email."""

    to: str
    subject: str
    body: str


class ExecutionResult(BaseModel):
    """Result of the execution stage, before/after the Gmail send.

    status:
      - "drafted": the execution agent produced an EmailDraft (the
        orchestrator will attempt to send it next).
      - "sent": the orchestrator successfully sent the drafted email.
      - "not_executed": nothing to execute (e.g. no Gmail thread context).
      - "error": drafting or sending failed; see `reason`.
    """

    status: Literal["drafted", "sent", "not_executed", "error"] = "drafted"
    thread_id: Optional[str] = None
    message_id: Optional[str] = None
    email: Optional[EmailDraft] = None
    gmail_message_id: Optional[str] = None
    reason: Optional[str] = None
