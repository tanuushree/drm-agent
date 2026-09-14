"""A drop-in stand-in for GmailService, used when USE_MOCK_GMAIL=true.

This exists so the full pipeline — profile building, classification,
planning, the deterministic approval policy, execution, and the "send"
step — can be exercised end-to-end without real donor email history or
live Google OAuth credentials. It implements the same two methods the
orchestrator actually calls (`get_donor_conversations`, `send_reply`), so
`orchestrator.process_donor(gmail, donor)` works identically regardless of
which implementation it's given.

Each fixture donor below is written to deterministically land on a
different branch of drm/approval.py or drm/agent.py, so a single run over
the matching CSV (see data/mock_donors.csv) demonstrates every action type
and every hard approval rule in one pass:

    D001  Priya Sharma    -> Thank You      (recent unacknowledged donation)
    D002  Arjun Mehta     -> Follow-Up      (donor re-asked; NGO commitment open)
    D003  Neha Kapoor     -> Outreach       (warm donor, long silence)
    D004  Rohan Verma     -> Human Review   (donation amount over threshold)
    D005  Sanjay Gupta    -> Human Review   (asks for bank/payment details)
    D006  Meera Iyer      -> Human Review   (refund / cancellation request)
    D007  Vikram Desai    -> Human Review   (transaction dispute + urgency)
    D008  Ananya Rao      -> Human Review   (no Gmail history at all)
    D009  Kavita Nair     -> Human Review   (single sparse/ambiguous message)

Timestamps are computed relative to "now" at call time (not hardcoded),
so the "recent donation" / "long silence" narratives stay realistic no
matter when you run the demo.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from drmagent.models import DonorConversation, EmailMessage

NGO_EMAIL = "outreach@ngo.org"


def _ts(days_ago: float = 0, hours_ago: float = 0) -> tuple[str, int]:
    """Return (rfc-ish date string, epoch-ms timestamp) for `days_ago`/
    `hours_ago` before now, matching the shape GmailService produces."""
    when = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    return when.strftime("%a, %d %b %Y %H:%M:%S +0000"), int(when.timestamp() * 1000)


def _msg(
    thread_id: str,
    seq: int,
    *,
    from_donor: bool,
    donor_email: str,
    subject: str,
    body: str,
    days_ago: float = 0,
    hours_ago: float = 0,
) -> EmailMessage:
    date_str, timestamp = _ts(days_ago, hours_ago)
    sender = donor_email if from_donor else NGO_EMAIL
    recipients = [NGO_EMAIL] if from_donor else [donor_email]
    return EmailMessage(
        id=f"mock-msg-{thread_id}-{seq}",
        thread_id=thread_id,
        subject=subject,
        sender=sender,
        recipients=recipients,
        date=date_str,
        timestamp=timestamp,
        body_text=body,
    )


def _thread(
    donor_id: str,
    donor_email: str,
    thread_key: str,
    subject: str,
    messages: list[EmailMessage],
) -> DonorConversation:
    thread_id = f"mock-thread-{thread_key}"
    return DonorConversation(
        donor_id=donor_id,
        donor_email=donor_email,
        thread_id=thread_id,
        subject=subject,
        messages=messages,
    )


def _build_fixtures() -> dict[str, list[DonorConversation]]:
    fixtures: dict[str, list[DonorConversation]] = {}

    # D001 -- Thank You: recent donation, never acknowledged.
    email = "priya.sharma@example.com"
    tid = "D001-1"
    fixtures["D001"] = [
        _thread(
            "D001", email, tid, "Donation Confirmation - Rs 5,000",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="Donation Confirmation - Rs 5,000",
                    body=(
                        "Hi team,\n\nI just donated Rs 5,000 towards the school "
                        "kits program. Could you please confirm you received it?\n\n"
                        "Thanks,\nPriya"
                    ),
                    days_ago=4,
                ),
            ],
        )
    ]

    # D002 -- Follow-Up: donor re-asked, NGO's earlier commitment is unresolved.
    email = "arjun.mehta@example.com"
    tid = "D002-1"
    fixtures["D002"] = [
        _thread(
            "D002", email, tid, "Question about my donation receipt",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="Question about my donation receipt",
                    body=(
                        "Hello, could you please send me the 80G tax exemption "
                        "certificate for my donation last month? I need it for "
                        "filing my taxes."
                    ),
                    days_ago=10,
                ),
                _msg(
                    tid, 2, from_donor=False, donor_email=email,
                    subject="Re: Question about my donation receipt",
                    body=(
                        "Hi Arjun, thanks for reaching out! We're compiling your "
                        "80G certificate and will send it over shortly."
                    ),
                    days_ago=9,
                ),
                _msg(
                    tid, 3, from_donor=True, donor_email=email,
                    subject="Re: Question about my donation receipt",
                    body=(
                        "Just following up on this -- I haven't received the "
                        "certificate yet. Could you please share it soon?"
                    ),
                    days_ago=3,
                ),
            ],
        )
    ]

    # D003 -- Outreach: warm past contact, no NGO-requiring action, long silence.
    email = "neha.kapoor@example.com"
    tid = "D003-1"
    fixtures["D003"] = [
        _thread(
            "D003", email, tid, "Great meeting you at the fundraiser!",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="Great meeting you at the fundraiser!",
                    body=(
                        "Loved meeting the team at the gala last week. Looking "
                        "forward to hearing more about how the literacy program "
                        "is progressing!"
                    ),
                    days_ago=95,
                ),
                _msg(
                    tid, 2, from_donor=False, donor_email=email,
                    subject="Re: Great meeting you at the fundraiser!",
                    body=(
                        "So glad you enjoyed it, Neha! We'll keep you posted on "
                        "the literacy program's progress."
                    ),
                    days_ago=93,
                ),
            ],
        )
    ]

    # D004 -- Human Review: donation amount over the approval threshold.
    email = "rohan.verma@example.com"
    tid = "D004-1"
    fixtures["D004"] = [
        _thread(
            "D004", email, tid, "Major gift pledge - Rs 5,00,000",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="Major gift pledge - Rs 5,00,000",
                    body=(
                        "Hi, I'd like to pledge Rs 5,00,000 towards the new "
                        "library building. Please let me know the next steps "
                        "for the transfer."
                    ),
                    days_ago=2,
                ),
            ],
        )
    ]

    # D005 -- Human Review: donor is asking for bank/payment details.
    email = "sanjay.gupta@example.com"
    tid = "D005-1"
    fixtures["D005"] = [
        _thread(
            "D005", email, tid, "Need to update my payment method",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="Need to update my payment method",
                    body=(
                        "I want to switch my recurring donation to a direct "
                        "bank transfer. Can you share your bank account details "
                        "and IFSC code?"
                    ),
                    days_ago=3,
                ),
            ],
        )
    ]

    # D006 -- Human Review: refund / cancellation request.
    email = "meera.iyer@example.com"
    tid = "D006-1"
    fixtures["D006"] = [
        _thread(
            "D006", email, tid, "Please cancel my donation",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="Please cancel my donation",
                    body=(
                        "I made a mistake -- please cancel my donation from "
                        "yesterday and refund the amount to my card."
                    ),
                    days_ago=1,
                ),
            ],
        )
    ]

    # D007 -- Human Review: transaction dispute + high urgency/negative sentiment.
    email = "vikram.desai@example.com"
    tid = "D007-1"
    fixtures["D007"] = [
        _thread(
            "D007", email, tid, "Charged twice for the same donation",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="Charged twice for the same donation",
                    body=(
                        "I was charged twice for my Rs 2,000 donation this "
                        "week. This is unacceptable -- I need this fixed "
                        "immediately."
                    ),
                    hours_ago=12,
                ),
            ],
        )
    ]

    # D008 -- Human Review: no Gmail history at all (zero-confidence path).
    fixtures["D008"] = []

    # D009 -- Human Review: single sparse/ambiguous message (low-confidence path).
    email = "kavita.nair@example.com"
    tid = "D009-1"
    fixtures["D009"] = [
        _thread(
            "D009", email, tid, "hi",
            [
                _msg(
                    tid, 1, from_donor=True, donor_email=email,
                    subject="hi",
                    body="hi just checking in",
                    days_ago=20,
                ),
            ],
        )
    ]

    return fixtures


_FIXTURES = _build_fixtures()


class MockGmailService:
    """Same interface as drmagent.gmail.service.GmailService, backed by
    the scripted fixtures above instead of the real Gmail API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Accepts and ignores the same constructor args as GmailService
        # (credentials, max_threads, max_messages_per_thread) so callers
        # don't need an `if mock:` branch when instantiating.
        self.sent: list[dict[str, Any]] = []

    def get_donor_conversations(self, donor: Any) -> list[DonorConversation]:
        return list(_FIXTURES.get(donor.donor_id, []))

    def send_reply(
        self,
        thread_id: str,
        message_id: str,
        to: str,
        subject: str,
        body: str,
    ) -> dict[str, Any]:
        """Doesn't call any external API -- just records the send so the
        demo can show a real 'sent' result without a real Gmail account."""
        fake_id = f"mock-sent-{uuid.uuid4().hex[:12]}"
        record = {
            "id": fake_id,
            "threadId": thread_id,
            "in_reply_to_message_id": message_id,
            "to": to,
            "subject": subject,
            "body": body,
        }
        self.sent.append(record)
        return record
