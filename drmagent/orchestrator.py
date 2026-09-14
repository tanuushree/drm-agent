import csv
import io

from drmagent.config import settings
from drmagent.drm.agent import create_drm_agent, classify_and_plan
from drmagent.drm.approval import apply_human_approval_policy
from drmagent.drm.execution_agent import create_execution_agent, execute_plan
from drmagent.models import (
    ActionClassification,
    ActionPlan,
    DonorRecord,
)
from drmagent.profile.service import (
    build_donor_profile,
    get_gmail_context,
    format_conversations,
)
from drmagent.gmail.service import GmailService, get_gmail_context
from drmagent.state_store import DonorStateStore, is_in_cooldown

logger = logging.getLogger(__name__)

_state_store = DonorStateStore(settings.donor_state_path)


def load_donors_from_csv(content: bytes) -> list[DonorRecord]:
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))

    required = {"donor_id", "email"}
    missing = required - set(reader.fieldnames or [])

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {', '.join(sorted(missing))}"
        )

    return [
        DonorRecord(
            donor_id=(row.get("donor_id") or "").strip(),
            name=(row.get("name") or "").strip() or None,
            email=(row.get("email") or "").strip(),
        )
        for row in reader
        if (row.get("donor_id") or "").strip()
        and (row.get("email") or "").strip()
    ]


def process_donor(gmail: GmailService, donor: DonorRecord) -> dict:
    # ------------------------------------------------------------------
    # 1. Build donor profile and retain the original Gmail conversations.
    # ------------------------------------------------------------------
    profile, conversations = build_donor_profile(
        gmail,
        donor,
        settings.max_conversation_chars,
    )

    # ------------------------------------------------------------------
    # 2. Classification + planning.
    #
    # If there is NO Gmail conversation history, this is deterministically
    # treated as Outreach. We do not ask the LLM to classify this case.
    # ------------------------------------------------------------------
    drm_agent = create_drm_agent()

    if not conversations:
        classification = ActionClassification(
            donor_id=donor.donor_id,
            action="Outreach",
            reason=(
                "No Gmail conversation history was found for this donor. "
                "The donor is therefore treated as a new prospective donor "
                "and requires an introductory outreach email."
            ),
            urgency="low",
            confidence=1.0,
        )

        plan = drm_agent.structured_output(
            ActionPlan,
            f"""
Create an execution plan for this NEW PROSPECTIVE DONOR.

The action is deterministically fixed as Outreach.

DONOR PROFILE:
{profile.model_dump_json(indent=2)}

ACTION:
{classification.model_dump_json(indent=2)}

Requirements:
- action MUST be Outreach.
- The objective must be to introduce the NGO and establish initial
  communication with the prospective donor.
- The recommended action must be to send an introductory email.
- Do not invent specific NGO facts.
- Do not invent programs, achievements, statistics, dates, events,
  or commitments.
- Do not request information that is not supported by the profile.
- This is a new email, not a reply.
""",
        )

        # Ensure the LLM cannot change the deterministic action.
        plan.action = "Outreach"

    else:
        classification, plan = classify_and_plan(
            drm_agent,
            profile,
        )

    # ------------------------------------------------------------------
    # 3. HARD DETERMINISTIC APPROVAL POLICY.
    # ------------------------------------------------------------------
    plan, approval_reasons = apply_human_approval_policy(
        profile=profile,
        classification=classification,
        plan=plan,
        conversations=conversations,
        donation_threshold=settings.donation_approval_threshold,
    )

    # ------------------------------------------------------------------
    # 4. HARD APPROVAL GATE.
    # ------------------------------------------------------------------
    execution = None

    if not plan.requires_human_approval:

        gmail_context = get_gmail_context(conversations)

        # --------------------------------------------------------------
        # EXISTING DONOR CONVERSATION
        # --------------------------------------------------------------
        if gmail_context:
            execution_agent = create_execution_agent()

            conversation_context = format_conversations(
                conversations,
                settings.max_conversation_chars,
            )

            execution_result = execute_plan(
                agent=execution_agent,
                profile=profile,
                classification=classification,
                plan=plan,
                thread_id=gmail_context.thread_id,
                message_id=gmail_context.message_id,
                conversation_context=conversation_context,
            )

            if (
                execution_result.status == "drafted"
                and execution_result.email is not None
            ):
                email = execution_result.email

                gmail_result = gmail.send_reply(
                    thread_id=execution_result.thread_id,
                    message_id=execution_result.message_id,
                    to=email.to,
                    subject=email.subject,
                    body=email.body,
                )

                execution = {
                    "status": "sent",
                    "thread_id": execution_result.thread_id,
                    "message_id": execution_result.message_id,
                    "gmail_message_id": gmail_result.get("id"),
                    "email": email.model_dump(),
                }

            else:
                execution = execution_result.model_dump()

        # --------------------------------------------------------------
        # NEW PROSPECTIVE DONOR
        # --------------------------------------------------------------
        else:
            # This branch is only reached when there is no Gmail history.
            # The classification above has already been deterministically
            # set to Outreach.

            execution_agent = create_execution_agent()

            execution_result = execute_plan(
                agent=execution_agent,
                profile=profile,
                classification=classification,
                plan=plan,
                thread_id=None,
                message_id=None,
                conversation_context="",
            )

            if (
                execution_result.status == "drafted"
                and execution_result.email is not None
            ):
                email = execution_result.email

                gmail_result = gmail.send_email(
                    to=email.to,
                    subject=email.subject,
                    body=email.body,
                )

                execution = {
                    "status": "sent",
                    "thread_id": None,
                    "message_id": None,
                    "gmail_message_id": gmail_result.get("id"),
                    "email": email.model_dump(),
                }

            else:
                execution = execution_result.model_dump()
        if gmail_context is None:
            execution = {
                "status": "not_executed",
                "reason": "No Gmail thread/message context available.",
            }
        else:
            try:
                execution_agent = create_execution_agent()

                conversation_context = format_conversations(
                    conversations,
                    settings.max_conversation_chars,
                )

                execution_result = execute_plan(
                    agent=execution_agent,
                    profile=profile,
                    classification=classification,
                    plan=plan,
                    thread_id=gmail_context.thread_id,
                    message_id=gmail_context.message_id,
                    conversation_context=conversation_context,
                )

                if (
                    execution_result.status == "drafted"
                    and execution_result.email is not None
                ):
                    email = execution_result.email

                    # Drafting succeeded; the actual Gmail send is a
                    # separate failure mode (auth/scopes/API errors) and
                    # shouldn't be conflated with a drafting failure, so
                    # it's wrapped separately below.
                    try:
                        gmail_result = gmail.send_reply(
                            thread_id=execution_result.thread_id,
                            message_id=execution_result.message_id,
                            to=email.to,
                            subject=email.subject,
                            body=email.body,
                        )
                        execution_result.status = "sent"
                        execution_result.gmail_message_id = gmail_result.get("id")
                    except Exception as send_exc:  # noqa: BLE001
                        logger.exception(
                            "Gmail send_reply failed for donor_id=%s",
                            donor.donor_id,
                        )
                        execution_result.status = "error"
                        execution_result.reason = (
                            f"Draft was created but sending failed: "
                            f"{send_exc.__class__.__name__}: {send_exc}"
                        )

                execution = execution_result.model_dump()

            except Exception as exec_exc:  # noqa: BLE001
                logger.exception(
                    "Execution stage failed for donor_id=%s",
                    donor.donor_id,
                )
                execution = {
                    "status": "error",
                    "thread_id": gmail_context.thread_id,
                    "message_id": gmail_context.message_id,
                    "reason": (
                        f"Execution agent failed: "
                        f"{exec_exc.__class__.__name__}: {exec_exc}"
                    ),
                }

    # ------------------------------------------------------------------
    # 5. Return planning + approval + execution information.
    # ------------------------------------------------------------------
    return {
        "donor": donor.model_dump(),
        "conversation_count": len(conversations),
        "profile": profile.model_dump(),
        "classification": classification.model_dump(),
        "plan": plan.model_dump(),
        "approval": {
            "requires_human_approval": plan.requires_human_approval,
            "reason": approval_reasons,
        },
        "execution": execution,
    }
        "cooldown_suppressed": suppressed_for_cooldown,
    }


def process_donor(gmail: GmailService, donor: DonorRecord) -> dict:
    """Process a single donor, never letting an unexpected failure here take
    down the rest of a batch run. Any unhandled error is itself treated as a
    reason for human review rather than as a crash."""

    try:
        return _process_donor_inner(gmail, donor)
    except Exception as exc:  # noqa: BLE001 - deliberate top-level fail-safe
        logger.exception("process_donor failed for donor_id=%s", donor.donor_id)
        return {
            "donor": donor.model_dump(),
            "conversation_count": 0,
            "profile": None,
            "classification": None,
            "plan": {
                "donor_id": donor.donor_id,
                "action": "Human Review",
                "objective": "Recover from an unhandled processing error.",
                "recommended_action": "Escalate to a human before any donor communication.",
                "next_step": "Human approval required.",
                "requires_human_approval": True,
                "consistency_notes": [f"Unhandled error during processing: {exc}"],
            },
            "approval": {
                "requires_human_approval": True,
                "reason": [f"Unhandled processing error: {exc.__class__.__name__}: {exc}"],
            },
            "execution": None,
            "cooldown_suppressed": False,
        }
