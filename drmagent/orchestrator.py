import copy
import csv
import io
import logging
from collections.abc import Callable
from typing import Any

from drmagent.audit import AuditLog
from drmagent.config import settings
from drmagent.drm.agent import (
    classify_and_plan,
    create_classification_agent,
    create_planning_agent,
)
from drmagent.drm.approval import apply_human_approval_policy
from drmagent.drm.execution_agent import create_execution_agent, execute_plan
from drmagent.gmail.service import GmailService, get_gmail_context
from drmagent.llm_utils import PhaseCallback, report_phase
from drmagent.models import ActionClassification, DonorRecord
from drmagent.profile.service import build_donor_profile, format_conversations
from drmagent.review_store import ReviewStore
from drmagent.state_store import DonorStateStore, is_in_cooldown

logger = logging.getLogger(__name__)

_state_store = DonorStateStore(settings.donor_state_path)
review_store = ReviewStore(settings.review_store_path)
audit_log = AuditLog(settings.audit_log_path)

PIPELINE_ACTOR = "drm-agent-pipeline"
ProgressCallback = Callable[[str, str], None]


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


def _process_donor_inner(
    gmail: GmailService,
    donor: DonorRecord,
    actor: str = PIPELINE_ACTOR,
    progress: ProgressCallback | None = None,
    on_phase: PhaseCallback | None = None,
) -> dict[str, Any]:
    """
    Process one donor through:

        Gmail -> profile -> classification -> planning
        -> deterministic approval policy -> cooldown
        -> draft/review/send

    The LLM never has authority to bypass the deterministic approval gate.
    Human-review cases are drafted but never sent automatically.
    """

    def phase(name: str, detail: str) -> None:
        if progress:
            progress(name, detail)
        if on_phase:
            report_phase(on_phase, detail)

    # ------------------------------------------------------------------
    # 1. Fetch Gmail history and build the donor profile.
    # ------------------------------------------------------------------
    phase("fetching", f"Fetching Gmail threads for {donor.email}")

    # build_donor_profile is intentionally called without an on_phase
    # argument so this orchestrator remains compatible with both the
    # current profile service and the pre-merge version.
    profile, conversations = build_donor_profile(
        gmail,
        donor,
        settings.max_conversation_chars,
    )

    # ------------------------------------------------------------------
    # 2. Classification + planning.
    # ------------------------------------------------------------------
    phase("classifying", "Classifying the donor relationship")

    classification_agent = create_classification_agent()
    planning_agent = create_planning_agent()

    classification, plan = classify_and_plan(
        classification_agent,
        planning_agent,
        profile,
    )

    # Keep the raw model proposal. If deterministic policy later changes
    # the public plan to Human Review, reviewers still need the original
    # proposed action in order to understand what the model intended.
    proposed_plan = copy.deepcopy(plan)

    # A donor with no Gmail history is a new prospective donor. This is
    # deterministic: there is no prior relationship to reply to, so the
    # action is forced to Outreach.
    if not conversations:
        classification = ActionClassification(
            donor_id=donor.donor_id,
            action="Outreach",
            reason=(
                "No Gmail conversation history was found for this donor. "
                "The donor is therefore treated as a new prospective donor "
                "and requires introductory outreach."
            ),
            urgency="low",
            confidence=1.0,
        )

        plan.action = "Outreach"
        plan.objective = (
            "Introduce the NGO and establish initial communication with "
            "the prospective donor."
        )
        plan.recommended_action = "Send an introductory email."
        plan.next_step = "Prepare and send an introductory outreach email."
        plan.message_type = "introductory outreach"
        plan.message_context = (
            "New prospective donor with no prior Gmail conversation."
        )
        plan.requires_human_approval = False

        proposed_plan = copy.deepcopy(plan)

    # ------------------------------------------------------------------
    # 3. HARD DETERMINISTIC APPROVAL POLICY.
    #
    # The LLM-produced requires_human_approval flag is never trusted.
    # The policy recalculates approval requirements and can override the
    # action itself when a safety rule fires.
    # ------------------------------------------------------------------
    phase("approval", "Applying the deterministic approval policy")

    plan, approval_reasons = apply_human_approval_policy(
        profile=profile,
        classification=classification,
        plan=plan,
        conversations=conversations,
        donation_threshold=settings.donation_approval_threshold,
        confidence_threshold=settings.min_confidence_threshold,
    )

    # ------------------------------------------------------------------
    # 4. Cooldown.
    #
    # Do not suppress Human Review or Follow-Up. For low-stakes outbound
    # actions, only suppress a repeat action during the configured window.
    #
    # IMPORTANT: state is recorded only after a message is actually sent.
    # ------------------------------------------------------------------
    donor_state = _state_store.get(donor.donor_id)
    suppressed_for_cooldown = False

    if is_in_cooldown(
        donor_state,
        plan.action,
        settings.action_cooldown_days,
    ):
        suppressed_for_cooldown = True

        original_action = plan.action
        plan.action = "Wait"
        plan.recommended_action = (
            f"No action ({original_action} already sent recently)."
        )
        plan.next_step = "None; wait for the cooldown window to pass."
        plan.message_type = None
        plan.message_context = None

    # ------------------------------------------------------------------
    # 5. Execution gate.
    #
    # Existing Gmail conversations can be replied to deterministically
    # using the latest real Gmail message. New donors have no reply target;
    # where the GmailService exposes send_email(), the execution agent can
    # prepare an introductory email and the application can send it.
    #
    # Human Review NEVER sends here.
    # ------------------------------------------------------------------
    execution: dict[str, Any] | None = None
    gmail_context = get_gmail_context(conversations)

    if plan.requires_human_approval:
        # Draft for review only. Do not send.
        if gmail_context is not None:
            phase("drafting", "Preparing a draft for human review")

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
                    # Use the original proposal for drafting. The public plan
                    # may have been safety-mutated to "Human Review", which
                    # should not cause the draft itself to say "Human Review".
                    plan=proposed_plan,
                    thread_id=gmail_context.thread_id,
                    message_id=gmail_context.message_id,
                    conversation_context=conversation_context,
                )

                execution = execution_result.model_dump()

            except Exception as exec_exc:  # noqa: BLE001
                logger.exception(
                    "Review draft generation failed for donor_id=%s",
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

        else:
            execution = {
                "status": "not_executed",
                "reason": (
                    "Human Review required. No Gmail thread/message context "
                    "was available for an automatic reply draft."
                ),
            }

    elif not suppressed_for_cooldown:
        if gmail_context is not None:
            # --------------------------------------------------------------
            # Existing donor: prepare and send a reply.
            # --------------------------------------------------------------
            phase("drafting", "Preparing a donor communication draft")

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
                    phase("sending", "Sending approved low-risk communication")

                    try:
                        email = execution_result.email

                        gmail_result = gmail.send_reply(
                            thread_id=execution_result.thread_id,
                            message_id=execution_result.message_id,
                            to=email.to,
                            subject=email.subject,
                            body=email.body,
                        )

                        execution_result.status = "sent"
                        execution_result.gmail_message_id = gmail_result.get("id")

                        # Record cooldown only after Gmail confirms the send.
                        _state_store.record_action(
                            donor.donor_id,
                            plan.action,
                        )

                        audit_log.log(
                            event_type="automated_email_sent",
                            donor_id=donor.donor_id,
                            actor=actor,
                            details={
                                "action": plan.action,
                                "gmail_message_id": gmail_result.get("id"),
                                "thread_id": execution_result.thread_id,
                                "message_id": execution_result.message_id,
                            },
                        )

                    except Exception as send_exc:  # noqa: BLE001
                        logger.exception(
                            "Gmail send_reply failed for donor_id=%s",
                            donor.donor_id,
                        )
                        execution_result.status = "error"
                        execution_result.reason = (
                            "Draft was created but sending failed: "
                            f"{send_exc.__class__.__name__}: {send_exc}"
                        )

                execution = execution_result.model_dump()

            except Exception as exec_exc:  # noqa: BLE001
                logger.exception(
                    "Execution/drafting failed for donor_id=%s",
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

        else:
            # --------------------------------------------------------------
            # New prospective donor.
            #
            # There is no thread to reply to. If the current GmailService
            # exposes send_email(), use it; otherwise leave the result as
            # not_executed instead of inventing a Gmail context.
            # --------------------------------------------------------------
            phase("drafting", "Preparing an introductory outreach email")

            try:
                execution_agent = create_execution_agent()

                execution_result = execute_plan(
                    agent=execution_agent,
                    profile=profile,
                    classification=classification,
                    plan=plan,
                    thread_id=None,  # type: ignore[arg-type]
                    message_id=None,  # type: ignore[arg-type]
                    conversation_context="",
                )

                if (
                    execution_result.status == "drafted"
                    and execution_result.email is not None
                ):
                    send_email = getattr(gmail, "send_email", None)

                    if not callable(send_email):
                        execution_result.status = "not_executed"
                        execution_result.reason = (
                            "No Gmail thread exists for this new donor and "
                            "the configured GmailService does not expose "
                            "send_email()."
                        )
                    else:
                        phase(
                            "sending",
                            "Sending approved introductory outreach",
                        )

                        try:
                            email = execution_result.email

                            gmail_result = send_email(
                                to=email.to,
                                subject=email.subject,
                                body=email.body,
                            )

                            execution_result.status = "sent"
                            execution_result.gmail_message_id = (
                                gmail_result.get("id")
                                if isinstance(gmail_result, dict)
                                else None
                            )

                            _state_store.record_action(
                                donor.donor_id,
                                plan.action,
                            )

                            audit_log.log(
                                event_type="automated_email_sent",
                                donor_id=donor.donor_id,
                                actor=actor,
                                details={
                                    "action": plan.action,
                                    "gmail_message_id": execution_result.gmail_message_id,
                                    "thread_id": None,
                                    "message_id": None,
                                },
                            )

                        except Exception as send_exc:  # noqa: BLE001
                            logger.exception(
                                "Gmail send_email failed for donor_id=%s",
                                donor.donor_id,
                            )
                            execution_result.status = "error"
                            execution_result.reason = (
                                "Introductory draft was created but sending "
                                "failed: "
                                f"{send_exc.__class__.__name__}: {send_exc}"
                            )

                execution = execution_result.model_dump()

            except Exception as exec_exc:  # noqa: BLE001
                logger.exception(
                    "New-donor execution failed for donor_id=%s",
                    donor.donor_id,
                )
                execution = {
                    "status": "error",
                    "thread_id": None,
                    "message_id": None,
                    "reason": (
                        f"Execution agent failed: "
                        f"{exec_exc.__class__.__name__}: {exec_exc}"
                    ),
                }

    else:
        execution = {
            "status": "not_executed",
            "reason": "Action suppressed by the donor cooldown window.",
        }

    # ------------------------------------------------------------------
    # 6. Build the public result.
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
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
        "cooldown_suppressed": suppressed_for_cooldown,
    }

    # Private server-side review context. Main/API code can strip this
    # before returning the result to the browser.
    if plan.requires_human_approval:
        result["_review_context"] = {
            "donor": donor.model_dump(),
            "profile": profile.model_dump(),
            "classification": classification.model_dump(),
            "proposed_plan": proposed_plan.model_dump(),
            "final_plan": plan.model_dump(),
            "conversations": [
                conversation.model_dump()
                for conversation in conversations
            ],
            "gmail_context": (
                gmail_context.model_dump()
                if gmail_context is not None
                else None
            ),
            "execution": execution,
        }

        review_store.upsert_pending(donor.donor_id, result)

        audit_log.log(
            event_type="flagged_for_review",
            donor_id=donor.donor_id,
            actor=actor,
            details={"reasons": approval_reasons},
        )

    audit_log.log(
        event_type="processed",
        donor_id=donor.donor_id,
        actor=actor,
        details={
            "action": plan.action,
            "requires_human_approval": plan.requires_human_approval,
            "approval_reasons": approval_reasons,
            "cooldown_suppressed": suppressed_for_cooldown,
            "execution_status": (
                execution.get("status") if execution else None
            ),
        },
    )

    return result


def process_donor(
    gmail: GmailService,
    donor: DonorRecord,
    actor: str = PIPELINE_ACTOR,
    progress: ProgressCallback | None = None,
    on_phase: PhaseCallback | None = None,
) -> dict[str, Any]:
    """
    Process a single donor without allowing one donor failure to terminate
    the rest of a batch.

    Any unexpected failure is converted into a Human Review result and is
    persisted to the review queue as a fail-safe.
    """

    try:
        return _process_donor_inner(
            gmail,
            donor,
            actor=actor,
            progress=progress,
            on_phase=on_phase,
        )

    except Exception as exc:  # noqa: BLE001 - deliberate top-level fail-safe
        logger.exception(
            "process_donor failed for donor_id=%s",
            donor.donor_id,
        )

        result = {
            "donor": donor.model_dump(),
            "conversation_count": 0,
            "profile": None,
            "classification": None,
            "plan": {
                "donor_id": donor.donor_id,
                "action": "Human Review",
                "objective": "Recover from an unhandled processing error.",
                "recommended_action": (
                    "Escalate to a human before any donor communication."
                ),
                "next_step": "Human approval required.",
                "requires_human_approval": True,
            },
            "approval": {
                "requires_human_approval": True,
                "reason": [
                    "Unhandled processing error: "
                    f"{exc.__class__.__name__}: {exc}"
                ],
            },
            "execution": None,
            "cooldown_suppressed": False,
        }

        audit_log.log(
            event_type="processing_error",
            donor_id=donor.donor_id,
            actor=actor,
            details={"error": str(exc)},
        )

        try:
            review_store.upsert_pending(donor.donor_id, result)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Could not persist failed donor to review queue: donor_id=%s",
                donor.donor_id,
            )

        return result
