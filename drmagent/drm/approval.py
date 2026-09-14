"""Deterministic human-approval policy for DRM actions."""

import re
from typing import Optional

from drmagent.models import (
    ActionClassification,
    ActionPlan,
    DonorConversation,
    DonorProfile,
)


DEFAULT_DONATION_APPROVAL_THRESHOLD = 100_000
DEFAULT_CONFIDENCE_THRESHOLD = 0.70


BANK_PAYMENT_PATTERNS = [
    r"\bbank\s+details?\b",
    r"\bbank\s+account\b",
    r"\baccount\s+(?:number|details?)\b",
    r"\baccount\s+information\b",
    r"\bifsc\b",
    r"\bupi\b",
    r"\bneft\b",
    r"\brtgs\b",
    r"\bwire\s+transfer\b",
    r"\bpayment\s+details?\b",
    r"\bpayment\s+information\b",
    r"\btransfer\s+(?:details?|information)\b",
    r"\bdonation\s+transfer\b",
]


REFUND_CANCELLATION_PATTERNS = [
    r"\brefund\b",
    r"\brefund\s+(?:my|the|this)?\s*donation\b",
    r"\bcancel\s+(?:my|the|this)?\s*donation\b",
    r"\bcancellation\s+(?:of\s+)?(?:my|the|this)?\s*donation\b",
    r"\breverse\s+(?:my|the|this)?\s*(?:donation|payment|transaction)\b",
]


TRANSACTION_DISPUTE_PATTERNS = [
    r"\bunauthori[sz]ed\s+(?:transaction|payment|charge)\b",
    r"\bunknown\s+(?:transaction|payment|charge)\b",
    r"\bwrong\s+(?:transaction|payment|charge|amount)\b",
    r"\bincorrect\s+(?:transaction|payment|charge|amount)\b",
    r"\btransaction\s+(?:dispute|error)\b",
    r"\bpayment\s+(?:dispute|error)\b",
    r"\bcharged\s+(?:twice|double)\b",
    r"\bduplicate\s+(?:transaction|payment|charge)\b",
    r"\bdispute\s+(?:a\s+)?(?:transaction|payment|charge)\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    normalized = " ".join(text.lower().split())

    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in patterns
    )


def _conversation_text(
    conversations: list[DonorConversation],
) -> str:
    parts: list[str] = []

    for conversation in conversations:
        if conversation.subject:
            parts.append(conversation.subject)

        for message in conversation.messages:
            if message.body_text:
                parts.append(message.body_text)

            if message.subject:
                parts.append(message.subject)

    return "\n".join(parts)


def _profile_text(profile: DonorProfile) -> str:
    parts: list[str] = [
        profile.communication_summary,
        *profile.donation_context,
        *profile.requests,
        *profile.commitments_made_by_ngo,
        *profile.unresolved_items,
        *profile.notes,
    ]

    return "\n".join(value for value in parts if value)


def _extract_inr_amounts(text: str) -> list[float]:
    """Extract INR amounts from donor communication."""

    patterns = [
        r"₹\s*([\d,]+(?:\.\d+)?)",
        r"\brs\.?\s*([\d,]+(?:\.\d+)?)",
        r"\binr\s*([\d,]+(?:\.\d+)?)",
        r"([\d,]+(?:\.\d+)?)\s*(?:rupees|inr)\b",
    ]

    amounts: list[float] = []

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            raw_amount = match.group(1).replace(",", "")

            try:
                amounts.append(float(raw_amount))
            except ValueError:
                continue

    return amounts


def _rule_donation_amount(
    text: str,
    threshold: float,
) -> Optional[str]:
    amounts = _extract_inr_amounts(text)

    if any(amount > threshold for amount in amounts):
        return (
            "Donation/payment amount exceeds the human-approval "
            f"threshold of ₹{threshold:,.0f}."
        )

    return None


def _rule_low_confidence(
    classification: ActionClassification,
    confidence_threshold: float,
) -> Optional[str]:
    """Require review when the model is not sufficiently confident."""

    if classification.confidence < confidence_threshold:
        return (
            f"Classification confidence ({classification.confidence:.2f}) "
            f"is below the human-approval threshold "
            f"({confidence_threshold:.2f})."
        )

    return None


def _rule_low_profile_confidence(
    profile: DonorProfile,
    confidence_threshold: float,
) -> Optional[str]:
    """Require review when the extracted donor profile is uncertain."""

    if profile.confidence < confidence_threshold:
        return (
            f"Donor profile confidence ({profile.confidence:.2f}) is below "
            f"the human-approval threshold ({confidence_threshold:.2f})."
        )

    return None


def _rule_bank_or_payment_details(
    text: str,
) -> Optional[str]:
    if _matches_any(text, BANK_PAYMENT_PATTERNS):
        return "Bank or payment details request requires human approval."

    return None


def _rule_refund_or_cancellation(
    text: str,
) -> Optional[str]:
    if _matches_any(text, REFUND_CANCELLATION_PATTERNS):
        return (
            "Donation refund or cancellation request requires "
            "human approval."
        )

    return None


def _rule_transaction_dispute(
    text: str,
) -> Optional[str]:
    if _matches_any(text, TRANSACTION_DISPUTE_PATTERNS):
        return (
            "Transaction or payment dispute requires human approval."
        )

    return None


def _rule_explicit_human_review(
    profile: DonorProfile,
    classification: ActionClassification,
    plan: ActionPlan,
) -> Optional[str]:
    if profile.human_review_required:
        return "Donor profile explicitly requires human review."

    if classification.action == "Human Review":
        return "Classification explicitly requires human review."

    if plan.action == "Human Review":
        return "Action plan explicitly requires human review."

    if plan.requires_human_approval:
        return "Action plan explicitly requests human approval."

    return None


def determine_human_approval(
    profile: DonorProfile,
    classification: ActionClassification,
    plan: ActionPlan,
    conversations: list[DonorConversation],
    donation_threshold: float = DEFAULT_DONATION_APPROVAL_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[bool, list[str]]:
    """Evaluate every deterministic approval rule.

    Returns:
        (
            requires_human_approval,
            list_of_all_triggered_reasons,
        )
    """

    text = (
        f"{_profile_text(profile)}\n"
        f"{_conversation_text(conversations)}"
    )

    reasons: list[str] = []

    # Rule 1: large donation/payment.
    reason = _rule_donation_amount(
        text,
        donation_threshold,
    )
    if reason:
        reasons.append(reason)

    # Rule 2: low classification confidence.
    reason = _rule_low_confidence(
        classification,
        confidence_threshold,
    )
    if reason:
        reasons.append(reason)

    # Rule 3: low extracted-profile confidence.
    reason = _rule_low_profile_confidence(
        profile,
        confidence_threshold,
    )
    if reason:
        reasons.append(reason)

    # Rule 4: bank/payment details.
    reason = _rule_bank_or_payment_details(text)
    if reason:
        reasons.append(reason)

    # Rule 5: refund/cancellation.
    reason = _rule_refund_or_cancellation(text)
    if reason:
        reasons.append(reason)

    # Rule 6: transaction/payment dispute.
    reason = _rule_transaction_dispute(text)
    if reason:
        reasons.append(reason)

    # Rule 7: explicit human review.
    reason = _rule_explicit_human_review(
        profile,
        classification,
        plan,
    )
    if reason:
        reasons.append(reason)

    return bool(reasons), reasons


def apply_human_approval_policy(
    profile: DonorProfile,
    classification: ActionClassification,
    plan: ActionPlan,
    conversations: list[DonorConversation],
    donation_threshold: float = DEFAULT_DONATION_APPROVAL_THRESHOLD,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> tuple[ActionPlan, list[str]]:
    """Apply deterministic approval policy to the action plan."""

    requires_human_approval, reasons = determine_human_approval(
        profile=profile,
        classification=classification,
        plan=plan,
        conversations=conversations,
        donation_threshold=donation_threshold,
        confidence_threshold=confidence_threshold,
    )

    # The deterministic policy is authoritative.
    plan.requires_human_approval = requires_human_approval

    # If any deterministic rule fires, force the action itself to Human
    # Review too (not just the boolean flag) so a downstream consumer that
    # only checks `plan.action` can't mistake this for a sendable action.
    if requires_human_approval:
        plan.action = "Human Review"

    return plan, reasons