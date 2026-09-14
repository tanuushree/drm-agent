from collections import defaultdict

from drmagent.gmail.service import GmailService
from drmagent.models import DonorConversation, DonorRecord, DonorProfile
from drmagent.profile.agents import (
    consolidate,
    create_consolidation_agent,
    create_extraction_agent,
    extract,
)


def format_conversations(conversations: list[DonorConversation], max_chars: int) -> str:
    blocks: list[str] = []
    for conversation in conversations:
        lines = [f"--- Thread: {conversation.subject or 'No subject'} ---"]
        for message in conversation.messages:
            if not message.body_text.strip():
                continue
            lines.append(
                f"[{message.date or 'unknown date'}] From: {message.sender}\n"
                f"To: {', '.join(message.recipients)}\n{message.body_text}"
            )
        if len(lines) > 1:
            blocks.append("\n\n".join(lines))

    combined = "\n\n".join(blocks)
    return combined[:max_chars] + ("\n\n[...truncated...]" if len(combined) > max_chars else "")


def build_donor_profile(
    gmail: GmailService,
    donor: DonorRecord,
    max_chars: int,
) -> tuple[DonorProfile, list[DonorConversation]]:
    conversations = gmail.get_donor_conversations(donor)
    conversation_text = format_conversations(conversations, max_chars)

    if not conversation_text.strip():
        return DonorProfile(
            donor_id=donor.donor_id,
            name=donor.name,
            email=donor.email,
            notes=["No Gmail conversation history found."],
            confidence=0.0,
        ), conversations

    consolidation_agent = create_consolidation_agent()
    extraction_agent = create_extraction_agent()

    # Each thread is summarized independently so unrelated donor conversations
    # do not get accidentally merged into one factual narrative.
    summaries = []
    for conversation in conversations:
        thread_text = format_conversations([conversation], max_chars)
        summaries.append(consolidate(consolidation_agent, donor.donor_id, thread_text))

    aggregate = summaries[0] if len(summaries) == 1 else consolidate(
        consolidation_agent,
        donor.donor_id,
        "\n\n--- THREAD SUMMARY ---\n".join(s.model_dump_json() for s in summaries),
    )
    profile = extract(extraction_agent, aggregate, donor.name, donor.email)
    profile.name = donor.name or profile.name
    profile.email = donor.email
    return profile, conversations
