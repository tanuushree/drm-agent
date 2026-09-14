import base64
import re
from email.message import EmailMessage as MimeEmailMessage
from email.utils import parsedate_to_datetime
from typing import Any

from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from drmagent.models import DonorConversation, EmailMessage, GmailContext


class GmailService:
    def __init__(self, credentials: Credentials, max_threads: int = 20, max_messages_per_thread: int = 30):
        self.service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        self.max_threads = max_threads
        self.max_messages_per_thread = max_messages_per_thread

    @staticmethod
    def _header(headers: list[dict[str, Any]], name: str) -> str:
        for header in headers:
            if header.get("name", "").lower() == name.lower():
                return header.get("value", "")
        return ""

    @staticmethod
    def _decode(data: str | None) -> str:
        if not data:
            return ""
        padding = "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="ignore")

    @staticmethod
    def _html_to_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["style", "script", "head", "noscript"]):
            tag.decompose()
        for tag in soup.find_all(["br", "p", "div", "tr", "li", "h1", "h2", "h3"]):
            tag.append("\n")
        text = soup.get_text(" ", strip=True).replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()

    @classmethod
    def _body(cls, payload: dict[str, Any]) -> str:
        plain = None
        html = None

        def walk(part: dict[str, Any]) -> None:
            nonlocal plain, html
            mime = part.get("mimeType", "")
            data = part.get("body", {}).get("data")
            if data and mime == "text/plain" and plain is None:
                plain = cls._decode(data)
            elif data and mime == "text/html" and html is None:
                html = cls._decode(data)
            for child in part.get("parts", []) or []:
                walk(child)

        walk(payload)
        if plain and plain.strip():
            return plain.strip()
        if html:
            return cls._html_to_text(html)
        return ""

    def find_thread_ids(self, donor_email: str) -> list[str]:
        # Gmail braces mean OR. Searching both directions captures the full
        # donor↔NGO relationship instead of only donor-authored messages.
        query = f"{{from:{donor_email} to:{donor_email}}}"
        response = self.service.users().threads().list(
            userId="me", q=query, maxResults=self.max_threads
        ).execute()
        return [item["id"] for item in response.get("threads", [])]

    def get_thread(self, donor_id: str, donor_email: str, thread_id: str) -> DonorConversation:
        thread = self.service.users().threads().get(
            userId="me", id=thread_id, format="full"
        ).execute()
        messages: list[EmailMessage] = []

        for raw in thread.get("messages", [])[: self.max_messages_per_thread]:
            payload = raw.get("payload", {})
            headers = payload.get("headers", [])
            date_header = self._header(headers, "Date")
            timestamp = int(raw.get("internalDate", "0"))
            if not timestamp and date_header:
                try:
                    timestamp = int(parsedate_to_datetime(date_header).timestamp() * 1000)
                except (TypeError, ValueError, OverflowError):
                    timestamp = 0

            body = self._body(payload) or raw.get("snippet", "")
            recipients = [
                value.strip()
                for value in self._header(headers, "To").split(",")
                if value.strip()
            ]
            messages.append(
                EmailMessage(
                    id=raw["id"],
                    thread_id=raw.get("threadId", thread_id),
                    subject=self._header(headers, "Subject"),
                    sender=self._header(headers, "From"),
                    recipients=recipients,
                    date=date_header,
                    timestamp=timestamp,
                    body_text=body,
                )
            )

        messages.sort(key=lambda message: message.timestamp)
        return DonorConversation(
            donor_id=donor_id,
            donor_email=donor_email,
            thread_id=thread_id,
            subject=messages[0].subject if messages else "",
            messages=messages,
        )

    def get_donor_conversations(self, donor: Any) -> list[DonorConversation]:
        return [
            self.get_thread(donor.donor_id, donor.email, thread_id)
            for thread_id in self.find_thread_ids(donor.email)
        ]

    def send_reply(
        self,
        thread_id: str,
        message_id: str,
        to: str,
        subject: str,
        body: str,
    ) -> dict[str, Any]:
        """
        Send an email as a reply to an existing Gmail thread.

        Args:
            thread_id: Gmail thread ID.
            message_id: Gmail internal message ID of the message being replied to.
            to: Recipient email address.
            subject: Email subject.
            body: Plain-text email body.

        Returns:
            Gmail API response for the sent message.
        """

        # Fetch the original Gmail message so we can obtain its RFC Message-ID.
        original = (
            self.service.users()
            .messages()
            .get(
                userId="me",
                id=message_id,
                format="metadata",
                metadataHeaders=["Message-ID", "Subject"],
            )
            .execute()
        )

        headers = original.get("payload", {}).get("headers", [])

        parent_message_id = self._header(headers, "Message-ID")

        if not parent_message_id:
            raise ValueError(
                f"Could not find RFC Message-ID for Gmail message {message_id}"
            )

        # Gmail threading requires the RFC message ID headers in addition
        # to the Gmail threadId.
        mime_message = MimeEmailMessage()
        mime_message["To"] = to
        mime_message["Subject"] = subject
        mime_message["In-Reply-To"] = parent_message_id
        mime_message["References"] = parent_message_id
        mime_message.set_content(body)

        encoded_message = base64.urlsafe_b64encode(
            mime_message.as_bytes()
        ).decode("utf-8")

        result = (
            self.service.users()
            .messages()
            .send(
                userId="me",
                body={
                    "raw": encoded_message,
                    "threadId": thread_id,
                },
            )
            .execute()
        )

        return result

    def send_email(
                self,
                to: str,
                subject: str,
                body: str,
            ) -> dict[str, Any]:
                """
                Send a brand-new email that is not a reply to an existing thread.
    
                Args:
                    to: Recipient email address.
                    subject: Email subject.
                    body: Plain-text email body.
    
                Returns:
                    Gmail API response for the sent message.
                """
                mime_message = MimeEmailMessage()
                mime_message["To"] = to
                mime_message["Subject"] = subject
                mime_message.set_content(body)
    
                encoded_message = base64.urlsafe_b64encode(
                    mime_message.as_bytes()
                ).decode("utf-8")
    
                result = (
                    self.service.users()
                    .messages()
                    .send(
                        userId="me",
                        body={
                            "raw": encoded_message,
                        },
                    )
                    .execute()
                )
    
                return result
    

def get_gmail_context(
    conversations: list[DonorConversation],
) -> GmailContext | None:
    """Pick which existing Gmail message a reply should be attached to.

    Strategy: reply to the most recent message across every thread we
    fetched for this donor (by internal timestamp), and thread it into
    that message's thread. This is deliberately simple and deterministic
    — it is never inferred by an LLM — so "what are we replying to" is
    always traceable back to a real, existing Gmail message.

    Returns None when the donor has no conversation history at all (or
    every fetched thread came back with zero messages), which the caller
    uses to skip execution rather than fabricate a thread to reply into.
    """

    latest_message: EmailMessage | None = None
    latest_thread_id: str | None = None

    for conversation in conversations:
        for message in conversation.messages:
            if latest_message is None or message.timestamp > latest_message.timestamp:
                latest_message = message
                latest_thread_id = conversation.thread_id

    if latest_message is None or latest_thread_id is None:
        return None

    return GmailContext(
        thread_id=latest_thread_id,
        message_id=latest_message.id,
    )
