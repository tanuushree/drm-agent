"""Encryption-at-rest for Gmail OAuth credentials while they sit in the
in-memory session store.

Previously `session["gmail_credentials"]` held the raw OAuth refresh/access
token JSON in plaintext, in a plain Python dict, for the life of the
process. That's a real exposure: anything that can read process memory (a
debugger, a core dump, a logging accident, a future bug that serializes
`sessions` somewhere) gets a live Gmail token. This module encrypts that
value before it's stored and decrypts it only at the point of use.

Note: this only protects data at rest in memory / if ever persisted; it
does not replace using a real secret manager or a proper session store in
a non-demo deployment.
"""

from __future__ import annotations

import logging

from cryptography.fernet import Fernet, InvalidToken

from drmagent.config import settings

logger = logging.getLogger(__name__)


def _build_fernet() -> Fernet:
    key = settings.session_encryption_key
    if not key:
        # No key configured: generate an ephemeral one. Sessions are
        # in-memory only in this prototype anyway, so they don't outlive
        # the process, but production deployments must set
        # SESSION_ENCRYPTION_KEY so this is stable across restarts and so
        # this warning doesn't fire on every boot.
        logger.warning(
            "SESSION_ENCRYPTION_KEY is not set; generating an ephemeral key. "
            "Set SESSION_ENCRYPTION_KEY in production."
        )
        key = Fernet.generate_key().decode()
    return Fernet(key.encode() if isinstance(key, str) else key)


_fernet = _build_fernet()


def encrypt_credentials(raw_json: str) -> str:
    return _fernet.encrypt(raw_json.encode("utf-8")).decode("utf-8")


def decrypt_credentials(token: str) -> str:
    try:
        return _fernet.decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError(
            "Stored Gmail credentials could not be decrypted (key rotated "
            "or session corrupted); re-authentication is required."
        ) from exc
