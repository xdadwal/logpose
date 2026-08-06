"""JWT construction for the Codex suites.

Not named ``test_*`` so pytest does not collect it, matching
:mod:`tests.fake_provider`.
"""

from __future__ import annotations

import base64
import json
from typing import Any


def make_jwt(payload: dict[str, Any], *, header: dict[str, Any] | None = None) -> str:
    """Build an unsigned JWT whose payload segment decodes to ``payload``.

    The signature is a placeholder: nothing under test verifies one, because the
    tokens involved come off the local filesystem and the server re-checks them
    anyway. Padding is stripped, as real tokens do, so callers exercise the
    unpadded base64url path.

    Args:
        payload: Claims to encode.
        header: Optional header claims; a minimal default is used otherwise.

    Returns:
        A three-segment compact-serialization JWT.
    """

    def segment(data: dict[str, Any]) -> str:
        raw = json.dumps(data, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{segment(header or {'alg': 'none', 'typ': 'JWT'})}.{segment(payload)}.sig"
