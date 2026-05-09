"""Shared utility helpers and logger access."""

from datetime import datetime, timezone
from typing import Any, Dict

import legacy_app as legacy

logger = legacy.logger


def now_iso() -> str:
    """Return the current UTC timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def sse_packet(payload: Dict[str, Any]) -> str:
    """Serialize a server-sent-event payload."""

    return legacy.sse_packet(payload)


def sleep_ms(ms: int) -> None:
    """Sleep helper retained for parity in utility imports."""

    legacy.time.sleep(ms / 1000.0)
