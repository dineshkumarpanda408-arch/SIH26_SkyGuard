"""WeatherLock — SkyGuard's single-account visual security mechanism.

The operator authenticates with three factors in one step:

    Username  +  Visual Pattern (sequence of weather icons)  +  Security PIN

Nothing secret is stored in plaintext: the PIN and the icon sequence are both
kept as salted PBKDF2-SHA256 hashes. Session tokens are opaque random strings
held only in memory, with idle-expiry and a temporary lockout after repeated
failed attempts.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from .models import User

# ------------------------------- Weather icons --------------------------------

# Stable ids are what get hashed/compared server-side; the emoji + label are
# presentation only (served to the frontend via GET /auth/pattern).
WEATHER_ICONS: List[Dict[str, str]] = [
    {"id": "thermometer", "emoji": "\U0001F321\ufe0f", "label": "Temperature"},
    {"id": "droplet", "emoji": "\U0001F4A7", "label": "Humidity"},
    {"id": "chart", "emoji": "\U0001F4CA", "label": "Pressure / Data"},
    {"id": "cloud", "emoji": "\u2601\ufe0f", "label": "Cloud"},
    {"id": "rain", "emoji": "\U0001F327\ufe0f", "label": "Rain"},
    {"id": "zap", "emoji": "\u26a1", "label": "Alert"},
    {"id": "sun", "emoji": "\U0001F324\ufe0f", "label": "Weather"},
    {"id": "wind", "emoji": "\U0001F4A8", "label": "Wind"},
    {"id": "bell", "emoji": "\U0001F514", "label": "Notification"},
]

ICON_IDS = [icon["id"] for icon in WEATHER_ICONS]

# --------------------------- Single demo account -----------------------------

DEFAULT_USERNAME = "skyguard"
DEFAULT_PATTERN: List[str] = ["thermometer", "droplet", "chart", "zap"]
DEFAULT_PIN = "739214"

PATTERN_LENGTH = len(DEFAULT_PATTERN)
PIN_LENGTH = len(DEFAULT_PIN)

# ------------------------------- Security policy ------------------------------

MAX_FAILED_ATTEMPTS = 3
LOCKOUT_SECONDS = 30
SESSION_IDLE_SECONDS = 30 * 60  # auto logout after 30 minutes of inactivity
PBKDF2_ITERATIONS = 200_000

# ------------------------------- Hash helpers ---------------------------------


def _hash_secret(secret: str, salt: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    dk = hashlib.pbkdf2_hmac(
        "sha256", secret.encode("utf-8"), salt.encode("utf-8"), iterations
    )
    return dk.hex()


def _verify_secret(secret: str, salt: str, expected: str) -> bool:
    return hmac.compare_digest(_hash_secret(secret, salt), expected)


def _new_salt() -> str:
    return secrets.token_hex(16)


def canonical_pattern(pattern: List[str]) -> str:
    """Normalise an icon sequence (deduplicate order-preserving, lowercase ids)."""
    seen = []
    for icon_id in pattern:
        cleaned = str(icon_id).strip().lower()
        if cleaned in ICON_IDS and cleaned not in seen:
            seen.append(cleaned)
    return ":".join(seen)


# ------------------------------ Account seeding -------------------------------


def ensure_default_user(db: Session) -> User:
    """Create the single WeatherLock account on first startup (idempotent)."""
    user = db.query(User).filter(User.username == DEFAULT_USERNAME).first()
    if user is not None:
        return user

    pattern_text = canonical_pattern(DEFAULT_PATTERN)
    pin_salt = _new_salt()
    pattern_salt = _new_salt()
    user = User(
        username=DEFAULT_USERNAME,
        pin_hash=_hash_secret(DEFAULT_PIN, pin_salt),
        pin_salt=pin_salt,
        pattern_hash=_hash_secret(pattern_text, pattern_salt),
        pattern_salt=pattern_salt,
        pattern_length=PATTERN_LENGTH,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# --------------------------------- Sessions -----------------------------------

_session_lock = threading.Lock()
# token -> {"username": str, "created": float, "last_active": float}
_SESSIONS: Dict[str, Dict[str, object]] = {}


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _session_lock:
        _SESSIONS[token] = {"username": username, "created": now, "last_active": now}
    return token


def get_session(token: Optional[str]) -> Optional[Dict[str, object]]:
    """Return a live session, refreshing its idle timer. None when absent/expired."""
    if not token:
        return None
    with _session_lock:
        session = _SESSIONS.get(token)
        if session is None:
            return None
        now = time.time()
        if now - float(session["last_active"]) > SESSION_IDLE_SECONDS:
            del _SESSIONS[token]  # idle timeout -> auto logout
            return None
        session["last_active"] = now
        return session


def invalidate_session(token: Optional[str]) -> bool:
    if not token:
        return False
    with _session_lock:
        return _SESSIONS.pop(token, None) is not None


def invalidate_all_sessions() -> None:
    with _session_lock:
        _SESSIONS.clear()


def session_expiry(token: str) -> Optional[datetime]:
    with _session_lock:
        session = _SESSIONS.get(token)
        if session is None:
            return None
        expiry = float(session["last_active"]) + SESSION_IDLE_SECONDS
        return datetime.fromtimestamp(expiry, tz=timezone.utc)


# ------------------------------- Rate limiting --------------------------------

_fail_lock = threading.Lock()
# username -> {"fails": int, "locked_until": float}
_FAILED: Dict[str, Dict[str, float]] = {}


def lockout_remaining(username: str) -> int:
    """Seconds left of temporary lockout, or 0 if not locked."""
    now = time.time()
    with _fail_lock:
        state = _FAILED.get(username)
        if not state:
            return 0
        return int(max(0.0, float(state["locked_until"]) - now))


def register_failure(username: str) -> int:
    """Count a failed attempt; returns seconds of lockout applied (0 = not locked)."""
    now = time.time()
    with _fail_lock:
        state = _FAILED.get(username, {"fails": 0, "locked_until": 0.0})
        if float(state["locked_until"]) <= now:
            state = {"fails": int(state["fails"]), "locked_until": 0.0}
        state["fails"] = int(state["fails"]) + 1
        if state["fails"] >= MAX_FAILED_ATTEMPTS:
            state["locked_until"] = now + LOCKOUT_SECONDS
            state["fails"] = 0  # reset the counter once the lock expires
        _FAILED[username] = state
        return int(max(0.0, float(state["locked_until"]) - now))


def clear_failures(username: str) -> None:
    with _fail_lock:
        _FAILED.pop(username, None)


def attempts_left(username: str) -> int:
    with _fail_lock:
        state = _FAILED.get(username)
        if not state:
            return MAX_FAILED_ATTEMPTS
        if float(state["locked_until"]) > time.time():
            return 0
        return max(0, MAX_FAILED_ATTEMPTS - int(state["fails"]))


# --------------------------------- Login flow ----------------------------------


def verify_credentials(
    db: Session, username: str, pattern: List[str], pin: str
) -> Tuple[bool, str]:
    """Username -> pattern -> PIN. Returns (ok, reason_if_failed)."""
    username = (username or "").strip()
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        return False, "unknown_username"

    pattern_text = canonical_pattern(pattern)
    if len(pattern_text.split(":")) != (user.pattern_length or PATTERN_LENGTH):
        return False, "incomplete_pattern"
    if not _verify_secret(pattern_text, user.pattern_salt, user.pattern_hash):
        return False, "invalid_pattern"

    pin_text = (pin or "").strip()
    if len(pin_text) != PIN_LENGTH or not pin_text.isdigit():
        return False, "invalid_pin"
    if not _verify_secret(pin_text, user.pin_salt, user.pin_hash):
        return False, "invalid_pin"

    return True, ""
