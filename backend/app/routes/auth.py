"""auth.py
--------
WeatherLock authentication endpoints: the single SkyGuard account signs in with
Username + Visual Pattern (weather-icon sequence) + Security PIN, and receives
an opaque session token used as a Bearer header on every other API call.
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import auth_security as auth
from ..database import SessionLocal, get_db
from ..models import User

router = APIRouter(prefix="/auth", tags=["auth"])

ERROR_MESSAGES = {
    "unknown_username": "Invalid username.",
    "incomplete_pattern": "Select the full security pattern before logging in.",
    "invalid_pattern": "Invalid security pattern. Please try again.",
    "invalid_pin": "Incorrect security PIN. Please try again.",
}


# ------------------------------- Dependencies --------------------------------


def _bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def require_auth(request: Request) -> str:
    """FastAPI dependency guarding every non-public endpoint.

    Returns the session token when valid (and refreshes its idle timer),
    otherwise raises 401 so the frontend drops back to the WeatherLock screen.
    """
    token = _bearer_token(request)
    session = auth.get_session(token)
    if session is None:
        raise HTTPException(
            status_code=401,
            detail={"message": "Session expired or invalid. Please sign in again."},
        )
    return token or ""


def seed_default_user() -> None:
    """Create the single WeatherLock account if it does not exist yet."""
    db = SessionLocal()
    try:
        auth.ensure_default_user(db)
    finally:
        db.close()


# --------------------------------- Schemas ------------------------------------


class LoginRequest(BaseModel):
    username: str = Field(default="")
    pattern: List[str] = Field(default_factory=list)
    pin: str = Field(default="")


# -------------------------------- Endpoints -----------------------------------


@router.get("/pattern")
def get_pattern():
    """Public: icon catalogue + secret shape (never the secret itself)."""
    return {
        "icons": auth.WEATHER_ICONS,
        "pattern_length": auth.PATTERN_LENGTH,
        "pin_length": auth.PIN_LENGTH,
        "default_username": auth.DEFAULT_USERNAME,
        "max_attempts": auth.MAX_FAILED_ATTEMPTS,
        "lockout_seconds": auth.LOCKOUT_SECONDS,
    }


@router.post("/login")
def login(payload: LoginRequest, db=Depends(get_db)):
    username = (payload.username or "").strip()

    locked_for = auth.lockout_remaining(username)
    if locked_for > 0:
        raise HTTPException(
            status_code=423,
            detail={
                "message": f"Too many attempts. Try again in {locked_for} seconds.",
                "retry_after": locked_for,
            },
        )

    ok, reason = auth.verify_credentials(db, username, payload.pattern, payload.pin)
    if not ok:
        retry_after = auth.register_failure(username)
        if retry_after > 0:
            raise HTTPException(
                status_code=423,
                detail={
                    "message": f"Too many attempts. Try again in {retry_after} seconds.",
                    "retry_after": retry_after,
                },
            )
        raise HTTPException(
            status_code=401,
            detail={
                "message": ERROR_MESSAGES.get(reason, "Authentication failed."),
                "reason": reason,
                "attempts_left": auth.attempts_left(username),
            },
        )

    auth.clear_failures(username)
    token = auth.create_session(username)
    expires = auth.session_expiry(token)
    return {
        "token": token,
        "username": username,
        "expires_at": expires.isoformat() if expires else None,
        "idle_timeout_seconds": auth.SESSION_IDLE_SECONDS,
    }


@router.post("/logout")
def logout(request: Request):
    auth.invalidate_session(_bearer_token(request))
    return {"status": "logged_out"}


@router.get("/me")
def me(request: Request, db=Depends(get_db)):
    token = _bearer_token(request)
    session = auth.get_session(token)  # also refreshes the idle timer
    if session is None:
        raise HTTPException(
            status_code=401,
            detail={"message": "Session expired or invalid. Please sign in again."},
        )

    user = db.query(User).filter(User.username == session["username"]).first()
    expires = auth.session_expiry(token or "")
    return {
        "username": session["username"],
        "authenticated": True,
        "expires_at": expires.isoformat() if expires else None,
        "idle_timeout_seconds": auth.SESSION_IDLE_SECONDS,
        "pattern_length": user.pattern_length if user else auth.PATTERN_LENGTH,
    }
