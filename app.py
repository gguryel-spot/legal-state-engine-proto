import os
import time
import hmac
import base64
import hashlib
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

APP_TITLE = "Legal State Engine API (Prototype)"
APP_VERSION = "0.1.0"

API_KEY = os.getenv("LSE_API_KEY", "")  # if set, requires X-API-Key header
SECRET = os.getenv("LSE_SECRET", "dev-secret-change-me")  # change in Render env

SESSIONS: Dict[str, Dict[str, Any]] = {}
PROFILES: Dict[str, Dict[str, Any]] = {}

LEGAL_STATES = {
    "no_relationship",
    "informational_only",
    "pre_contractual",
    "consent_pending",
    "consent_granted",
    "restricted_use",
    "terminated",
}

ACTION_NAMES = {
    "explain",
    "summarize",
    "general_information",
    "generate_content",
    "ask_clarifying_questions",
    "refuse",
    "request_assent",
    "escalate_to_human",
    "provide_links",
    "perform_tool_call",
}

ASSENT_SCOPES = {
    "basic_use",
    "content_generation",
    "external_tool_use",
    "data_processing_ack",
}


class EvaluateRequest(BaseModel):
    request_id: str
    app_id: str
    profile_id: Optional[str] = None
    session_id: str
    event_type: str
    prior_state: Optional[str] = None
    user_message: Optional[str] = None
    tool_context: Optional[Dict[str, Any]] = None
    assent_token: Optional[str] = None
    client_metadata: Optional[Dict[str, Any]] = None


class Constraint(BaseModel):
    rule_id: str
    summary: str


class TransitionOption(BaseModel):
    to_state: str
    requires: str
    rationale: str


class EvaluateResponse(BaseModel):
    request_id: str
    server_time: str
    engine_version: str
    profile_id: Optional[str] = None
    current_state: str
    allowed_actions: List[str]
    prohibited_actions: List[str]
    required_constraints: List[Constraint]
    transition_options: List[TransitionOption]
    response_directives: Dict[str, Any] = Field(default_factory=dict)


class AssentRequest(BaseModel):
    request_id: str
    app_id: str
    session_id: str
    scope: str
    assent_text_presented: str
    user_assent_signal: str
    client_metadata: Optional[Dict[str, Any]] = None


class AssentResponse(BaseModel):
    request_id: str
    server_time: str
    engine_version: str
    new_state: str
    assent_token: str
    token_expires_at: str
    scope: str


class ProfileGenerateRequest(BaseModel):
    request_id: str
    app_id: str
    app_description: str
    intended_actions: Optional[List[str]] = None
    disallowed_actions: Optional[List[str]] = None
    uses_external_tools: bool = False
    data_handling_notes: Optional[str] = None


class ProfileGenerateResponse(BaseModel):
    request_id: str
    server_time: str
    engine_version: str
    profile_id: str
    default_state: str
    suggested_constraints: List[Constraint]
    suggested_instruction_block: Optional[str] = None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and (x_api_key != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")


def token_for(session_id: str, scope: str, issued_at: int) -> str:
    msg = f"{session_id}|{scope}|{issued_at}".encode("utf-8")
    sig = hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(msg + b"." + sig).decode("utf-8")


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        raw = base64.urlsafe_b64decode(token.encode("utf-8"))
        msg, sig = raw.rsplit(b".", 1)
        expected = hmac.new(SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        session_id, scope, issued_at_str = msg.decode("utf-8").split("|", 2)
        return {"session_id": session_id, "scope": scope, "issued_at": int(issued_at_str)}
    except Exception:
        return None


def get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {"state": "informational_only", "assent": None, "history": []}
    return SESSIONS[session_id]


def infer_intent(user_message: Optional[str]) -> str:
    if not user_message:
        return "general"
    msg = user_message.lower()
    if any(k in msg for k in ["write", "draft", "create", "generate", "compose"]):
        return "content_generation"
    return "general"


ENGINE_VERSION = "lse-proto-0.1.0"
app = FastAPI(title=APP_TITLE, version=APP_VERSION)


@app.post("/v1/legal-state/profile/generate", response_model=ProfileGenerateResponse)
def generate_profile(req: ProfileGenerateRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    intended = req.intended_actions or []
    disallowed = req.disallowed_actions or []
    for a in intended + disallowed:
        if a not in ACTION_NAMES:
            raise HTTPException(status_code=400, detail=f"Unknown action: {a}")

    profile_id = f"prof_{int(time.time())}_{hashlib.sha1(req.app_description.encode()).hexdigest()[:8]}"
    PROFILES[profile_id] = {
        "app_id": req.app_id,
        "app_description": req.app_description,
        "intended_actions": intended,
        "disallowed_actions": disallowed,
        "uses_external_tools": req.uses_external_tools,
        "data_handling_notes": req.data_handling_notes,
    }

    suggested_constraints = [
        Constraint(rule_id="LS-BASE-0001", summary="Do not provide legal advice; provide information only."),
        Constraint(rule_id="LS-BASE-0002", summary="Request explicit assent before generating content intended for external use."),
    ]

    return ProfileGenerateResponse(
        request_id=req.request_id,
        server_time=now_iso(),
        engine_version=ENGINE_VERSION,
        profile_id=profile_id,
        default_state="informational_only",
        suggested_constraints=suggested_constraints,
        suggested_instruction_block=None,
    )


@app.post("/v1/legal-state/assent", response_model=AssentResponse)
def record_assent(req: AssentRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    if req.scope not in ASSENT_SCOPES:
        raise HTTPException(status_code=400, detail="Invalid scope")

    if "i agree" not in req.user_assent_signal.lower():
        raise HTTPException(status_code=400, detail="Assent not explicit enough (expected 'I AGREE').")

    sess = get_session(req.session_id)

    issued_at = int(time.time())
    expires_at = issued_at + 3600
    token = token_for(req.session_id, req.scope, issued_at)

    sess["assent"] = {"token": token, "scope": req.scope, "expires_at": expires_at}
    sess["state"] = "consent_granted"
    sess["history"].append({"t": now_iso(), "event": "assent_recorded", "scope": req.scope})

    return AssentResponse(
        request_id=req.request_id,
        server_time=now_iso(),
        engine_version=ENGINE_VERSION,
        new_state="consent_granted",
        assent_token=token,
        token_expires_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_at)),
        scope=req.scope,
    )


@app.post("/v1/legal-state/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    sess = get_session(req.session_id)

    token_valid = False
    scope = None
    if req.assent_token:
        parsed = verify_token(req.assent_token)
        if parsed and parsed["session_id"] == req.session_id:
            stored = sess.get("assent") or {}
            if stored.get("token") == req.assent_token and stored.get("expires_at", 0) > int(time.time()):
                token_valid = True
                scope = parsed["scope"]

    intent = infer_intent(req.user_message)

    required_constraints = [
        Constraint(rule_id="LS-BASE-0001", summary="Do not provide legal advice; provide information only."),
    ]

    transition_options: List[TransitionOption] = []
    prohibited_actions: List[str] = []
    allowed_actions: List[str] = []

    current_state = sess["state"]

    if intent == "content_generation":
        if token_valid and scope in ("content_generation", "basic_use"):
            current_state = "consent_granted"
            sess["state"] = current_state
            allowed_actions = ["generate_content", "ask_clarifying_questions", "refuse"]
            prohibited_actions = []
            required_constraints.append(
                Constraint(rule_id="LS-BASE-0002", summary="Generated content is informational and must be reviewed by a human before use.")
            )
        else:
            current_state = "consent_pending"
            sess["state"] = current_state
            allowed_actions = ["request_assent", "ask_clarifying_questions", "refuse"]
            prohibited_actions = ["generate_content"]
            transition_options = [
                TransitionOption(
                    to_state="consent_granted",
                    requires="explicit_assent_token",
                    rationale="Content generation requires explicit user assent.",
                )
            ]
            required_constraints.append(
                Constraint(rule_id="LS-ASSENT-0001", summary="Before generating content, obtain explicit user assent (e.g., 'I AGREE').")
            )
    else:
        if current_state != "terminated":
            current_state = "informational_only"
            sess["state"] = current_state
        allowed_actions = ["explain", "summarize", "general_information", "ask_clarifying_questions", "refuse"]
        prohibited_actions = ["generate_content"]
        transition_options = [
            TransitionOption(
                to_state="consent_pending",
                requires="explicit_user_action",
                rationale="If the user requests content generation, you must request explicit assent first.",
            )
        ]

    sess["history"].append({"t": now_iso(), "event": req.event_type, "intent": intent})

    return EvaluateResponse(
        request_id=req.request_id,
        server_time=now_iso(),
        engine_version=ENGINE_VERSION,
        profile_id=req.profile_id,
        current_state=current_state,
        allowed_actions=allowed_actions,
        prohibited_actions=prohibited_actions,
        required_constraints=required_constraints,
        transition_options=transition_options,
        response_directives={"intent": intent},
    )


@app.get("/health")
def health():
    return {"ok": True, "time": now_iso(), "engine_version": ENGINE_VERSION}
