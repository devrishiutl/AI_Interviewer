"""
Shared helpers for `main.py`.

Goal: keep `main.py` focused on request models + API routes only.
All heavy lifting (HROne, LiveKit, transcript persistence, dispatch) lives here.
"""

import asyncio
import io
import json
import logging
import os
import re
import time
import uuid
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException, Request
from pypdf import PdfReader
from livekit import api
from livekit.api.twirp_client import TwirpError
from livekit.protocol import agent as agent_proto
from livekit.protocol import room as room_proto

# Setup logging to file
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "api.log"

_logger = logging.getLogger("ai_interviewer")
_logger.setLevel(logging.DEBUG)

# File handler
_file_handler = logging.FileHandler(_LOG_FILE)
_file_handler.setLevel(logging.DEBUG)
_file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
_file_handler.setFormatter(_file_formatter)
_logger.addHandler(_file_handler)

# Console handler (also log to console)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_formatter = logging.Formatter("%(levelname)s: %(message)s")
_console_handler.setFormatter(_console_formatter)
_logger.addHandler(_console_handler)


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def _short(text: str, n: int = 300) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t if len(t) <= n else (t[:n] + "…")


# -----------------------------------------------------------------------------
# LiveKit room metadata sizing
# -----------------------------------------------------------------------------

# LiveKit hard limit is 65536 bytes. Keep a little headroom.
_ROOM_METADATA_MAX_BYTES = 60_000


def _trim_text(v: object, *, max_len: int) -> object:
    if isinstance(v, str) and len(v) > max_len:
        return v[:max_len] + "…"
    return v


def _room_metadata_json(interview_data: dict) -> str:
    """
    LiveKit enforces a 64KB metadata limit. HROne resumes/descriptions can exceed that.
    We keep required structure but trim/drop large optional fields until it fits.
    """
    md = json.loads(json.dumps(interview_data))  # deep copy with JSON-safe types

    def _size() -> int:
        return len(json.dumps(md, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))

    def _dump() -> str:
        return json.dumps(md, ensure_ascii=False)

    # First: truncate common large fields.
    app = md.get("applicant") or {}
    if isinstance(app, dict) and "resumeText" in app:
        app["resumeText"] = _trim_text(app.get("resumeText"), max_len=2000)
    job = md.get("job") or {}
    if isinstance(job, dict) and "description" in job:
        job["description"] = _trim_text(job.get("description"), max_len=2000)
    rnd = md.get("round") or {}
    if isinstance(rnd, dict) and "instructions" in rnd:
        rnd["instructions"] = _trim_text(rnd.get("instructions"), max_len=1500)
    if isinstance(rnd, dict) and isinstance(rnd.get("questions"), list) and len(rnd["questions"]) > 30:
        rnd["questions"] = rnd["questions"][:30]

    if _size() <= _ROOM_METADATA_MAX_BYTES:
        return _dump()

    # If still too big: drop biggest optional parts in order.
    drops: list[tuple[str, list[str]]] = [
        ("applicant.resumeText", ["applicant", "resumeText"]),
        ("job.description", ["job", "description"]),
        ("round.instructions", ["round", "instructions"]),
        ("round.questions", ["round", "questions"]),
        ("interviewer.description", ["interviewer", "description"]),
    ]
    for label, path in drops:
        cur = md
        for key in path[:-1]:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(key)
        if isinstance(cur, dict) and path[-1] in cur:
            cur.pop(path[-1], None)
            if _size() <= _ROOM_METADATA_MAX_BYTES:
                return _dump()

    # Last resort: keep only essentials used by agent, but still valid.
    essential = {
        "interviewId": md.get("interviewId"),
        "transcriptRecordId": md.get("transcriptRecordId"),
        "transcriptApiUrl": md.get("transcriptApiUrl"),
        "job": (md.get("job") or {}) if isinstance(md.get("job"), dict) else {},
        "applicant": (md.get("applicant") or {}) if isinstance(md.get("applicant"), dict) else {},
        "round": (md.get("round") or {}) if isinstance(md.get("round"), dict) else {},
        "interviewer": (md.get("interviewer") or {}) if isinstance(md.get("interviewer"), dict) else {},
    }
    md = essential
    return _dump()


# -----------------------------------------------------------------------------
# LiveKit cleanup (legacy room names)
# -----------------------------------------------------------------------------

_LEGACY_ROOM_RE = re.compile(r"^interview-\d+$")


async def _cleanup_legacy_numeric_rooms(lk: api.LiveKitAPI, *, keep_room: str) -> None:
    """
    Older versions used numeric interview ids (e.g. interview-1).
    Those rooms/dispatches can remain pending and get assigned to your worker later,
    making it look like the agent "joined a different room".

    We no longer support that flow, so delete legacy numeric rooms + their dispatches.
    """
    with suppress(Exception):
        lr = await lk.room.list_rooms(api.ListRoomsRequest())
        rooms = getattr(lr, "rooms", None) or []
        # proceed to cleanup below
        pass
    # if list_rooms failed
    if "rooms" not in locals():
        return

    for r in rooms:
        name = getattr(r, "name", None)
        if not isinstance(name, str) or not name or name == keep_room:
            continue
        if not _LEGACY_ROOM_RE.match(name):
            continue

        # Delete dispatches for that room.
        with suppress(Exception):
            existing = await lk.agent_dispatch.list_dispatch(room_name=name)
            for d in existing or []:
                did = getattr(d, "id", None)
                if isinstance(did, str) and did:
                    with suppress(Exception):
                        await lk.agent_dispatch.delete_dispatch(dispatch_id=did, room_name=name)

        # Kick participants, then delete room.
        with suppress(Exception):
            lp = await lk.room.list_participants(api.ListParticipantsRequest(room=name))
            participants = getattr(lp, "participants", None) or []
            for p in participants:
                ident = getattr(p, "identity", None)
                if isinstance(ident, str) and ident:
                    with suppress(Exception):
                        await lk.room.remove_participant(room_proto.RoomParticipantIdentity(room=name, identity=ident))

        with suppress(Exception):
            await lk.room.delete_room(api.DeleteRoomRequest(room=name))


# -----------------------------------------------------------------------------
# Env / Config
# -----------------------------------------------------------------------------

_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name} (expected in {_DOTENV_PATH})")
    return v


LIVEKIT_URL = _require_env("LIVEKIT_URL")
LIVEKIT_API_KEY = _require_env("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = _require_env("LIVEKIT_API_SECRET")
LIVEKIT_AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "interview-agent")
PUBLIC_API_URL = os.getenv("PUBLIC_API_URL", "")  # optional; needed for remote agent
LIVEKIT_EMPTY_TIMEOUT_S = int(os.getenv("LIVEKIT_EMPTY_TIMEOUT_S", "300"))
LIVEKIT_DEPARTURE_TIMEOUT_S = int(os.getenv("LIVEKIT_DEPARTURE_TIMEOUT_S", "60"))

HRONE_API = os.getenv("HRONE_API_URL", "https://api.hrone.studio/api")
APP_ID = _require_env("HRONE_APP_ID")
ORG_ID = _require_env("HRONE_ORG_ID")

JOBS_OBJECT_ID = _require_env("HRONE_JOBS_OBJECT_ID")
APPLICANTS_OBJECT_ID = _require_env("HRONE_APPLICANTS_OBJECT_ID")
ROUNDS_OBJECT_ID = _require_env("HRONE_ROUNDS_OBJECT_ID")
INTERVIEWERS_OBJECT_ID = _require_env("HRONE_INTERVIEWERS_OBJECT_ID")

TRANSCRIPTS_OBJECT_ID = _require_env("HRONE_TRANSCRIPTS_OBJECT_ID")
TRANSCRIPTS_VIEW_ID = os.getenv("HRONE_TRANSCRIPTS_VIEW_ID", "")
TRANSCRIPTS_FIELD_INTERVIEW_ID = os.getenv("HRONE_TRANSCRIPTS_FIELD_INTERVIEW_ID", "interviewId")
TRANSCRIPTS_FIELD_TRANSCRIPT_ID = os.getenv("HRONE_TRANSCRIPTS_FIELD_TRANSCRIPT_ID", "transcriptId")
TRANSCRIPTS_FIELD_TRANSCRIPT_JSON = os.getenv("HRONE_TRANSCRIPTS_FIELD_TRANSCRIPT_JSON", "transcriptJson")

TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID = os.getenv("HRONE_TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID", "")
TRANSCRIPTS_PROP_ID_INTERVIEW_ID = os.getenv("HRONE_TRANSCRIPTS_PROP_ID_INTERVIEW_ID", "")
TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON = os.getenv("HRONE_TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON", "")
TRANSCRIPTS_PROP_ID_ROLE = os.getenv("HRONE_TRANSCRIPTS_PROP_ID_ROLE", "")
TRANSCRIPTS_PROP_ID_SPEAKER_NAME = os.getenv("HRONE_TRANSCRIPTS_PROP_ID_SPEAKER_NAME", "")
TRANSCRIPTS_PROP_ID_TEXT = os.getenv("HRONE_TRANSCRIPTS_PROP_ID_TEXT", "")
TRANSCRIPTS_PROP_ID_TIMESTAMP = os.getenv("HRONE_TRANSCRIPTS_PROP_ID_TIMESTAMP", "")

# Feedback table
FEEDBACK_OBJECT_ID = os.getenv("HRONE_FEEDBACK_OBJECT_ID", "")
FEEDBACK_FIELD_INTERVIEW_ID = os.getenv("HRONE_FEEDBACK_FIELD_INTERVIEW_ID", "interviewId")
FEEDBACK_FIELD_EXPERIENCE = os.getenv("HRONE_FEEDBACK_FIELD_EXPERIENCE", "experience")
FEEDBACK_FIELD_RATING = os.getenv("HRONE_FEEDBACK_FIELD_RATING", "rating")
FEEDBACK_FIELD_TIMESTAMP = os.getenv("HRONE_FEEDBACK_FIELD_TIMESTAMP", "timestamp")

FEEDBACK_PROP_ID_INTERVIEW_ID = os.getenv("HRONE_FEEDBACK_PROP_ID_INTERVIEW_ID", "")
FEEDBACK_PROP_ID_EXPERIENCE = os.getenv("HRONE_FEEDBACK_PROP_ID_EXPERIENCE", "")
FEEDBACK_PROP_ID_RATING = os.getenv("HRONE_FEEDBACK_PROP_ID_RATING", "")
FEEDBACK_PROP_ID_TIMESTAMP = os.getenv("HRONE_FEEDBACK_PROP_ID_TIMESTAMP", "")

# Interviews table
INTERVIEWS_OBJECT_ID = _require_env("HRONE_INTERVIEWS_OBJECT_ID")


def _can_write_transcript_json() -> bool:
    return all(
        bool(x)
        for x in (
            TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON,
            TRANSCRIPTS_PROP_ID_ROLE,
            TRANSCRIPTS_PROP_ID_SPEAKER_NAME,
            TRANSCRIPTS_PROP_ID_TEXT,
            TRANSCRIPTS_PROP_ID_TIMESTAMP,
        )
    )


# -----------------------------------------------------------------------------
# Small in-memory state (single API process)
# -----------------------------------------------------------------------------

_HRONE_TOKEN_BY_INTERVIEW_ID: dict[str, str] = {}
_HRONE_TOKEN_BY_RECORD_ID: dict[str, str] = {}
_TRANSCRIPT_RECORD_BY_INTERVIEW_ID: dict[str, str] = {}


# -----------------------------------------------------------------------------
# LiveKit helpers
# -----------------------------------------------------------------------------

async def ensure_room_with_metadata(lk: api.LiveKitAPI, *, room: str, md_json: str) -> None:
    """Ensure the room exists and has metadata set."""
    try:
        await lk.room.update_room_metadata(api.UpdateRoomMetadataRequest(room=room, metadata=md_json))
        return
    except TwirpError as e:
        if getattr(e, "code", None) != "not_found":
            raise

    await lk.room.create_room(
        api.CreateRoomRequest(
            name=room,
            metadata=md_json,
            empty_timeout=LIVEKIT_EMPTY_TIMEOUT_S,
            departure_timeout=LIVEKIT_DEPARTURE_TIMEOUT_S,
        )
    )
    # Try update again for eventual consistency edge cases.
    try:
        await lk.room.update_room_metadata(api.UpdateRoomMetadataRequest(room=room, metadata=md_json))
    except TwirpError as e:
        if getattr(e, "code", None) == "not_found":
            await asyncio.sleep(0.2)
            await lk.room.update_room_metadata(api.UpdateRoomMetadataRequest(room=room, metadata=md_json))
        else:
            raise


async def ensure_agent_dispatched(lk: api.LiveKitAPI, *, room: str) -> api.AgentDispatch:
    """
    Ensure there's an agent in the room (or at least a valid dispatch queued).
    Avoid stacking duplicates.
    """
    agent_present = False
    participants_known = False
    try:
        lp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        participants_known = True
        participants = getattr(lp, "participants", None) or []
        agent_present = any(
            isinstance(getattr(p, "identity", None), str)
            and getattr(p, "identity")
            and str(getattr(p, "identity")).startswith("agent-")
            for p in participants
        )
    except Exception:
        # If we can't list participants, don't assume the agent is absent; avoid cancelling jobs.
        participants_known = False

    existing_valid = None
    existing_status = None
    existing_jobs: list | None = None
    try:
        dispatches = await lk.agent_dispatch.list_dispatch(room_name=room)
        for d in dispatches or []:
            if getattr(d, "room", None) == room and getattr(d, "agent_name", None) == LIVEKIT_AGENT_NAME:
                existing_valid = d
                with suppress(Exception):
                    st = getattr(d, "state", None)
                    jobs = getattr(st, "jobs", None) or []
                    existing_jobs = jobs
                    if jobs:
                        js = getattr(jobs[0], "state", None)
                        existing_status = getattr(js, "status", None)
                break
    except TwirpError as e:
        if getattr(e, "code", None) != "not_found":
            raise

    if agent_present:
        return existing_valid or await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(room=room, agent_name=LIVEKIT_AGENT_NAME)
        )

    if existing_valid is not None:
        # Re-dispatch if terminal.
        if existing_status in (2, 3):  # JS_SUCCESS=2, JS_FAILED=3
            did = getattr(existing_valid, "id", None)
            if isinstance(did, str) and did:
                with suppress(Exception):
                    await lk.agent_dispatch.delete_dispatch(dispatch_id=did, room_name=room)
            return await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(room=room, agent_name=LIVEKIT_AGENT_NAME)
            )

        # If dispatch claims there is an active job but we can *confirm* no agent is in the room,
        # it's very likely a stuck job (worker restarted/crashed). Recreate the dispatch.
        #
        # We ONLY do this when participant listing succeeded (`participants_known=True`) to avoid
        # accidental duplicate dispatches when the presence check is unavailable.
        if participants_known and (not agent_present) and isinstance(existing_status, int):
            with suppress(Exception):
                if agent_proto.JobStatus.Name(existing_status) == "JS_RUNNING":
                    did = getattr(existing_valid, "id", None)
                    if isinstance(did, str) and did:
                        with suppress(Exception):
                            await lk.agent_dispatch.delete_dispatch(dispatch_id=did, room_name=room)
                    return await lk.agent_dispatch.create_dispatch(
                        api.CreateAgentDispatchRequest(room=room, agent_name=LIVEKIT_AGENT_NAME)
                    )

        # If there is an existing dispatch with an active job/state, keep it (idempotent).
        if existing_jobs is not None and len(existing_jobs) > 0:
            return existing_valid

        # Wait briefly for jobs to appear (eventually consistent)
        did = getattr(existing_valid, "id", None)
        if isinstance(did, str) and did and participants_known:
            try:
                await asyncio.wait_for(
                    _wait_for_dispatch_jobs(lk, did, room),
                    timeout=1.0
                )
                return existing_valid
            except asyncio.TimeoutError:
                with suppress(Exception):
                    await lk.agent_dispatch.delete_dispatch(dispatch_id=did, room_name=room)
                return await lk.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(room=room, agent_name=LIVEKIT_AGENT_NAME)
                )
        return existing_valid
    
    # No existing dispatch and no agent present - create a new dispatch
    return await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(room=room, agent_name=LIVEKIT_AGENT_NAME)
    )


async def _wait_for_dispatch_jobs(lk: api.LiveKitAPI, dispatch_id: str, room: str) -> None:
    """Wait for jobs to appear in dispatch."""
    while True:
        with suppress(Exception):
            d = await lk.agent_dispatch.get_dispatch(dispatch_id=dispatch_id, room_name=room)
            if d and getattr(getattr(d, "state", None), "jobs", None):
                return
        await asyncio.sleep(0.2)


# -----------------------------------------------------------------------------
# HROne helpers
# -----------------------------------------------------------------------------

def _extract_access_token(request: Request | None) -> str | None:
    token, _source = _extract_access_token_with_source(request)
    return token


def _extract_access_token_with_source(request: Request | None) -> tuple[str | None, str]:
    """
    Returns (token, source) where source is one of:
    - "authorization"
    - "x-hrone-access-token"
    - "cookie"
    - "none"
    """
    if request is None:
        return None, "none"

    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        tok = auth.split(" ", 1)[1].strip() or None
        return tok, ("authorization" if tok else "none")

    header_token = request.headers.get("x-hrone-access-token") or request.headers.get("X-HROne-Access-Token")
    if header_token:
        tok = header_token.strip() or None
        return tok, ("x-hrone-access-token" if tok else "none")

    cookie_token = request.cookies.get("access_token")
    if cookie_token:
        tok = cookie_token.strip() or None
        return tok, ("cookie" if tok else "none")

    return None, "none"


def _hrone_headers(access_token: str | None) -> dict:
    token = (access_token or "").strip()
    if not token:
        raise HTTPException(401, "Missing HROne access_token (Authorization: Bearer <token> OR cookie access_token)")
    return {
        "Content-Type": "application/json",
        "x-app-id": APP_ID,
        "x-org-id": ORG_ID,
        "Cookie": f"access_token={token}",
    }


def _values_payload(values: list[dict]) -> dict:
    pids = [v["propertyId"] for v in values if isinstance(v.get("propertyId"), str) and v.get("propertyId")]
    return {"values": values, **({"propertyIds": pids} if pids else {})}


def extract_field_value(record, field_key: str):
    # Some endpoints return {"values":[{"key":..,"value":..}]}; some return sections/fields.
    if isinstance(record, list):
        for section in record:
            if isinstance(section, dict) and "fields" in section:
                for f in section.get("fields", []) or []:
                    if isinstance(f, dict) and f.get("key") == field_key:
                        return f.get("value")
    if isinstance(record, dict):
        for v in record.get("values", []) or []:
            if isinstance(v, dict) and v.get("key") == field_key:
                return v.get("value")
    return None


def extract_skills(record) -> list:
    skills_data = extract_field_value(record, "requiredSkills")
    if not skills_data or not isinstance(skills_data, list):
        return []
    skills = []
    for skill_group in skills_data:
        if isinstance(skill_group, list):
            for item in skill_group:
                if isinstance(item, dict) and item.get("key") == "skill":
                    skills.append(item.get("value"))
    return skills


def extract_round_questions(record) -> list:
    questions_data = extract_field_value(record, "_questions")
    if not questions_data or not isinstance(questions_data, list):
        return []
    questions = []
    for question_group in questions_data:
        if isinstance(question_group, list):
            question_text = None
            for item in question_group:
                if isinstance(item, dict) and item.get("key") == "_question":
                    question_text = item.get("value")
            if question_text:
                questions.append(question_text.strip() if isinstance(question_text, str) else str(question_text))
        elif isinstance(question_group, str):
            questions.append(question_group.strip())
    return questions


def extract_interviewer_skills(record) -> list:
    skills_data = extract_field_value(record, "interviewerSkills")
    if not skills_data or not isinstance(skills_data, list):
        return []
    skills = []
    for skill_group in skills_data:
        if isinstance(skill_group, list):
            for item in skill_group:
                if isinstance(item, dict) and item.get("key") == "skill":
                    skills.append(item.get("value"))
    return skills


def extract_personality_traits(record) -> dict:
    traits_data = extract_field_value(record, "personalityTraits")
    if not traits_data or not isinstance(traits_data, list):
        return {}
    traits = {}
    for item in traits_data:
        if isinstance(item, dict) and "key" in item and "value" in item:
            traits[item["key"]] = item["value"]
    return traits


async def extract_resume_text(file_path: str, client: httpx.AsyncClient, access_token: str | None) -> str | None:
    if not file_path:
        return None
    try:
        url = f"{HRONE_API}/storage-accounts/lego/download"
        full_url = f"{url}?name={quote(file_path, safe='')}"
        res = await client.get(full_url, headers=_hrone_headers(access_token))
        if res.status_code != 200:
            return None
        txt = res.text.strip()
        if txt.startswith('"http'):
            actual = txt.strip('"')
            pdf_res = await client.get(actual)
            if pdf_res.status_code != 200:
                return None
            pdf_bytes = pdf_res.content
        else:
            pdf_bytes = res.content
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
        return "\n".join(parts) if parts else None
    except Exception:
        return None


async def fetch_interview_data(*, job_id: str, applicant_id: str, round_id: str, interviewer_id: str, access_token: str | None) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        def _url(object_id: str, record_id: str) -> str:
            return f"{HRONE_API}/objects/{object_id}/records/{record_id}"

        job = await client.get(_url(JOBS_OBJECT_ID, job_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})
        applicant = await client.get(_url(APPLICANTS_OBJECT_ID, applicant_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})
        round_ = await client.get(_url(ROUNDS_OBJECT_ID, round_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})
        interviewer = await client.get(_url(INTERVIEWERS_OBJECT_ID, interviewer_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})

        missing: list[str] = []
        if job.status_code != 200:
            missing.append("job")
        if applicant.status_code != 200:
            missing.append("applicant")
        if round_.status_code != 200:
            missing.append("round")
        if interviewer.status_code != 200:
            missing.append("interviewer")
        if missing:
            raise HTTPException(502, f"Missing data from HROne: {', '.join(missing)}")

        job_data = job.json()
        applicant_data = applicant.json()
        round_data = round_.json()
        interviewer_data = interviewer.json()

        attachment = extract_field_value(applicant_data, "attachment")
        resume_text = None
        if isinstance(attachment, list) and attachment:
            resume_text = await extract_resume_text(str(attachment[0]), client, access_token)

        skills = extract_skills(job_data)
        min_exp = extract_field_value(job_data, "minExp")
        max_exp = extract_field_value(job_data, "maxExp")
        experience = f"{min_exp}-{max_exp} years" if min_exp and max_exp else None

        # If applicant name is very short / missing, fall back to a readable name from email.
        applicant_name = extract_field_value(applicant_data, "name")
        if not (isinstance(applicant_name, str) and applicant_name.strip() and len(applicant_name.strip()) >= 2):
            em = extract_field_value(applicant_data, "email")
            if isinstance(em, str) and "@" in em:
                base = em.split("@", 1)[0].replace(".", " ").replace("_", " ").strip()
                applicant_name = " ".join(w[:1].upper() + w[1:] for w in base.split() if w)

        return {
            "job": {
                "title": extract_field_value(job_data, "title"),
                "description": extract_field_value(job_data, "description"),
                "industry": extract_field_value(job_data, "industry"),
                "jobLevel": extract_field_value(job_data, "jobLevel"),
                "jobType": extract_field_value(job_data, "jobType"),
                "experience": experience,
                "skills": ", ".join(skills) if skills else None,
            },
            "applicant": {
                "name": applicant_name,
                "email": extract_field_value(applicant_data, "email"),
                "phone": extract_field_value(applicant_data, "phone"),
                "resumeText": resume_text,
            },
            "round": {
                "name": extract_field_value(round_data, "roundName"),
                "type": extract_field_value(round_data, "roundType"),
                "duration": extract_field_value(round_data, "duration"),
                "objective": extract_field_value(round_data, "roundObjective"),
                "language": extract_field_value(round_data, "language"),
                "instructions": extract_field_value(round_data, "interviewInstructions"),
                "questionsType": extract_field_value(round_data, "questionsType"),
                "numOfAiQuestions": extract_field_value(round_data, "numOfAiQuestions"),
                "questions": extract_round_questions(round_data),
            },
            "interviewer": {
                "name": extract_field_value(interviewer_data, "name"),
                "description": extract_field_value(interviewer_data, "description"),
                "skills": extract_interviewer_skills(interviewer_data),
                "voice": extract_field_value(interviewer_data, "voice"),
                "language": extract_field_value(interviewer_data, "language"),
                "roundType": extract_field_value(interviewer_data, "roundType"),
                "personalityTraits": extract_personality_traits(interviewer_data),
            },
        }


async def hrone_find_record_id_by_transcript_id(*, object_id: str, view_id: str, transcript_id: str, access_token: str | None):
    if not (object_id and view_id and transcript_id):
        return None
    async with httpx.AsyncClient(timeout=15.0) as client:
        url = f"{HRONE_API}/objects/{object_id}/views/{view_id}/records"
        res = await client.post(
            url,
            headers=_hrone_headers(access_token),
            params={"limit": 15, "offset": 0},
            json={
                "filters": {"$and": [{"key": "#.records.transcriptId", "operator": "$eq", "value": transcript_id, "type": "singleLineText"}]},
                "appId": APP_ID,
            },
        )
        return res


async def hrone_create_record(*, object_id: str, values: list[dict], access_token: str | None) -> str:
    async with httpx.AsyncClient(timeout=15.0) as client:
        url = f"{HRONE_API}/objects/{object_id}/records"
        res = await client.post(
            url,
            headers=_hrone_headers(access_token),
            params={"appId": APP_ID},
            json=_values_payload(values),
        )
        if res.status_code not in (200, 201):
            print(f"⚠️ HROne create failed: POST {url} -> {res.status_code} {_short(res.text)}")
            raise HTTPException(res.status_code, f"HROne create failed: {res.text}")
        try:
            payload = res.json()
        except Exception:
            # If API doesn't return JSON, fall back to view lookup by transcriptId.
            return ""
        # Common shapes: {"data":{"id":"..."}} or {"id":"..."}
        rid = None
        if isinstance(payload, dict):
            if isinstance(payload.get("id"), str):
                rid = payload.get("id")
            elif isinstance(payload.get("data"), dict) and isinstance(payload["data"].get("id"), str):
                rid = payload["data"].get("id")
        return rid or ""


async def hrone_get_record(*, object_id: str, record_id: str, access_token: str | None):
    async with httpx.AsyncClient(timeout=15.0) as client:
        url = f"{HRONE_API}/objects/{object_id}/records/{record_id}"
        res = await client.get(url, headers=_hrone_headers(access_token), params={"appId": APP_ID})
        return res


async def hrone_update_record(*, object_id: str, record_id: str, values: list[dict], access_token: str | None) -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        url = f"{HRONE_API}/objects/{object_id}/records/{record_id}"
        res = await client.patch(
            url,
            headers=_hrone_headers(access_token),
            params={"appId": APP_ID},
            json=_values_payload(values),
        )
        if res.status_code not in (200, 204):
            print(f"⚠️ HROne update failed: PATCH {url} -> {res.status_code} {_short(res.text)}")
            raise HTTPException(res.status_code, f"HROne update failed: {res.text}")


def _lower(v) -> str:
    if isinstance(v, dict):
        v = v.get("label") or v.get("id")
    return (str(v or "")).strip().lower()


async def resolve_ids_from_interview(*, interview_id: str, email: str, access_token: str | None) -> tuple[str, str, str, str]:
    res = await hrone_get_record(object_id=INTERVIEWS_OBJECT_ID, record_id=str(interview_id), access_token=access_token)
    if res.status_code == 401:
        raise HTTPException(401, f"Unauthorized access to HROne API: {res.text}")
    if res.status_code != 200:
        raise HTTPException(res.status_code, f"HROne API error: {res.text}")
    interview_rec = res.json()

    applicant_id = extract_field_value(interview_rec, "applicantEmail") or extract_field_value(interview_rec, "applicantId")
    job_id = extract_field_value(interview_rec, "jobTitle") or extract_field_value(interview_rec, "jobId")
    round_id = extract_field_value(interview_rec, "roundName") or extract_field_value(interview_rec, "roundId")
    interviewer_id = extract_field_value(interview_rec, "interviewerName") or extract_field_value(interview_rec, "interviewerId")

    missing: list[str] = []
    if not applicant_id:
        missing.append("applicantEmail/applicantId")
    if not job_id:
        missing.append("jobTitle/jobId")
    if not round_id:
        missing.append("roundName/roundId")
    if not interviewer_id:
        missing.append("interviewerName/interviewerId")
    if missing:
        raise HTTPException(500, f"Interview record missing fields: {', '.join(missing)}")

    res = await hrone_get_record(object_id=APPLICANTS_OBJECT_ID, record_id=str(applicant_id), access_token=access_token)
    if res.status_code != 200:
        raise HTTPException(res.status_code, f"HROne API error: {res.text}")
    applicant_rec = res.json()
    expected_email = extract_field_value(applicant_rec, "email")
    if not expected_email:
        raise HTTPException(500, "Applicant record missing email")
    if _lower(expected_email) != _lower(email):
        raise HTTPException(403, f"Email mismatch for this interviewId (expected: {expected_email})")

    return str(applicant_id), str(job_id), str(round_id), str(interviewer_id)


async def _create_transcript_record(interview_id: str, access_token: str | None) -> str | None:
    """Create a new transcript record in HROne."""
    transcript_id = uuid.uuid4().hex
    values = [
        {"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_ID, "value": transcript_id}
        if TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID else {"key": TRANSCRIPTS_FIELD_TRANSCRIPT_ID, "value": transcript_id},
        {"propertyId": TRANSCRIPTS_PROP_ID_INTERVIEW_ID, "key": TRANSCRIPTS_FIELD_INTERVIEW_ID, "value": interview_id}
        if TRANSCRIPTS_PROP_ID_INTERVIEW_ID else {"key": TRANSCRIPTS_FIELD_INTERVIEW_ID, "value": interview_id},
        {"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": []}
        if TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON else {"key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": []},
    ]
    
    record_id = await hrone_create_record(object_id=TRANSCRIPTS_OBJECT_ID, values=values, access_token=access_token)
    if TRANSCRIPTS_VIEW_ID and not record_id:
        res = await hrone_find_record_id_by_transcript_id(
            object_id=TRANSCRIPTS_OBJECT_ID,
            view_id=TRANSCRIPTS_VIEW_ID,
            transcript_id=transcript_id,
            access_token=access_token,
        )
        if res and res.status_code == 200:
            payload = res.json()
            records = payload.get("data", [])
            if records and isinstance(records[0], dict):
                record_id = records[0].get("id")
    return record_id


def _create_livekit_token(identity: str, name: str, room: str) -> api.AccessToken:
    """Create LiveKit access token."""
    return (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(name or "Participant")
        .with_grants(api.VideoGrants(
            room_join=True,
            room=room,
            can_subscribe=True,
            can_publish=True,
            can_publish_data=True
        ))
        .with_ttl(timedelta(hours=1))
    )


def _token_for_transcript(request: Request, *, interview_id: str, record_id: str | None) -> str:
    t = _extract_access_token(request)
    if t:
        return t
    if record_id and record_id in _HRONE_TOKEN_BY_RECORD_ID:
        return _HRONE_TOKEN_BY_RECORD_ID[record_id]
    if interview_id in _HRONE_TOKEN_BY_INTERVIEW_ID:
        return _HRONE_TOKEN_BY_INTERVIEW_ID[interview_id]
    raise HTTPException(401, "Missing HROne access_token (call /api/start-interview with token first)")


# -----------------------------------------------------------------------------
# Public route handlers (called by main.py)
# -----------------------------------------------------------------------------

async def handle_start_interview(*, interview_id: str, email: str, request: Request) -> dict:
    hrone_token, hrone_token_source = _extract_access_token_with_source(request)
    if hrone_token:
        _HRONE_TOKEN_BY_INTERVIEW_ID[interview_id] = hrone_token
    applicant_id, job_id, round_id, interviewer_id = await resolve_ids_from_interview(
        interview_id=interview_id, email=email, access_token=hrone_token
    )

    interview_data = await fetch_interview_data(
        job_id=job_id,
        applicant_id=applicant_id,
        round_id=round_id,
        interviewer_id=interviewer_id,
        access_token=hrone_token,
    )

    participant_name = (interview_data.get("applicant") or {}).get("name") or email or "Candidate"
    interview_data["interviewId"] = interview_id
    if PUBLIC_API_URL:
        interview_data["transcriptApiUrl"] = f"{PUBLIC_API_URL.rstrip('/')}/api/transcript"

    # Create transcript record
    transcript_record_id = await _create_transcript_record(interview_id, hrone_token) if TRANSCRIPTS_OBJECT_ID else None

    if transcript_record_id:
        # overwrite cache so /api/transcript can still work even if transcriptRecordId isn't sent
        _TRANSCRIPT_RECORD_BY_INTERVIEW_ID[interview_id] = transcript_record_id
        interview_data["transcriptRecordId"] = transcript_record_id
        if hrone_token:
            _HRONE_TOKEN_BY_RECORD_ID[transcript_record_id] = hrone_token

    # Setup LiveKit room and token
    room = f"interview-{interview_id}"
    identity = f"candidate-{applicant_id}"
    token = _create_livekit_token(identity, participant_name, room)

    # Setup LiveKit room and wait for agent
    agent_joined, agent_present_now, dispatch_status, agent_participant_identity, dispatch_id, dispatch_worker_id = await _setup_livekit_room(
        room, interview_data
    )

    return {
        "success": True,
        "token": token.to_jwt(),
        "room": room,
        "livekitUrl": os.getenv("PUBLIC_LIVEKIT_URL", LIVEKIT_URL),
        "livekitAgentName": LIVEKIT_AGENT_NAME,
        "identity": identity,
        "transcriptRecordId": transcript_record_id,
        "hroneTokenPresent": bool(hrone_token),
        "hroneTokenSource": hrone_token_source,
        "agentJoined": agent_joined,
        "dispatchId": dispatch_id,
        "dispatchWorkerId": dispatch_worker_id,
        "agentParticipantIdentity": agent_participant_identity,
        "dispatchStatus": dispatch_status,
        "agentPresentNow": agent_present_now,
        "resumed": False,
        "interview_data": interview_data,
    }


async def _setup_livekit_room(room: str, interview_data: dict) -> tuple[bool, bool, str | None, str | None, str | None, str | None]:
    """Setup LiveKit room, dispatch agent, and return status."""
    lk = api.LiveKitAPI(LIVEKIT_URL)
    try:
        await _cleanup_legacy_numeric_rooms(lk, keep_room=room)
        md_json = _room_metadata_json(interview_data)
        await ensure_room_with_metadata(lk, room=room, md_json=md_json)
        try:
            dispatch = await ensure_agent_dispatched(lk, room=room)
            dispatch_id = getattr(dispatch, "id", None) if dispatch else None
            if dispatch_id:
                _logger.info(f"Dispatch created: id={dispatch_id}")
        except Exception as e:
            _logger.exception(f"Failed to create dispatch: {e}")
            dispatch = None
            dispatch_id = None

        try:
            await asyncio.wait_for(_wait_for_agent_join(lk, room), timeout=30.0)
            agent_joined = True
        except asyncio.TimeoutError:
            agent_joined = False
            _logger.warning(f"Agent did not join room={room} within 30 seconds")

        agent_present_now = await _check_agent_present(lk, room)
        dispatch_status, agent_participant_identity, dispatch_worker_id = await _get_dispatch_info(lk, dispatch, room) if dispatch else (None, None, None)
        return agent_joined, agent_present_now, dispatch_status, agent_participant_identity, dispatch_id, dispatch_worker_id
    finally:
        await lk.aclose()


async def _wait_for_agent_join(lk: api.LiveKitAPI, room: str) -> None:
    """Wait for agent to join room."""
    while True:
        lp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        participants = getattr(lp, "participants", None) or []
        if any(
            isinstance(getattr(p, "identity", None), str)
            and getattr(p, "identity", "").startswith("agent-")
            for p in participants
        ):
            return
        await asyncio.sleep(0.5)


async def _check_agent_present(lk: api.LiveKitAPI, room: str) -> bool:
    """Check if agent is currently present."""
    try:
        lp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        participants = getattr(lp, "participants", None) or []
        return any(
            isinstance(getattr(p, "identity", None), str)
            and getattr(p, "identity", "").startswith("agent-")
            for p in participants
        )
    except Exception:
        return False


async def _get_dispatch_info(lk: api.LiveKitAPI, dispatch: api.AgentDispatch, room: str) -> tuple[str | None, str | None, str | None]:
    """Get dispatch status, agent identity, and worker ID."""
    try:
        did = getattr(dispatch, "id", None)
        if not isinstance(did, str):
            return None, None, None
        
        await asyncio.wait_for(
            _wait_for_dispatch_state(lk, did, room),
            timeout=5.0
        )
        d = await lk.agent_dispatch.get_dispatch(dispatch_id=did, room_name=room)
        if not d:
            return None, None, None
        
        st = getattr(d, "state", None)
        jobs = getattr(st, "jobs", None) or []
        if not jobs:
            return None, None, None
        
        js = getattr(jobs[0], "state", None)
        status = getattr(js, "status", None)
        status_str = agent_proto.JobStatus.Name(status) if isinstance(status, int) else str(status) if status else None
        identity = getattr(js, "participant_identity", None)
        
        # Get worker ID from job state
        worker_id = getattr(js, "worker_id", None) or getattr(jobs[0], "worker_id", None)
        
        return status_str, identity, worker_id
    except Exception:
        return None, None, None


async def _wait_for_dispatch_state(lk: api.LiveKitAPI, dispatch_id: str, room: str) -> None:
    """Wait for dispatch to have state."""
    while True:
        d = await lk.agent_dispatch.get_dispatch(dispatch_id=dispatch_id, room_name=room)
        if d and getattr(getattr(d, "state", None), "jobs", None):
            return
        await asyncio.sleep(0.25)


async def handle_transcript(
    *,
    request: Request,
    interview_id: str,
    role: str,
    speaker_name: str | None,
    text: str,
    timestamp_ms: int | None,
    room: str | None,
    transcript_record_id: str | None,
) -> dict:
    if not TRANSCRIPTS_OBJECT_ID or not _can_write_transcript_json():
        return {"success": False, "reason": "transcript not configured"}

    record_id = transcript_record_id or _TRANSCRIPT_RECORD_BY_INTERVIEW_ID.get(interview_id)
    if not record_id:
        return {"success": False, "reason": "missing transcriptRecordId"}

    hrone_token = _token_for_transcript(request, interview_id=interview_id, record_id=record_id)
    timestamp = timestamp_ms or int(time.time() * 1000)
    
    # Get existing transcript and append new row
    res = await hrone_get_record(object_id=TRANSCRIPTS_OBJECT_ID, record_id=record_id, access_token=hrone_token)
    if res.status_code != 200:
        return
    current = res.json()
    transcript_json = extract_field_value(current, TRANSCRIPTS_FIELD_TRANSCRIPT_JSON) if current else []
    if not isinstance(transcript_json, list):
        transcript_json = []
    
    transcript_json.append([
        {"propertyId": TRANSCRIPTS_PROP_ID_ROLE, "key": "role", "value": role},
        {"propertyId": TRANSCRIPTS_PROP_ID_SPEAKER_NAME, "key": "speakerName", "value": speaker_name or ""},
        {"propertyId": TRANSCRIPTS_PROP_ID_TEXT, "key": "text", "value": text},
        {"propertyId": TRANSCRIPTS_PROP_ID_TIMESTAMP, "key": "timestamp", "value": timestamp},
    ])

    try:
        await hrone_update_record(
            object_id=TRANSCRIPTS_OBJECT_ID,
            record_id=record_id,
            values=[{
                "propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON,
                "key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON,
                "value": transcript_json
            }],
            access_token=hrone_token,
        )
        return {"success": True}
    except HTTPException as e:
        return {"success": False, "reason": str(e.detail)}


async def handle_feedback(
    *,
    request: Request,
    interview_id: str,
    experience: str,
    rating: int,
) -> dict:
    """Create a feedback record in HROne."""
    if not FEEDBACK_OBJECT_ID:
        return {"success": False, "reason": "feedback not configured"}
    
    if not isinstance(rating, int) or not (1 <= rating <= 10):
        return {"success": False, "reason": "rating must be between 1 and 10"}
    
    if not isinstance(experience, str) or not experience.strip():
        return {"success": False, "reason": "experience text is required"}

    hrone_token = _extract_access_token(request)
    values = [
        {"propertyId": FEEDBACK_PROP_ID_INTERVIEW_ID, "key": FEEDBACK_FIELD_INTERVIEW_ID, "value": interview_id}
        if FEEDBACK_PROP_ID_INTERVIEW_ID else {"key": FEEDBACK_FIELD_INTERVIEW_ID, "value": interview_id},
        {"propertyId": FEEDBACK_PROP_ID_EXPERIENCE, "key": FEEDBACK_FIELD_EXPERIENCE, "value": experience.strip()}
        if FEEDBACK_PROP_ID_EXPERIENCE else {"key": FEEDBACK_FIELD_EXPERIENCE, "value": experience.strip()},
        {"propertyId": FEEDBACK_PROP_ID_RATING, "key": FEEDBACK_FIELD_RATING, "value": rating}
        if FEEDBACK_PROP_ID_RATING else {"key": FEEDBACK_FIELD_RATING, "value": rating},
        {"propertyId": FEEDBACK_PROP_ID_TIMESTAMP, "key": FEEDBACK_FIELD_TIMESTAMP, "value": int(time.time() * 1000)}
        if FEEDBACK_PROP_ID_TIMESTAMP else {"key": FEEDBACK_FIELD_TIMESTAMP, "value": int(time.time() * 1000)},
    ]

    try:
        record_id = await hrone_create_record(object_id=FEEDBACK_OBJECT_ID, values=values, access_token=hrone_token)
        return {"success": True, "feedbackRecordId": record_id}
    except HTTPException as e:
        return {"success": False, "reason": str(e.detail)}