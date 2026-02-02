"""
Shared helpers for `main.py`.

Goal: keep `main.py` focused on request models + API routes only.
All heavy lifting (HROne, LiveKit, transcript persistence, dispatch) lives here.
"""

import asyncio
import io
import json
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
from livekit.protocol import room as room_proto


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
                print(f"⚠️ LiveKit metadata trimmed to fit limit (dropped {label})")
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
    print("⚠️ LiveKit metadata trimmed to essentials to fit limit")
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
    with suppress(Exception):
        lp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        participants = getattr(lp, "participants", None) or []
        agent_present = any(
            isinstance(getattr(p, "identity", None), str)
            and getattr(p, "identity")
            and not str(getattr(p, "identity")).startswith("candidate-")
            for p in participants
        )

    existing_valid = None
    existing_status = None
    try:
        dispatches = await lk.agent_dispatch.list_dispatch(room_name=room)
        for d in dispatches or []:
            if getattr(d, "room", None) == room and getattr(d, "agent_name", None) == LIVEKIT_AGENT_NAME:
                existing_valid = d
                with suppress(Exception):
                    st = getattr(d, "state", None)
                    jobs = getattr(st, "jobs", None) or []
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
        return existing_valid

    return await lk.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(room=room, agent_name=LIVEKIT_AGENT_NAME)
    )


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
        url = "https://api.hrone.studio/api/storage-accounts/lego/download"
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


async def hrone_find_record_id_by_transcript_id(*, object_id: str, view_id: str, transcript_id: str, access_token: str | None) -> str | None:
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
        if res.status_code != 200:
            print(f"⚠️ HROne view records failed: POST {url} -> {res.status_code} {_short(res.text)}")
            return None
        try:
            payload = res.json()
        except Exception:
            print(f"⚠️ HROne view records invalid JSON: POST {url} -> {_short(res.text)}")
            return None
        records = payload.get("data", [])
        if records and isinstance(records[0], dict):
            record_id = records[0].get("id")
            if isinstance(record_id, str) and record_id:
                return record_id
    return None


async def hrone_create_record(*, object_id: str, values: list[dict], access_token: str | None) -> None:
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


async def hrone_get_record(*, object_id: str, record_id: str, access_token: str | None) -> dict | list | None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        url = f"{HRONE_API}/objects/{object_id}/records/{record_id}"
        res = await client.get(
            url,
            headers=_hrone_headers(access_token),
            params={"appId": APP_ID},
        )
        if res.status_code != 200:
            print(f"⚠️ HROne get failed: GET {url} -> {res.status_code} {_short(res.text)}")
            return None
        return res.json()


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
    interview_rec = await hrone_get_record(object_id=INTERVIEWS_OBJECT_ID, record_id=str(interview_id), access_token=access_token)
    if interview_rec is None:
        raise HTTPException(404, f"Interview record not found: {interview_id}")

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

    applicant_rec = await hrone_get_record(object_id=APPLICANTS_OBJECT_ID, record_id=str(applicant_id), access_token=access_token)
    if applicant_rec is None:
        raise HTTPException(502, "Missing data from HROne: applicant")
    expected_email = extract_field_value(applicant_rec, "email")
    if not expected_email:
        raise HTTPException(500, "Applicant record missing email")
    if _lower(expected_email) != _lower(email):
        raise HTTPException(403, f"Email mismatch for this interviewId (expected: {expected_email})")

    return str(applicant_id), str(job_id), str(round_id), str(interviewer_id)


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

    # Create transcript record only once per interviewId (per API process)
    transcript_record_id = _TRANSCRIPT_RECORD_BY_INTERVIEW_ID.get(interview_id)
    if TRANSCRIPTS_OBJECT_ID and not transcript_record_id:
        transcript_id = uuid.uuid4().hex
        values: list[dict] = []
        if TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID and TRANSCRIPTS_FIELD_TRANSCRIPT_ID:
            values.append({"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_ID, "value": transcript_id})
        else:
            values.append({"key": TRANSCRIPTS_FIELD_TRANSCRIPT_ID, "value": transcript_id})

        if TRANSCRIPTS_PROP_ID_INTERVIEW_ID:
            values.append({"propertyId": TRANSCRIPTS_PROP_ID_INTERVIEW_ID, "key": TRANSCRIPTS_FIELD_INTERVIEW_ID, "value": interview_id})
        else:
            values.append({"key": TRANSCRIPTS_FIELD_INTERVIEW_ID, "value": interview_id})

        if TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON:
            values.append({"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": []})
        else:
            values.append({"key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": []})

        await hrone_create_record(object_id=TRANSCRIPTS_OBJECT_ID, values=values, access_token=hrone_token)
        transcript_record_id = None
        if TRANSCRIPTS_VIEW_ID:
            transcript_record_id = await hrone_find_record_id_by_transcript_id(
                object_id=TRANSCRIPTS_OBJECT_ID,
                view_id=TRANSCRIPTS_VIEW_ID,
                transcript_id=transcript_id,
                access_token=hrone_token,
            )

    if transcript_record_id:
        _TRANSCRIPT_RECORD_BY_INTERVIEW_ID[interview_id] = transcript_record_id
        interview_data["transcriptRecordId"] = transcript_record_id
        if hrone_token:
            _HRONE_TOKEN_BY_RECORD_ID[transcript_record_id] = hrone_token

    # LiveKit room + dispatch
    room = f"interview-{interview_id}"
    identity = f"candidate-{applicant_id}-{uuid.uuid4().hex[:8]}"

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(participant_name or "Participant")
        .with_grants(
            api.VideoGrants(room_join=True, room=room, can_subscribe=True, can_publish=True, can_publish_data=True)
        )
        .with_ttl(timedelta(hours=1))
    )

    dispatch = None
    agent_joined = False
    agent_present_now = False
    dispatch_state_str = ""
    assigned_worker_id = None
    agent_participant_identity = None
    dispatch_status = None

    lk = api.LiveKitAPI(LIVEKIT_URL)
    try:
        # Remove stale legacy rooms like interview-1 that can steal worker assignments.
        await _cleanup_legacy_numeric_rooms(lk, keep_room=room)

        md_json = _room_metadata_json(interview_data)
        await ensure_room_with_metadata(lk, room=room, md_json=md_json)
        dispatch = await ensure_agent_dispatched(lk, room=room)

        for _ in range(30):
            try:
                lp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
                participants = getattr(lp, "participants", None) or []
                agent_joined = any(
                    isinstance(getattr(p, "identity", None), str)
                    and getattr(p, "identity")
                    and not getattr(p, "identity").startswith("candidate-")
                    for p in participants
                )
                if agent_joined:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)

        # Current presence snapshot (more reliable than agentJoined after the fact)
        try:
            lp2 = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
            participants2 = getattr(lp2, "participants", None) or []
            agent_present_now = any(
                isinstance(getattr(p, "identity", None), str)
                and getattr(p, "identity")
                and not getattr(p, "identity").startswith("candidate-")
                for p in participants2
            )
        except Exception:
            agent_present_now = agent_joined

        # Helpful debug fields: which worker LiveKit assigned this dispatch to (if any).
        # Note: CreateDispatch can return before the state is populated. Poll briefly using get_dispatch().
        dispatch_state_str = str(getattr(dispatch, "state", "") or "")
        try:
            did = getattr(dispatch, "id", None)
            if isinstance(did, str) and did:
                for _ in range(20):
                    d2 = await lk.agent_dispatch.get_dispatch(dispatch_id=did, room_name=room)
                    if d2 is None:
                        await asyncio.sleep(0.25)
                        continue
                    dispatch_state_str = str(getattr(d2, "state", "") or "")
                    if dispatch_state_str:
                        break
                    await asyncio.sleep(0.25)
        except Exception:
            pass

        if dispatch_state_str:
            m = re.search(r'worker_id:\s*"([^"]+)"', dispatch_state_str)
            assigned_worker_id = m.group(1) if m else None
            m = re.search(r'participant_identity:\s*"([^"]+)"', dispatch_state_str)
            agent_participant_identity = m.group(1) if m else None
            m = re.search(r'status:\s*(JS_[A-Z_]+)', dispatch_state_str)
            dispatch_status = m.group(1) if m else None
    finally:
        await lk.aclose()

    return {
        "success": True,
        "token": token.to_jwt(),
        "room": room,
        "livekitUrl": LIVEKIT_URL,
        "livekitAgentName": LIVEKIT_AGENT_NAME,
        "identity": identity,
        "transcriptRecordId": transcript_record_id,
        # HROne auth debug (explains why transcript may 401 later)
        "hroneTokenPresent": bool(hrone_token),
        "hroneTokenSource": hrone_token_source,
        # useful ids (debug)
        "interview_id": interview_id,
        "applicant_id": applicant_id,
        "job_id": job_id,
        "round_id": round_id,
        "interviewer_id": interviewer_id,
        # dispatch/agent debug
        "agentJoined": agent_joined,
        "dispatchId": getattr(dispatch, "id", None) if dispatch is not None else None,
        "dispatchWorkerId": assigned_worker_id,
        "agentParticipantIdentity": agent_participant_identity,
        "dispatchStatus": dispatch_status,
        "agentPresentNow": agent_present_now,
        "resumed": True,
        "interview_data": interview_data,
    }


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
    if not TRANSCRIPTS_OBJECT_ID:
        raise HTTPException(500, "HRONE_TRANSCRIPTS_OBJECT_ID not configured")
    if not _can_write_transcript_json():
        return {"success": False, "reason": "missing transcriptJson propertyIds"}

    record_id = transcript_record_id or _TRANSCRIPT_RECORD_BY_INTERVIEW_ID.get(interview_id)
    if not record_id:
        return {"success": False, "reason": "missing transcriptRecordId"}

    hrone_token = _token_for_transcript(request, interview_id=interview_id, record_id=record_id)

    ts = timestamp_ms if timestamp_ms is not None else int(time.time() * 1000)
    row = [
        {"propertyId": TRANSCRIPTS_PROP_ID_ROLE, "key": "role", "value": role},
        {"propertyId": TRANSCRIPTS_PROP_ID_SPEAKER_NAME, "key": "speakerName", "value": speaker_name or ""},
        {"propertyId": TRANSCRIPTS_PROP_ID_TEXT, "key": "text", "value": text},
        {"propertyId": TRANSCRIPTS_PROP_ID_TIMESTAMP, "key": "timestamp", "value": ts},
    ]

    current = await hrone_get_record(object_id=TRANSCRIPTS_OBJECT_ID, record_id=record_id, access_token=hrone_token)
    existing = extract_field_value(current, TRANSCRIPTS_FIELD_TRANSCRIPT_JSON) if current is not None else None
    transcript_json = existing if isinstance(existing, list) else []
    transcript_json.append(row)

    try:
        await hrone_update_record(
            object_id=TRANSCRIPTS_OBJECT_ID,
            record_id=record_id,
            values=[{"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": transcript_json}],
            access_token=hrone_token,
        )
        return {"success": True}
    except HTTPException as e:
        print(f"⚠️ transcript update failed: {e.detail}")
        return {"success": False, "reason": str(e.detail)}