"""
AI Interview Bot API (minimal)
- Fetch interview data from HROne
- Create LiveKit room + dispatch agent
- Create/append InterviewTranscripts.transcriptJson via HROne propertyIds payload
"""

import io
import asyncio
import json
import os
import time
import uuid
from datetime import timedelta
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from livekit import api
from livekit.api.twirp_client import TwirpError
from livekit.protocol import room as room_proto

_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)

app = FastAPI(title="AI Interview Bot API")

# Allow browser clients to call this API (local dev / playground / frontend)
app.add_middleware(
    CORSMiddleware,
    # If allow_credentials=True, CORS cannot use wildcard "*" origins.
    # Add your frontend origins explicitly.
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

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

# Interviews mapping table (required for /api/start-interview)
INTERVIEWS_OBJECT_ID = _require_env("HRONE_INTERVIEWS_OBJECT_ID")
INTERVIEWS_VIEW_ID = _require_env("HRONE_INTERVIEWS_VIEW_ID")


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
_DISPATCHED_INTERVIEWS: set[str] = set()


async def _room_has_any_other_participant(lk: api.LiveKitAPI, room: str, *, exclude_identity: str) -> bool:
    """
    Minimal heuristic:
    - If the only participant is the candidate, there is probably no agent in the room.
    - If there's anyone else, assume agent is present.
    (We avoid relying on SDK-specific ParticipantInfo fields.)
    """
    try:
        resp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        participants = getattr(resp, "participants", None) or []
        for p in participants:
            ident = getattr(p, "identity", None)
            if isinstance(ident, str) and ident and ident != exclude_identity:
                return True
    except Exception:
        # If we can't list, be conservative and allow dispatch (better than a dead room).
        return False
    return False


# -----------------------------------------------------------------------------
# HROne helpers
# -----------------------------------------------------------------------------

def _extract_access_token(request: Request | None) -> str | None:
    if request is None:
        return None
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip() or None
    header_token = request.headers.get("x-hrone-access-token") or request.headers.get("X-HROne-Access-Token")
    if header_token:
        return header_token.strip() or None
    cookie_token = request.cookies.get("access_token")
    return cookie_token.strip() if cookie_token else None


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

def _get_field_any(record: object, key: str):
    v = extract_field_value(record, key)
    if v is not None:
        return v
    if isinstance(record, dict):
        if key in record:
            return record.get(key)
        recs = record.get("records")
        if isinstance(recs, dict) and key in recs:
            return recs.get(key)
    return None


def extract_skills(record) -> list:
    """Extract skills from nested requiredSkills structure."""
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
    """Extract custom questions from nested _questions structure."""
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
    """Extract skills from interviewer's interviewerSkills nested array."""
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
    """Extract personality traits from interviewer's personalityTraits nested field."""
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


async def fetch_interview_data(*, job_id: str, applicant_id: str, round_id: str, interviewer_id: str, access_token: str | None) -> dict | None:
    async with httpx.AsyncClient(timeout=60.0) as client:
        def _url(object_id: str, record_id: str) -> str:
            return f"{HRONE_API}/objects/{object_id}/records/{record_id}"

        job = await client.get(_url(JOBS_OBJECT_ID, job_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})
        applicant = await client.get(_url(APPLICANTS_OBJECT_ID, applicant_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})
        round_ = await client.get(_url(ROUNDS_OBJECT_ID, round_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})
        interviewer = await client.get(_url(INTERVIEWERS_OBJECT_ID, interviewer_id), headers=_hrone_headers(access_token), params={"appId": APP_ID})
        if any(r.status_code != 200 for r in (job, applicant, round_, interviewer)):
            return None

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
                "name": extract_field_value(applicant_data, "name"),
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


def _extract_record_id(payload: object) -> str | None:
    """Extract record ID from HROne API response. Create API returns {"message": "Record created"} with no ID."""
    if isinstance(payload, dict):
        # HROne create API doesn't return ID in response, only {"message": "Record created"}
        # Check if "id" key exists (for other endpoints that might return it)
        if "id" in payload:
            val = payload["id"]
            if isinstance(val, str) and val:
                return val
    return None
    


async def hrone_find_record_id_by_transcript_id(*, object_id: str, view_id: str, transcript_id: str, access_token: str | None) -> str | None:
    """
    Fallback for cases where create returns 200/201 but doesn't include the new record id.
    We query the view using transcriptId filter to find the record.
    """
    if not (object_id and view_id and transcript_id):
        return None
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Use filter format from working curl: filter by transcriptId
        res = await client.post(
            f"{HRONE_API}/objects/{object_id}/views/{view_id}/records",
            headers=_hrone_headers(access_token),
            params={"limit": 15, "offset": 0},
            json={
                "filters": {
                    "$and": [
                        {
                            "key": "#.records.transcriptId",
                            "operator": "$eq",
                            "value": transcript_id,
                            "type": "singleLineText"
                        }
                    ]
                },
                "appId": APP_ID
            },
        )
        if res.status_code != 200:
            return None
        try:
            payload = res.json()
        except Exception:
            return None
        # View API returns records in "data" array with "id" field
        records = payload.get("data", [])
        if records and isinstance(records[0], dict):
            record_id = records[0].get("id")
            if isinstance(record_id, str) and record_id:
                return record_id
    return None


async def hrone_create_record(*, object_id: str, values: list[dict], access_token: str | None) -> str | None:
    """Create HROne record. Note: Create API returns {"message": "Record created"} with no ID - use fallback lookup."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.post(
            f"{HRONE_API}/objects/{object_id}/records",
            headers=_hrone_headers(access_token),
            params={"appId": APP_ID},
            json=_values_payload(values),
        )
        if res.status_code not in (200, 201):
            raise HTTPException(res.status_code, f"HROne create failed: {res.text}")
        # HROne create API doesn't return record ID, only {"message": "Record created"}
        # Return None - caller should use fallback lookup by transcriptId
        return None


async def hrone_get_record(*, object_id: str, record_id: str, access_token: str | None) -> dict | list | None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.get(
            f"{HRONE_API}/objects/{object_id}/records/{record_id}",
            headers=_hrone_headers(access_token),
            params={"appId": APP_ID},
        )
        if res.status_code != 200:
            return None
        return res.json()


async def hrone_update_record(*, object_id: str, record_id: str, values: list[dict], access_token: str | None) -> None:
    print(f"{HRONE_API}/objects/{object_id}/records/{record_id}")
    async with httpx.AsyncClient(timeout=15.0) as client:
        res = await client.patch(
            f"{HRONE_API}/objects/{object_id}/records/{record_id}",
            headers=_hrone_headers(access_token),
            params={"appId": APP_ID},
            json=_values_payload(values),
        )
        if res.status_code not in (200, 204):
            raise HTTPException(res.status_code, f"HROne update failed: {res.text}")

def _lower(v) -> str:
    if isinstance(v, dict):
        v = v.get("label") or v.get("id")
    return (str(v or "")).strip().lower()


async def _resolve_ids_from_interview(*, interview_id: str, email: str, access_token: str | None) -> tuple[str, str, str, str]:
    # HROne can return a generic 500 if filter "type" doesn't match field type.
    # Keep this tiny but robust by retrying with two common types.
    res = None
    for t in ("singleLineText", "number"):
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(
                f"{HRONE_API}/objects/{INTERVIEWS_OBJECT_ID}/views/{INTERVIEWS_VIEW_ID}/records",
                headers=_hrone_headers(access_token),
                params={"limit": 1, "offset": 0},
                json={
                    "filters": {
                        "$and": [
                            {
                                # In HROne views this is often stored under a custom key like "_interviewId"
                                "key": "#.records._interviewId",
                                "operator": "$eq",
                                "value": str(interview_id),
                                "type": t,
                            }
                        ]
                    },
                    "appId": APP_ID,
                },
            )
        if res.status_code == 200:
            break
    if res is None or res.status_code != 200:
        raise HTTPException(getattr(res, "status_code", 500), f"Interview lookup failed: {getattr(res, 'text', '')}")

    rec = (res.json().get("data") or [None])[0]
    if not isinstance(rec, dict):
        raise HTTPException(404, f"Interview not found for interviewId={interview_id}")

    def _id(v):
        return v.get("id") if isinstance(v, dict) else v

    applicant_id = _id(_get_field_any(rec, "applicantId"))
    job_id = _id(_get_field_any(rec, "jobId"))
    round_id = _id(_get_field_any(rec, "roundId"))
    interviewer_id = _id(_get_field_any(rec, "interviewerId"))
    mapped_email = _get_field_any(rec, "applicantEmail")

    if not (applicant_id and job_id and round_id and interviewer_id):
        raise HTTPException(500, "Interview mapping missing applicantId/jobId/roundId/interviewerId")

    if not mapped_email:
        raise HTTPException(500, "Interview mapping missing applicantEmail")
    if _lower(mapped_email) != _lower(email):
        expected = mapped_email.get("label") if isinstance(mapped_email, dict) else str(mapped_email)
        raise HTTPException(403, f"Email mismatch for this interviewId (expected: {expected})")

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
# Request models
# -----------------------------------------------------------------------------

class InterviewDataRequest(BaseModel):
    applicantId: str
    jobId: str
    roundId: str
    interviewerId: str


class StartInterviewRequest(BaseModel):
    interviewId: str
    email: str


class TranscriptRequest(BaseModel):
    interviewId: str
    role: str
    speakerName: str | None = None
    text: str
    timestampMs: int | None = None
    room: str | None = None
    transcriptRecordId: str | None = None


# -----------------------------------------------------------------------------
# APIs
# -----------------------------------------------------------------------------

@app.post("/api/interview-data")
async def api_interview_data(req: InterviewDataRequest, request: Request):
    token = _extract_access_token(request)
    data = await fetch_interview_data(
        job_id=req.jobId,
        applicant_id=req.applicantId,
        round_id=req.roundId,
        interviewer_id=req.interviewerId,
        access_token=token,
    )
    if not data:
        raise HTTPException(404, "Failed to fetch interview data")
    return data


@app.post("/api/start-interview")
async def api_start_interview(req: StartInterviewRequest, request: Request):
    if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise HTTPException(500, "LiveKit credentials not configured")
    if not req.interviewId:
        raise HTTPException(400, "interviewId is required")

    hrone_token = _extract_access_token(request)
    if hrone_token:
        _HRONE_TOKEN_BY_INTERVIEW_ID[req.interviewId] = hrone_token

    applicant_id, job_id, round_id, interviewer_id = await _resolve_ids_from_interview(
        interview_id=req.interviewId, email=req.email, access_token=hrone_token
    )

    interview_data = await fetch_interview_data(
        job_id=job_id,
        applicant_id=applicant_id,
        round_id=round_id,
        interviewer_id=interviewer_id,
        access_token=hrone_token,
    )
    participant_name = (interview_data.get("applicant") or {}).get("name") or req.email or "Candidate"

    if not (isinstance(interview_data, dict) and interview_data.get("job") and interview_data.get("applicant")):
        raise HTTPException(400, "Missing required interview data (job/applicant)")

    interview_data["interviewId"] = req.interviewId
    if PUBLIC_API_URL:
        interview_data["transcriptApiUrl"] = f"{PUBLIC_API_URL.rstrip('/')}/api/transcript"

    # Create transcript record only once per interviewId (per API process)
    transcript_record_id = _TRANSCRIPT_RECORD_BY_INTERVIEW_ID.get(req.interviewId)
    if TRANSCRIPTS_OBJECT_ID and not transcript_record_id:
        transcript_id = uuid.uuid4().hex
        values: list[dict] = []
        if TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID and TRANSCRIPTS_FIELD_TRANSCRIPT_ID:
            values.append({"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_ID, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_ID, "value": transcript_id})
        else:
            values.append({"key": TRANSCRIPTS_FIELD_TRANSCRIPT_ID, "value": transcript_id})

        if TRANSCRIPTS_PROP_ID_INTERVIEW_ID:
            values.append({"propertyId": TRANSCRIPTS_PROP_ID_INTERVIEW_ID, "key": TRANSCRIPTS_FIELD_INTERVIEW_ID, "value": req.interviewId})
        else:
            values.append({"key": TRANSCRIPTS_FIELD_INTERVIEW_ID, "value": req.interviewId})

        if TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON:
            values.append({"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": []})
        else:
            values.append({"key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": []})

        await hrone_create_record(object_id=TRANSCRIPTS_OBJECT_ID, values=values, access_token=hrone_token)
        # HROne create API doesn't return ID, so lookup by transcriptId
        transcript_record_id = None
        if TRANSCRIPTS_VIEW_ID:
            transcript_record_id = await hrone_find_record_id_by_transcript_id(
                object_id=TRANSCRIPTS_OBJECT_ID,
                view_id=TRANSCRIPTS_VIEW_ID,
                transcript_id=transcript_id,
                access_token=hrone_token,
            )
        if transcript_record_id:
            print(f"✅ InterviewTranscripts record created: {transcript_record_id}")
        else:
            print("⚠️ Record created but ID not found (will retry on /api/transcript)")

    if transcript_record_id:
        _TRANSCRIPT_RECORD_BY_INTERVIEW_ID[req.interviewId] = transcript_record_id
        interview_data["transcriptRecordId"] = transcript_record_id
        if hrone_token:
            _HRONE_TOKEN_BY_RECORD_ID[transcript_record_id] = hrone_token

    # LiveKit room + dispatch
    room = f"interview-{req.interviewId}"
    applicant_id_for_identity = applicant_id or req.interviewId
    # IMPORTANT: identity must be unique per join, otherwise the browser can fail to connect
    # if a stale session with the same identity still exists.
    identity = f"candidate-{applicant_id_for_identity}-{uuid.uuid4().hex[:8]}"

    token = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(participant_name or "Participant")
        # IMPORTANT: the browser (Agents Playground) needs subscribe/publish/data permissions,
        # otherwise you won't see/hear anything.
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_subscribe=True,
                can_publish=True,
                can_publish_data=True,
            )
        )
        .with_ttl(timedelta(hours=1))
    )

    lk = api.LiveKitAPI(LIVEKIT_URL)
    try:
        md_json = json.dumps(interview_data)
        # Ensure metadata is set even for long-lived rooms.
        # If room doesn't exist, create it, then update metadata.
        try:
            await lk.room.update_room_metadata(api.UpdateRoomMetadataRequest(room=room, metadata=md_json))
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
            await lk.room.update_room_metadata(api.UpdateRoomMetadataRequest(room=room, metadata=md_json))

        # --- Agent dispatch idempotency ---
        # Goal: only one agent in the room and avoid stacking dispatch jobs.
        dispatch = None

        # 0) Cleanup dispatch list so we don't accumulate old/invalid rows.
        # Keep at most one dispatch for (room, LIVEKIT_AGENT_NAME); delete everything else.
        try:
            existing = await lk.agent_dispatch.list_dispatch(room_name=room)
            kept = None
            for d in existing or []:
                did = getattr(d, "id", None)
                d_room = getattr(d, "room", None)
                d_agent = getattr(d, "agent_name", None)
                is_valid = (d_room == room) and (d_agent == LIVEKIT_AGENT_NAME)

                if not isinstance(did, str) or not did:
                    # Can't delete without an id; just ignore.
                    continue

                if is_valid and kept is None:
                    kept = d
                    continue

                # Delete invalid dispatches and extra valid ones.
                await lk.agent_dispatch.delete_dispatch(dispatch_id=did, room_name=room)

            if kept is not None:
                dispatch = kept
        except TwirpError as e:
            if getattr(e, "code", None) != "not_found":
                raise
        except Exception:
            # Best-effort only; a failure here should not block interview start.
            pass

        # 1) If multiple agents are already in the room, kick extras (keep the first).
        try:
            lp0 = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
            participants0 = getattr(lp0, "participants", None) or []
            agent_identities = [
                str(getattr(p, "identity", "") or "")
                for p in participants0
                if isinstance(getattr(p, "identity", None), str) and getattr(p, "identity", "").startswith("agent-")
            ]
            if len(agent_identities) > 1:
                # Keep the oldest/first one and remove the rest.
                for extra in agent_identities[1:]:
                    await lk.room.remove_participant(room_proto.RoomParticipantIdentity(room=room, identity=extra))
        except Exception:
            # Best-effort cleanup only.
            pass

        # 2) If an agent is already present, don't dispatch again. Also delete any queued dispatches.
        agent_already_present = False
        try:
            lp1 = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
            participants1 = getattr(lp1, "participants", None) or []
            agent_already_present = any(
                isinstance(getattr(p, "identity", None), str)
                and getattr(p, "identity")
                and not str(getattr(p, "identity")).startswith("candidate-")
                for p in participants1
            )
        except Exception:
            agent_already_present = False

        if agent_already_present:
            try:
                existing = await lk.agent_dispatch.list_dispatch(room_name=room)
                for d in existing or []:
                    did = getattr(d, "id", None)
                    if isinstance(did, str) and did:
                        await lk.agent_dispatch.delete_dispatch(dispatch_id=did, room_name=room)
            except Exception:
                pass
        else:
            # 3) If a matching dispatch already exists, reuse it (don't create more).
            # (We may have already set dispatch via cleanup step above.)
            if dispatch is None:
                dispatch = await lk.agent_dispatch.create_dispatch(
                    api.CreateAgentDispatchRequest(room=room, agent_name=LIVEKIT_AGENT_NAME)
                )
                _DISPATCHED_INTERVIEWS.add(req.interviewId)

        # Best-effort: wait briefly for agent to join so the client can know if dispatch worked.
        agent_joined = False
        for _ in range(30):
            try:
                lp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
                participants = getattr(lp, "participants", None) or []
                # Treat any non-candidate participant as the agent (avoids false positives from other candidates).
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
    finally:
        await lk.aclose()

    return {
        "success": True,
        "token": token.to_jwt(),
        "room": room,
        "livekitUrl": LIVEKIT_URL,
        "identity": identity,
        "transcriptRecordId": transcript_record_id,
        "agentJoined": agent_joined,
        "dispatchId": getattr(dispatch, "id", None),
    }


@app.post("/api/transcript")
async def api_transcript(req: TranscriptRequest, request: Request):
    print(f"📥 /api/transcript hit: interviewId={req.interviewId} transcriptRecordId={req.transcriptRecordId} role={req.role}")
    if not TRANSCRIPTS_OBJECT_ID:
        raise HTTPException(500, "HRONE_TRANSCRIPTS_OBJECT_ID not configured")
    if not _can_write_transcript_json():
        return {"success": False, "reason": "missing transcriptJson propertyIds"}

    record_id = req.transcriptRecordId or _TRANSCRIPT_RECORD_BY_INTERVIEW_ID.get(req.interviewId)
    if not record_id:
        return {"success": False, "reason": "missing transcriptRecordId"}

    hrone_token = _token_for_transcript(request, interview_id=req.interviewId, record_id=record_id)

    ts = req.timestampMs if req.timestampMs is not None else int(time.time() * 1000)
    row = [
        {"propertyId": TRANSCRIPTS_PROP_ID_ROLE, "key": "role", "value": req.role},
        {"propertyId": TRANSCRIPTS_PROP_ID_SPEAKER_NAME, "key": "speakerName", "value": req.speakerName or ""},
        {"propertyId": TRANSCRIPTS_PROP_ID_TEXT, "key": "text", "value": req.text},
        {"propertyId": TRANSCRIPTS_PROP_ID_TIMESTAMP, "key": "timestamp", "value": ts},
    ]

    current = await hrone_get_record(object_id=TRANSCRIPTS_OBJECT_ID, record_id=record_id, access_token=hrone_token)
    existing = extract_field_value(current, TRANSCRIPTS_FIELD_TRANSCRIPT_JSON) if current is not None else None
    transcript_json = existing if isinstance(existing, list) else []
    transcript_json.append(row)

    await hrone_update_record(
        object_id=TRANSCRIPTS_OBJECT_ID,
        record_id=record_id,
        values=[{"propertyId": TRANSCRIPTS_PROP_ID_TRANSCRIPT_JSON, "key": TRANSCRIPTS_FIELD_TRANSCRIPT_JSON, "value": transcript_json}],
        access_token=hrone_token,
    )
    return {"success": True}




@app.get("/api/token")
async def api_token(room: str, identity: str, name: str = "Participant"):
    if not (LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
        raise HTTPException(500, "LiveKit credentials not configured")
    t = (
        api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
                can_subscribe=True,
                can_publish=True,
                can_publish_data=True,
            )
        )
        .with_ttl(timedelta(hours=1))
    )
    return {"token": t.to_jwt(), "room": room, "livekitUrl": LIVEKIT_URL}


@app.get("/api/debug/room/{interview_id}")
async def api_debug_room(interview_id: str):
    room = f"interview-{interview_id}"
    lk = api.LiveKitAPI(LIVEKIT_URL)
    try:
        try:
            rp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
        except TwirpError as e:
            if getattr(e, "code", None) == "not_found":
                return {"room": room, "exists": False, "participant_count": 0, "participants": [], "metadata_present": False}
            raise

        participants = getattr(rp, "participants", None) or []
        rr = await lk.room.list_rooms(api.ListRoomsRequest(names=[room]))
        rooms = getattr(rr, "rooms", None) or []
        md = rooms[0].metadata if rooms else None
        return {
            "room": room,
            "exists": True,
            "participant_count": len(participants),
            "participants": [{"identity": getattr(p, "identity", None), "name": getattr(p, "name", None)} for p in participants],
            "metadata_present": bool(md),
        }
    finally:
        await lk.aclose()


@app.get("/api/debug/dispatches/{interview_id}")
async def api_debug_dispatches(interview_id: str):
    """
    Debug helper: shows agent dispatch jobs created for this room.
    If dispatches exist but agent isn't in participants, the worker likely crashed or couldn't connect.
    """
    room = f"interview-{interview_id}"
    lk = api.LiveKitAPI(LIVEKIT_URL)
    try:
        try:
            dispatches = await lk.agent_dispatch.list_dispatch(room_name=room)
        except TwirpError as e:
            # LiveKit returns not_found if the room hasn't been created yet.
            if getattr(e, "code", None) == "not_found":
                return {"room": room, "exists": False, "dispatch_count": 0, "dispatches": []}
            raise

        def _jsonable(v):
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            # protobuf enums / timestamps / other objects
            try:
                return int(v)
            except Exception:
                return str(v)

        return {
            "room": room,
            "exists": True,
            "dispatch_count": len(dispatches or []),
            "dispatches": [
                {
                    "id": getattr(d, "id", None),
                    "agent_name": getattr(d, "agent_name", None),
                    "room": getattr(d, "room", None),
                    "metadata": getattr(d, "metadata", None),
                    "state": _jsonable(getattr(d, "state", None)),
                    "created_at": _jsonable(getattr(d, "created_at", None)),
                }
                for d in (dispatches or [])
            ],
        }
    finally:
        await lk.aclose()


@app.post("/api/debug/reset/{interview_id}")
async def api_debug_reset(interview_id: str):
    """
    Hard reset a room for debugging:
    - delete all dispatches for the room
    - remove all participants (agent + candidates)
    - delete the room
    This helps when stale workers/dispatches make behavior look random.
    """
    room = f"interview-{interview_id}"
    lk = api.LiveKitAPI(LIVEKIT_URL)
    kicked: list[str] = []
    deleted_dispatches: list[str] = []
    try:
        # Delete dispatches (ignore not_found)
        try:
            existing = await lk.agent_dispatch.list_dispatch(room_name=room)
            for d in existing or []:
                did = getattr(d, "id", None)
                if isinstance(did, str) and did:
                    await lk.agent_dispatch.delete_dispatch(dispatch_id=did, room_name=room)
                    deleted_dispatches.append(did)
        except TwirpError as e:
            if getattr(e, "code", None) != "not_found":
                raise

        # Kick participants (ignore not_found)
        try:
            rp = await lk.room.list_participants(api.ListParticipantsRequest(room=room))
            participants = getattr(rp, "participants", None) or []
            for p in participants:
                ident = getattr(p, "identity", None)
                if isinstance(ident, str) and ident:
                    await lk.room.remove_participant(room_proto.RoomParticipantIdentity(room=room, identity=ident))
                    kicked.append(ident)
        except TwirpError as e:
            if getattr(e, "code", None) != "not_found":
                raise

        # Delete room (ignore not_found)
        try:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room))
        except TwirpError as e:
            if getattr(e, "code", None) != "not_found":
                raise

        return {"room": room, "deleted_dispatches": deleted_dispatches, "kicked": kicked, "deleted_room": True}
    finally:
        await lk.aclose()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8021"))
    # For auto-reload, uvicorn needs an import string (recommended: run via CLI).
    # This works too when running `python main.py`.
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

