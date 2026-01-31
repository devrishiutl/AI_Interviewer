from dotenv import load_dotenv
try:
    load_dotenv()
except Exception:
    # Don't fail import if .env isn't readable in the current environment.
    pass

import os
import re
import time

import httpx

# ============================================================================
# BUILD PROMPTS FROM DISPATCHED DATA
# ============================================================================

def require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def download_turn_detector_files() -> None:
    """Download turn-detector assets without requiring a LiveKit job context."""
    from livekit.plugins.turn_detector.english import _EUORunnerEn
    from livekit.plugins.turn_detector.multilingual import _EUORunnerMultilingual

    print("⬇️ Downloading LiveKit turn-detector model files...")
    _EUORunnerEn._download_files()
    _EUORunnerMultilingual._download_files()
    print("✅ Done.")


def maybe_turn_detection():
    """Create turn-detection model if available; otherwise return None."""
    try:
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        return MultilingualModel()
    except Exception as e:
        print(f"⚠️ turn_detection disabled: {type(e).__name__}: {e}")
        return None


def tts_speaker(md: dict) -> str:
    voice = (md.get("interviewer") or {}).get("voice") or "Female"
    return "abhilash" if voice == "Male" else "anushka"


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def dedupe_text(last: tuple[str, int], text: str, *, window_ms: int = 10_000) -> tuple[bool, tuple[str, int]]:
    now = int(time.time() * 1000)
    norm = _norm_text(text)
    last_text, last_ts = last
    if norm and last_text and (now - last_ts) < window_ms and (norm == last_text or norm in last_text or last_text in norm):
        return True, last
    return False, (norm, now)


async def post_transcript(ctx: dict, *, role: str, text: str) -> None:
    url = (
        os.getenv("TRANSCRIPT_API_URL")
        or ctx.get("transcriptApiUrl")
        or "http://127.0.0.1:8021/api/transcript"
    )
    interview_id = ctx.get("interviewId")
    if not interview_id:
        return

    speaker_name = ctx.get("candidateName") if role == "Candidate" else ctx.get("interviewerName")
    payload = {
        "room": ctx.get("room") or "",
        "interviewId": str(interview_id),
        "transcriptRecordId": ctx.get("transcriptRecordId"),
        "role": role,
        "speakerName": speaker_name or "",
        "text": text,
        "timestampMs": int(time.time() * 1000),
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(url, json=payload)
            if r.status_code >= 300:
                print(f"⚠️ /api/transcript failed {r.status_code}: {r.text}")
    except Exception as e:
        print(f"⚠️ /api/transcript error: {type(e).__name__}: {e}")
        return

def build_instructions(data: dict) -> str:
    """Build agent instructions from interview data (data is required)"""
    
    job = data.get("job", {})
    applicant = data.get("applicant", {})
    round_info = data.get("round", {}) or {}
    interviewer = data.get("interviewer", {}) or {}
    
    # Job info
    job_title = job.get("title", "the position")
    job_skills = job.get("skills", "")
    experience = job.get("experience", "")

    # Applicant info
    name = applicant.get("name", "Candidate")
    resume = applicant.get("resumeText", "")
    
    # Round info
    round_name = round_info.get("name", "Interview")
    round_type = round_info.get("type", "General")
    round_duration = round_info.get("duration", "30 minutes")
    round_objective = round_info.get("objective", "Evaluate candidate")
    custom_questions = round_info.get("questions", [])
    
    # Interviewer info
    interviewer_name = interviewer.get("name", "AI Interviewer")
    interviewer_desc = interviewer.get("description", "")
    interviewer_skills = interviewer.get("skills", [])
    personality = interviewer.get("personalityTraits", {})
    
    # Personality-based style
    empathy = personality.get("empathy", 5)
    rapport = personality.get("rapport", 5)
    
    # Truncate long resume
    if resume and len(resume) > 2500:
        resume = resume[:2500] + "..."
    
    interviewer_skills_text = ", ".join(interviewer_skills) if interviewer_skills else "General"
    
    # Format priority questions
    priority_questions_section = ""
    if custom_questions:
        # HROne sometimes returns nested arrays/objects for questions; accept only strings here.
        valid_questions = []
        for q in custom_questions:
            if isinstance(q, str) and q.strip():
                valid_questions.append(q.strip())
        if valid_questions:
            priority_questions_section = f"""
PRIORITY QUESTIONS (Ask these FIRST, one by one):
{chr(10).join(f'• {q}' for q in valid_questions)}

After asking these priority questions, you may ask follow-up questions based on their resume.
"""
    
    return f"""You are {interviewer_name}, a professional interviewer with expertise in {interviewer_skills_text}. {interviewer_desc}

INTERVIEW DETAILS:
- Round: {round_name} ({round_type})
- Duration: {round_duration}
- Position: {job_title}

CANDIDATE: {name}
JOB: {job_title} ({experience} experience required)
SKILLS TO EVALUATE: {job_skills}
OBJECTIVE: {round_objective}
{priority_questions_section}
CANDIDATE'S RESUME:
{resume if resume else "No resume provided"}

HOW TO CONDUCT THIS INTERVIEW:
1. After greeting, START with the priority questions listed above (if any)
2. Ask ONE question at a time, wait for response
3. Ask brief follow-ups to dig deeper on their answers
4. Then ask questions about their resume projects (LangGraph, Roberta NER, Rasa, etc.)
5. Evaluate their {job_skills} skills
6. Address them as {name}
7. Keep the interview within {round_duration}
8. Be {"warm and encouraging" if empathy >= 7 else "professional but friendly"}
9. {"Build rapport with friendly conversation before diving into technical questions" if rapport >= 7 else "Stay focused on evaluating skills"}
10. NEVER announce question numbers or say "next question" - just ask naturally
11. If they say "quit", "exit", "bye" - thank them and end gracefully

Keep it conversational. No scripted announcements."""


def build_welcome(data: dict) -> str:
    """Build personalized welcome message (data is required)"""
    name = data.get("applicant", {}).get("name", "there")
    job_title = data.get("job", {}).get("title", "this position")
    round_info = data.get("round", {}) or {}
    interviewer = data.get("interviewer", {}) or {}
    
    round_name = round_info.get("name", "")
    round_duration = round_info.get("duration", "")
    interviewer_name = interviewer.get("name", "your interviewer")
    
    # Build natural welcome
    welcome = f"Hi {name}! I'm {interviewer_name}."
    
    if round_name:
        welcome += f" Welcome to your {round_name} for the {job_title} role."
    else:
        welcome += f" Welcome to your interview for the {job_title} position."
    
    if round_duration:
        welcome += f" This should take about {round_duration}."
    
    welcome += " I've gone through your resume. Whenever you're ready, we can get started."
    
    return welcome

