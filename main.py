"""
AI Interview Bot API (minimal)

This file intentionally contains ONLY:
- request model classes
- FastAPI app + routes

All implementation details live in `all_main_functions.py`.
"""

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from all_main_functions import handle_start_interview, handle_transcript


app = FastAPI(title="AI Interview Bot API")

# Allow browser clients to call this API (local dev / playground / frontend)
app.add_middleware(
    CORSMiddleware,
    # If allow_credentials=True, CORS cannot use wildcard "*" origins.
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Request models
# -----------------------------------------------------------------------------


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


@app.post("/api/start-interview")
async def api_start_interview(req: StartInterviewRequest, request: Request):
    return await handle_start_interview(interview_id=req.interviewId, email=req.email, request=request)


@app.post("/api/transcript")
async def api_transcript(req: TranscriptRequest, request: Request):
    return await handle_transcript(
        request=request,
        interview_id=req.interviewId,
        role=req.role,
        speaker_name=req.speakerName,
        text=req.text,
        timestamp_ms=req.timestampMs,
        room=req.room,
        transcript_record_id=req.transcriptRecordId,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8021"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

