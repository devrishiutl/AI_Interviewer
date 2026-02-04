"""
AI Interview Bot API (minimal)

This file intentionally contains ONLY:
- request model classes
- FastAPI app + routes

All implementation details live in `all_main_functions.py`.
"""

import os

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from all_main_functions import handle_start_interview, handle_transcript, handle_feedback


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


class FeedbackRequest(BaseModel):
    interviewId: str
    experience: str  # "How was your experience with AI interviewer?"
    rating: int  # "Rate AI interviewer" (e.g., 1-5 or 1-10)


# -----------------------------------------------------------------------------
# APIs
# -----------------------------------------------------------------------------


@app.post("/api/start-interview")
async def api_start_interview(req: StartInterviewRequest, request: Request):
    try:
        return await handle_start_interview(interview_id=req.interviewId, email=req.email, request=request)
    except HTTPException:
        # Re-raise HTTPException to return proper status codes
        raise
    except ValueError as e:
        # Client error - invalid input
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        # Server error - log and return 500
        import traceback
        error_details = traceback.format_exc()
        print(f"⚠️ /api/start-interview failed: {type(e).__name__}: {e}")
        print(f"Traceback:\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )


@app.post("/api/transcript")
async def api_transcript(req: TranscriptRequest, request: Request):
    try:
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
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"⚠️ /api/transcript failed: {type(e).__name__}: {e}")
        print(f"Traceback:\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )


@app.post("/api/feedback")
async def api_feedback(req: FeedbackRequest, request: Request):
    try:
        return await handle_feedback(
            request=request,
            interview_id=req.interviewId,
            experience=req.experience,
            rating=req.rating,
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"⚠️ /api/feedback failed: {type(e).__name__}: {e}")
        print(f"Traceback:\n{error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {type(e).__name__}: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8021"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

