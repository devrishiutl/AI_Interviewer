"""
AI Interview Agent (minimal)
- Reads interview data from LiveKit room metadata
- Speaks welcome + conducts interview
- Posts both user + assistant messages to /api/transcript
- Includes `python agent.py download-files` for turn-detector assets
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from all_functions import (
    build_instructions,
    build_welcome,
    dedupe_text,
    download_turn_detector_files,
    maybe_turn_detection,
    post_transcript,
    require_env,
    tts_speaker,
)

try:
    load_dotenv()
except Exception:
    # Don't fail module import if .env isn't readable in the current environment.
    pass


class InterviewerAgent(Agent):
    def __init__(self, instructions: str):
        super().__init__(instructions=instructions)


class InterviewSession(AgentSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ctx = {
            "room": "",
            "interviewId": None,
            "transcriptRecordId": None,
            "candidateName": "Candidate",
            "interviewerName": "Interviewer",
            "transcriptApiUrl": None,
        }
        self._last_user = ("", 0)
        self._last_assistant = ("", 0)

    def set_ctx(self, ctx: dict):
        self._ctx.update(ctx or {})

    def _dedupe(self, last: tuple[str, int], text: str) -> tuple[bool, tuple[str, int]]:
        return dedupe_text(last, text)

    def _conversation_item_added(self, message):
        if not hasattr(message, "role"):
            return
        content = message.content[0] if isinstance(message.content, list) else message.content
        if not isinstance(content, str) or not content.strip():
            return

        if message.role == "user":
            dup, upd = self._dedupe(self._last_user, content)
            if dup:
                return
            self._last_user = upd
            print(f"\n👤 CANDIDATE: {content}")
            asyncio.create_task(post_transcript(self._ctx, role="Candidate", text=content))
        elif message.role == "assistant":
            dup, upd = self._dedupe(self._last_assistant, content)
            if dup:
                return
            self._last_assistant = upd
            print(f"\n🎤 INTERVIEWER: {content}")
            asyncio.create_task(post_transcript(self._ctx, role="Interviewer", text=content))

    # LiveKit Agents versions differ on which callback is invoked.
    def conversation_item_added(self, message):
        return self._conversation_item_added(message)

    def on_conversation_item_added(self, message):
        return self._conversation_item_added(message)


async def entrypoint(ctx: agents.JobContext):
    print("🎯 Agent starting...")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_key:
        print("❌ SARVAM_API_KEY missing")
        return

    # Lazy imports so module import doesn't hard-fail if plugins aren't installed.
    from livekit.plugins import deepgram, noise_cancellation, openai, sarvam, silero

    await ctx.connect()
    # Metadata can arrive slightly after connect.
    for _ in range(10):
        if ctx.room.metadata:
            break
        await asyncio.sleep(0.3)
    if not ctx.room.metadata:
        print("❌ No room metadata (start interview via /api/start-interview)")
        return

    try:
        md = json.loads(ctx.room.metadata)
    except Exception as e:
        print(f"❌ Invalid metadata JSON: {e}")
        return
    if not isinstance(md, dict) or not md.get("applicant") or not md.get("job"):
        print("❌ Missing required metadata (applicant/job)")
        return

    session = InterviewSession(
        stt=deepgram.STT(model="nova-2", language="en"),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.3),
        tts=sarvam.TTS(
            target_language_code="en-IN",
            model="bulbul:v2",
            speaker=tts_speaker(md),
            api_key=sarvam_key,
        ),
        vad=silero.VAD.load(),
        turn_detection=maybe_turn_detection(),
    )

    session.set_ctx(
        {
            "room": getattr(ctx.room, "name", "") or "",
            "interviewId": md.get("interviewId"),
            "transcriptRecordId": md.get("transcriptRecordId"),
            "candidateName": (md.get("applicant") or {}).get("name") or "Candidate",
            "interviewerName": (md.get("interviewer") or {}).get("name") or "Interviewer",
            "transcriptApiUrl": md.get("transcriptApiUrl"),
        }
    )

    await session.start(
        room=ctx.room,
        agent=InterviewerAgent(instructions=build_instructions(md)),
        # Keep agent running even if the browser disconnects/reconnects.
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC(), close_on_disconnect=False),
    )

    welcome = build_welcome(md)
    await session.say(welcome)
    # Ensure welcome is captured even if the SDK doesn't emit a conversation-item callback for TTS.
    asyncio.create_task(post_transcript(session._ctx, role="Interviewer", text=welcome))

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "download-files":
        download_turn_detector_files()
        raise SystemExit(0)

    require_env("LIVEKIT_URL")
    require_env("LIVEKIT_API_KEY")
    require_env("LIVEKIT_API_SECRET")

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "interview-agent"),
        )
    )

