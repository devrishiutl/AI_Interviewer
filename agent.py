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
import traceback
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions
from all_agent_functions import (
    build_instructions,
    build_welcome,
    create_llm,
    create_tts,
    dedupe_text,
    download_turn_detector_files,
    maybe_turn_detection,
    publish_playground_chat,
    post_transcript,
    require_env,
    tts_speaker,
)

_DOTENV_PATH = Path(__file__).with_name(".env")
try:
    load_dotenv(dotenv_path=_DOTENV_PATH, override=False)
except Exception:
    # Don't fail module import if .env isn't readable in the current environment.
    pass


_PLUGINS = {
    "deepgram": None,
    "noise_cancellation": None,
    "openai": None,
    "sarvam": None,
    "silero": None,
}


def prewarm(proc: agents.JobProcess):
    """
    LiveKit plugins register themselves at import time.
    On Windows (and with multiprocessing 'spawn'), importing plugins from a job task can crash with:
      RuntimeError: Plugins must be registered on the main thread

    prewarm_fnc runs on the job process main thread, and must be a TOP-LEVEL function
    (so it can be pickled/imported by multiprocessing).
    """
    try:
        from livekit.plugins import deepgram, noise_cancellation, openai, sarvam, silero

        _PLUGINS["deepgram"] = deepgram
        _PLUGINS["noise_cancellation"] = noise_cancellation
        _PLUGINS["openai"] = openai
        _PLUGINS["sarvam"] = sarvam
        _PLUGINS["silero"] = silero
    except Exception:
        # Let the worker boot; failures will be visible in logs.
        print("⚠️ Plugin prewarm import failed:")
        print(traceback.format_exc())


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
        self._room_obj = None
        self._last_user = ("", 0)
        self._last_assistant = ("", 0)

    def set_ctx(self, ctx: dict):
        self._ctx.update(ctx or {})
        self._room_obj = (ctx or {}).get("roomObj")

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
            asyncio.create_task(post_transcript(self._ctx, role="Candidate", text=content))
            # Publish candidate message to data channel with sender info
            if self._room_obj:
                # Include sender info so frontend can identify candidate messages
                import json
                msg_data = json.dumps({"role": "candidate", "text": content})
                publish_playground_chat(self._room_obj, msg_data)
        elif message.role == "assistant":
            dup, upd = self._dedupe(self._last_assistant, content)
            if dup:
                return
            self._last_assistant = upd
            asyncio.create_task(post_transcript(self._ctx, role="Interviewer", text=content))
            # Publish interviewer message to data channel with sender info
            if self._room_obj:
                import json
                msg_data = json.dumps({"role": "interviewer", "text": content})
                publish_playground_chat(self._room_obj, msg_data)

    def conversation_item_added(self, message):
        return self._conversation_item_added(message)

    # def on_conversation_item_added(self, message):
    #     return self._conversation_item_added(message)


async def _wait_for_metadata(ctx: agents.JobContext) -> None:
    """Wait for room metadata to arrive."""
    while not ctx.room.metadata:
        await asyncio.sleep(0.25)


async def _wait_for_candidate(ctx: agents.JobContext) -> None:
    """Wait for candidate to join using simple polling."""
    while True:
        try:
            remotes = getattr(ctx.room, "remote_participants", None) or {}
            if any(getattr(p, "identity", "").startswith("candidate-") for p in remotes.values()):
                return
        except Exception:
            pass
        await asyncio.sleep(0.5)


async def entrypoint(ctx: agents.JobContext):
    try:
        print("🎯 Agent job received; starting…")
        # Plugins are imported/registered during process prewarm on the main thread (see prewarm()).
        deepgram = _PLUGINS["deepgram"]
        noise_cancellation = _PLUGINS["noise_cancellation"]
        openai = _PLUGINS["openai"]
        sarvam = _PLUGINS["sarvam"]
        silero = _PLUGINS["silero"]
        if not all([deepgram, noise_cancellation, openai, sarvam, silero]):
            raise RuntimeError("LiveKit plugins not loaded. Did prewarm() fail?")

        await ctx.connect()

        # Wait for metadata with timeout
        try:
            await asyncio.wait_for(_wait_for_metadata(ctx), timeout=5.0)
        except asyncio.TimeoutError:
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

        try:
            tts = create_tts(md=md, sarvam_plugin=sarvam)
        except Exception as e:
            print(f"❌ Failed to create TTS: {type(e).__name__}: {e}")
            return
        if tts is None:
            # If TTS is missing/disabled, LiveKit Playground will show "Waiting for agent audio track…".
            # Make this explicit so it doesn't look like the agent is "disabled".
            print(
                "❌ TTS is not configured (no audio will be published). "
                "Set SPEECHIFY_API_KEY (+ SPEECHIFY_VOICE_ID) or SARVAM_API_KEY, "
                "or set TTS_PROVIDER=speechify|sarvam explicitly."
            )
            return

        session = InterviewSession(
            stt=deepgram.STT(model="nova-2", language="en"),
            llm=create_llm(openai_plugin=openai),
            tts=tts,
            vad=silero.VAD.load(),
            turn_detection=maybe_turn_detection(),
        )

        session.set_ctx(
            {
                "room": getattr(ctx.room, "name", "") or "",
                "roomObj": ctx.room,
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
            # Note: RoomInputOptions is deprecated but still works in current version
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
                close_on_disconnect=False
            ),
        )

        # Wait for candidate
        try:
            await asyncio.wait_for(_wait_for_candidate(ctx), timeout=60.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            raise

        welcome = build_welcome(md)
        try:
            await session.say(welcome)
        except Exception as e:
            print(f"⚠️ welcome TTS failed: {type(e).__name__}: {e}")
        publish_playground_chat(ctx.room, welcome)
        asyncio.create_task(post_transcript(session._ctx, role="Interviewer", text=welcome))

        while True:
            await asyncio.sleep(1)
    except Exception:
        print("❌ Agent crashed while starting/joining:")
        print(traceback.format_exc())
        raise


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
            prewarm_fnc=prewarm,
            agent_name=os.getenv("LIVEKIT_AGENT_NAME", "interview-agent"),
            # Reduce "assignment timed out" by keeping at least one warm job process ready.
            # In dev the default can be 0 which forces a cold spawn on first job.
            num_idle_processes=1,
            # Give the process more time to initialize under load / slow machines.
            initialize_process_timeout=30.0,
        )
    )

