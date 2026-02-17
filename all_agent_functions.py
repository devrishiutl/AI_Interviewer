from dotenv import load_dotenv
try:
    load_dotenv()
except Exception:
    # Don't fail import if .env isn't readable in the current environment.
    pass

import os
import re
import time
import base64

import httpx

from livekit.agents.llm import llm as lk_llm
import asyncio

def publish_playground_chat(room, text: str, *, topic: str = "lk-chat") -> None:
    """
    Publish a chat message so it shows up in the LiveKit Playground / useChat UI.
    This uses LiveKit data packets (reliable).
    """
    if not text or not isinstance(text, str):
        return
    lp = getattr(room, "local_participant", None)
    if lp is None:
        return
    # Most LiveKit UIs listen on "lk-chat"; also send on "chat" for compatibility.
    try:
        # asyncio.create_task(lp.publish_data(text, topic=topic, reliable=True))
        # if topic != "chat":
        #     asyncio.create_task(lp.publish_data(text, topic="chat", reliable=True))
        asyncio.create_task(lp.publish_data(text.encode(), topic="chat", reliable=True))

    except RuntimeError:
        # No running loop (shouldn't happen inside an agent job); best-effort.
        return

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
    """
    Create turn-detection model if available; otherwise return None.
    
    Note: Turn detection requires an inference executor. If not available,
    VAD (Voice Activity Detection) will handle turn detection instead.
    """
    # Disable turn detection to avoid "no inference executor" errors
    # VAD will handle turn detection which works without inference executor
    return None


# ============================================================================
# TTS (plug-and-play)
# ============================================================================

def _create_speechify_tts(md: dict) -> "SpeechifyTTS":
    """Create Speechify TTS instance."""
    key = os.getenv("SPEECHIFY_API_KEY")
    if not key:
        raise RuntimeError("SPEECHIFY_API_KEY is missing")
    
    voice = (md.get("interviewer") or {}).get("voice") or "Female"
    voice_id = os.getenv("SPEECHIFY_VOICE_ID_MALE" if voice == "Male" else "SPEECHIFY_VOICE_ID_FEMALE", "lauren")
    return SpeechifyTTS(api_key=key, voice_id=voice_id)


def _create_sarvam_tts(md: dict, sarvam_plugin) -> "lk_tts.TTS":
    """Create Sarvam TTS instance."""
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise RuntimeError("SARVAM_API_KEY is missing")
    
    if sarvam_plugin is None:
        raise RuntimeError("Sarvam plugin not loaded")
    
    voice = (md.get("interviewer") or {}).get("voice") or "Female"
    speaker = "abhilash" if voice == "Male" else "anushka"
    
    return sarvam_plugin.TTS(
        target_language_code="en-IN",
        model="bulbul:v2",
        speaker=speaker,
        api_key=key,
    )


# Legacy function for backward compatibility
def tts_speaker(md: dict) -> str:
    voice = (md.get("interviewer") or {}).get("voice") or "Female"
    return "abhilash" if voice == "Male" else "anushka"


# ============================================================================
# LLM (plug-and-play via DSPy)
# ============================================================================

def _llm_backend() -> str:
    return (os.getenv("LLM_BACKEND") or "livekit").strip().lower()


def _llm_model() -> str:
    return (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()


def _llm_temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0.3"))
    except Exception:
        return 0.3


def _dspy_provider() -> str:
    # matches your sample naming: openai / claude / anthropic
    return (os.getenv("DSPY_PROVIDER") or "openai").strip().lower()


def _dspy_model_path(provider: str, model_name: str) -> str:
    if provider in ("claude", "anthropic"):
        return f"anthropic/{model_name}"
    if provider == "openai":
        return f"openai/{model_name}"
    if provider == "azure":
        # Azure OpenAI: LiteLLM expects just "azure/<deployment_name>"
        # The endpoint/version are set via environment variables (see _run)
        return f"azure/{model_name}"
    # If you want a custom backend, set DSPY_MODEL_PATH directly.
    custom = (os.getenv("DSPY_MODEL_PATH") or "").strip()
    if custom:
        return custom
    raise RuntimeError(f"Unsupported DSPY_PROVIDER={provider!r} (set DSPY_MODEL_PATH for custom providers)")


def _chat_ctx_to_prompt(chat_ctx: lk_llm.ChatContext) -> str:
    # Minimal prompt formatting: preserve roles and content.
    parts: list[str] = []
    for item in getattr(chat_ctx, "items", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        role = getattr(item, "role", None) or "user"
        content_list = getattr(item, "content", None) or []
        text = content_list[0] if isinstance(content_list, list) and content_list else ""
        if isinstance(text, str) and text.strip():
            parts.append(f"{role}: {text.strip()}")
    parts.append("assistant:")
    return "\n".join(parts).strip()


class _DspyChatSignature:  # defined lazily inside _run to avoid importing dspy at import-time
    pass


class DspyLLMStream(lk_llm.LLMStream):
    async def _run(self) -> None:
        # Import only when used
        try:
            import dspy  # type: ignore
        except Exception as e:
            raise RuntimeError(f"DSPy is not installed: {type(e).__name__}: {e}") from e

        prompt = _chat_ctx_to_prompt(self.chat_ctx)

        provider = _dspy_provider()
        model_name = _llm_model()
        api_key = (os.getenv("DSPY_API_KEY") or "").strip() or None
        if not api_key:
            raise RuntimeError("Missing DSPY_API_KEY (or OPENAI_API_KEY) for DSPy LLM")

        model_path = _dspy_model_path(provider, model_name)
        temperature = _llm_temperature()

        # For Azure, set environment variables that LiteLLM expects
        if provider == "azure":
            api_base = (os.getenv("AZURE_OPENAI_API_BASE") or "").strip()
            api_version = (os.getenv("AZURE_OPENAI_API_VERSION") or "2025-01-01-preview").strip()
            if not api_base:
                raise RuntimeError("AZURE_OPENAI_API_BASE is required when DSPY_PROVIDER=azure")
            # Temporarily set env vars for LiteLLM to pick up
            os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
            os.environ["AZURE_OPENAI_API_VERSION"] = api_version
            os.environ["AZURE_OPENAI_API_KEY"] = api_key

        # Minimal DSPy signature: prompt -> response
        class ChatSig(dspy.Signature):
            prompt: str = dspy.InputField()
            response: str = dspy.OutputField()

        # Ensure api_base is never None for LiteLLM (workaround for LiteLLM logging bug)
        # DSPy/LiteLLM internally may pass None, so we set it explicitly if needed
        lm_kwargs = {"api_key": api_key, "cache": False, "temperature": temperature}
        if provider == "azure" and api_base:
            # For Azure, ensure api_base is set to avoid LiteLLM logging errors
            lm_kwargs["api_base"] = api_base
        lm = dspy.LM(model_path, **lm_kwargs)
        with dspy.context(lm=lm):
            predictor = dspy.Predict(ChatSig)
            pred = predictor(prompt=prompt)
            text = (getattr(pred, "response", "") or "").strip()

        # Emit a single delta chunk
        chunk_id = f"dspy-{int(time.time() * 1000)}"
        self._event_ch.send_nowait(
            lk_llm.ChatChunk(
                id=chunk_id,
                delta=lk_llm.ChoiceDelta(role="assistant", content=text),
            )
        )


class DspyLLM(lk_llm.LLM):
    def __init__(self):
        super().__init__()
        self._model_name = _llm_model()
        self._label = "dspy"

    @property
    def model(self) -> str:
        return self._model_name or "unknown"

    def chat(
        self,
        *,
        chat_ctx: lk_llm.ChatContext,
        tools=None,
        conn_options=None,
        parallel_tool_calls=None,
        tool_choice=None,
        extra_kwargs=None,
    ) -> lk_llm.LLMStream:
        # Tools are intentionally ignored for now (interview flow doesn't need function calling).
        if conn_options is None:
            conn_options = lk_llm.DEFAULT_API_CONNECT_OPTIONS
        return DspyLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


def create_llm(*, openai_plugin):
    """
    Factory:
    - LLM_BACKEND=livekit (default): uses livekit.plugins.openai.LLM
    - LLM_BACKEND=dspy: uses DSPy (set DSPY_PROVIDER, DSPY_API_KEY, LLM_MODEL)
    """
    backend = _llm_backend()
    if backend == "dspy":
        return DspyLLM()
    if backend == "livekit":
        return openai_plugin.LLM(model=_llm_model(), temperature=_llm_temperature())
    raise RuntimeError(f"Unknown LLM_BACKEND={backend!r}")


from livekit.agents import tts as lk_tts


class SpeechifyTTS(lk_tts.TTS):
    """
    Minimal Speechify TTS adapter for LiveKit Agents.

    Uses the Speechify AI API with API key auth:
      Authorization: Bearer <SPEECHIFY_API_KEY>
    """

    def __init__(self, *, api_key: str, voice_id: str, output_format: str | None = None):
        super().__init__(
            capabilities=lk_tts.TTSCapabilities(streaming=False, aligned_transcript=False),
            sample_rate=24000,
            num_channels=1,
        )
        self._api_key = api_key.strip()
        self._voice_id = voice_id.strip()
        self._output_format = (output_format or "").strip().lower() or None

    def synthesize(self, text: str, *, conn_options=None):
        if conn_options is None:
            conn_options = lk_tts.DEFAULT_API_CONNECT_OPTIONS

        api_key, voice_id, output_format = self._api_key, self._voice_id, self._output_format

        class _SpeechifyChunkedStream(lk_tts.ChunkedStream):
            async def _run(self, output_emitter):
                base_url = (os.getenv("SPEECHIFY_BASE_URL") or "https://api.sws.speechify.com").rstrip("/")
                url = f"{base_url}/v1/audio/speech"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {"input": text, "voice_id": voice_id}
                if output_format:
                    payload["audio_format"] = output_format

                async with httpx.AsyncClient(timeout=conn_options.timeout) as client:
                    r = await client.post(url, headers=headers, json=payload)
                    ct = (r.headers.get("content-type") or "").lower()
                    if r.status_code >= 300:
                        body = (r.text or "").strip()
                        raise RuntimeError(f"Speechify TTS failed {r.status_code}: {body[:1000]}")

                    request_id = r.headers.get("x-request-id") or r.headers.get("x-requestid") or "speechify"
                    mime_type = "audio/mpeg"
                    audio_bytes = None

                    # Some Speechify setups can return raw audio; docs show JSON with base64 `audio_data`.
                    if ct.startswith("audio/"):
                        audio_bytes = r.content
                        mime_type = ct.split(";", 1)[0].strip() or mime_type
                    else:
                        data = r.json() if r.content else {}
                        audio_b64 = (data.get("audio_data") or data.get("audioData")) if isinstance(data, dict) else None
                        if isinstance(audio_b64, str) and audio_b64.strip():
                            audio_bytes = base64.b64decode(audio_b64)
                            fmt = (data.get("audio_format") or data.get("audioFormat") or "").strip().lower()
                            if fmt == "wav":
                                mime_type = "audio/wav"

                    if not audio_bytes:
                        body = (r.text or "").strip()
                        raise RuntimeError(f"Speechify TTS returned no audio bytes (status={r.status_code}, ct={ct}, body={body[:1000]})")

                    output_emitter.initialize(
                        request_id=request_id,
                        # livekit.agents stores TTS on the stream instance as `_tts`
                        sample_rate=self._tts.sample_rate,
                        num_channels=self._tts.num_channels,
                        mime_type=mime_type,
                        stream=False,
                    )
                    output_emitter.push(audio_bytes)
                    output_emitter.flush()

        return _SpeechifyChunkedStream(tts=self, input_text=text, conn_options=conn_options)


def create_tts(*, md: dict, sarvam_plugin=None):
    """
    Factory function to create TTS instance based on TTS_PROVIDER.
    
    Set TTS_PROVIDER=speechify or TTS_PROVIDER=sarvam in .env
    Voice is automatically selected from interview metadata (interviewer.voice)
    """
    provider = (os.getenv("TTS_PROVIDER") or "").strip().lower()
    
    if provider in ("none", "off", "0", "false", ""):
        return None
    
    if provider == "speechify":
        return _create_speechify_tts(md)
    
    if provider == "sarvam":
        return _create_sarvam_tts(md, sarvam_plugin)
    
    raise RuntimeError(f"Unknown TTS_PROVIDER={provider!r} (supported: speechify, sarvam, none)")


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

