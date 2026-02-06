"""
LLM functions for main.py API endpoints.

Uses DSPy for LLM interactions (job description generation, etc.)
"""

import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path

_DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=_DOTENV_PATH, override=False)


async def generate_job_description(
    *,
    job_title: str,
    industry: str | None = None,
    job_level: str | None = None,
    job_type: str | None = None,
    min_exp: int | None = None,
    max_exp: int | None = None,
) -> str:
    """Generate job description using DSPy LLM."""
    try:
        import dspy
    except ImportError:
        raise RuntimeError("DSPy is not installed. Install it with: pip install dspy-ai")
    
    provider = (os.getenv("DSPY_PROVIDER") or "openai").strip().lower()
    model_name = (os.getenv("LLM_MODEL") or "gpt-4o-mini").strip()
    api_key = (os.getenv("DSPY_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing DSPY_API_KEY or OPENAI_API_KEY")
    
    try:
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7
    
    # Model path
    if provider in ("claude", "anthropic"):
        model_path = f"anthropic/{model_name}"
    elif provider == "openai":
        model_path = f"openai/{model_name}"
    elif provider == "azure":
        model_path = f"azure/{model_name}"
        api_base = (os.getenv("AZURE_OPENAI_API_BASE") or "").strip()
        if not api_base:
            raise RuntimeError("AZURE_OPENAI_API_BASE is required when DSPY_PROVIDER=azure")
        os.environ["AZURE_OPENAI_ENDPOINT"] = api_base
        os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
    else:
        model_path = os.getenv("DSPY_MODEL_PATH", "")
        if not model_path:
            raise RuntimeError(f"Unsupported DSPY_PROVIDER={provider!r}")
    
    # Build prompt
    prompt = f"""Generate a comprehensive job description for the position: {job_title}

Please include:
- Job overview and purpose
- Key responsibilities
- Required qualifications and skills
- Preferred qualifications
- Experience requirements
- Education requirements (if applicable)

Write a professional, detailed job description that would attract qualified candidates."""

    # DSPy signature
    class JobDescriptionSig(dspy.Signature):
        prompt: str = dspy.InputField(desc="Job details and requirements")
        job_description: str = dspy.OutputField(desc="Complete job description")
    
    # Initialize and generate
    lm_kwargs = {"api_key": api_key, "cache": False, "temperature": temperature}
    if provider == "azure":
        lm_kwargs["api_base"] = api_base
    lm = dspy.LM(model_path, **lm_kwargs)
    
    def _generate():
        with dspy.context(lm=lm):
            result = dspy.Predict(JobDescriptionSig)(prompt=prompt)
            return (getattr(result, "job_description", "") or "").strip()
    
    job_description = await asyncio.to_thread(_generate)
    if not job_description:
        raise RuntimeError("Failed to generate job description")
    return job_description
