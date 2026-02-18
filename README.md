Local endpoints
- `http://localhost:8021/api/start-interview`
- `http://localhost:8021/api/transcript`
- `http://localhost:8021/api/feedback`

Playground (Easiest)
1. Go to: https://agents-playground.livekit.io/
2. Click "Connect" (top right)
3. Enter:
	- LiveKit URL: `wss://voicebot-kj0vxeoj.livekit.cloud`
	- Token: (paste the token from Step 1 response)
4. Click "Connect"

Running with Docker

Prerequisites:
- Docker Engine
- Docker Compose (or use `docker compose` on newer Docker)

Build the image:

```bash
docker build -t ai-interviewer .
```

Run with Docker directly:

```bash
docker run --rm -p 8021:8021 --env-file .env --name ai_interviewer_app ai-interviewer
```

Or use docker-compose (recommended for local dev):

```bash
docker compose up --build
```

Notes:
- This repository reads configuration from a `.env` file. The project includes example env entries in `.env` — ensure sensitive keys are set appropriately before running in production.
- The `requirements.txt` was generated from `pyproject.toml`. Some audio/LLM-related packages may require system libraries (e.g., `ffmpeg`, `libsndfile`) — the `Dockerfile` installs common runtime packages, but you may need to extend it for specific providers.
- Use `curl` or your browser to hit the health endpoint `http://localhost:8021/` to verify the app is running.

If you want, I can add a small `Makefile` to simplify build/run steps.
