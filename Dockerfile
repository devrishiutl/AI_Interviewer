FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Install minimal system deps (extend if needed for audio/ML packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       gfortran \
       python3-dev \
       pkg-config \
       libopenblas-dev \
       liblapack-dev \
       libsndfile1 \
       ffmpeg \
       ninja-build \
       curl \
       meson \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app

EXPOSE 8021

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8021"]
