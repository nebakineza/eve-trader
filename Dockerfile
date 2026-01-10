FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    iputils-ping \
    xvfb \
    xdotool \
    xauth \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
# Note: we keep filenames intact so specialized requirement sets (e.g. requirements.oracle.txt)
# can safely include the base requirements.txt without self-referencing loops.
ARG REQUIREMENTS_FILE=requirements.txt
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS_FILE}

# Copy source
COPY . .

# Environment
ENV PYTHONPATH=/app

# Default command (overridden in docker-compose)
CMD ["python", "app.py"]
