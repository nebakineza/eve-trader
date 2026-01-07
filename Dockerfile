FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    iputils-ping \
    xvfb \
    xauth \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Environment
ENV PYTHONPATH=/app

# Default command (overridden in docker-compose)
CMD ["python", "app.py"]
