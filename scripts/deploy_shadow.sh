#!/bin/bash
set -e

# EVE Trader Shadow Mode Deployment Script
echo "=========================================="
echo "   EVE Trader - Shadow Mode Deployment    "
echo "=========================================="

# 1. Check Pre-requisites
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    # Check for 'docker compose' plugin style as well
    if ! docker compose version &> /dev/null; then
        echo "❌ Error: Docker Compose is not installed."
        exit 1
    fi
fi
echo "✅ Docker environment found."

# 2. Setup Data Directories
echo "📂 Setting up data directories..."
mkdir -p data/postgres
mkdir -p data/redis

# Set permissions for Postgres (UID 999 is default for postgres container)
# We use sudo if current user doesn't have permissions, but script might run as user.
# Warning about sudo if needed.
if [ -w "data/postgres" ]; then
    echo "   Permissions check passed."
else
    echo "⚠️  Setting permissions for data/postgres (requires sudo)..."
    sudo chown -R 999:999 data/postgres
fi
echo "✅ Directories ready."

# 3. Environment Configuration
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "⚙️  Configuration (.env not found)"
    read -sp "   Enter a secure Password for the Database: " DB_PASS
    echo ""
    
    echo "POSTGRES_PASSWORD=$DB_PASS" > $ENV_FILE
    echo "POSTGRES_USER=eve_user" >> $ENV_FILE
    echo "POSTGRES_DB=eve_market_data" >> $ENV_FILE
    echo "ORACLE_DEVICE=cpu" >> $ENV_FILE
    echo "SYSTEM_MODE=SHADOW" >> $ENV_FILE
    
    echo "✅ .env file created."
else
    echo "✅ Using existing .env file."
fi

# Ensure .env is ignored by git (if inside git repo)
if [ -d ".git" ]; then
    if ! grep -q ".env" .gitignore; then
        echo ".env" >> .gitignore
        echo "✅ Added .env to .gitignore."
    fi
fi

# 4. Launch Services
echo "🚀 Launching Docker Stack..."
# Support both hyphenated and space-separated docker compose
if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
else
    docker compose up -d --build
fi

echo "=========================================="
echo "✅ Deployment Complete!"
echo "   Dashboard: http://localhost:8501"
echo "   API:       http://localhost:8000"
echo "=========================================="
