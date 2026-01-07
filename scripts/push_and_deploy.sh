#!/bin/bash
set -e

# Configuration
REMOTE_USER="seb"
REMOTE_HOST="192.168.14.105"
REMOTE_DIR="/home/seb/nebakineza/eve-trader"

echo "==================================================="
echo "   🚀 Pushing Code from Brain (SKYNET) to Body ($REMOTE_HOST)"
echo "==================================================="

# 1. Create Remote Directory
echo "📂 Ensuring remote directory exists..."
ssh -o StrictHostKeyChecking=no $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"

# 2. Sync Files (Excluding heavy data/git)
echo "cwRsyching files..."
rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
    --exclude '.git' \
    --exclude 'data' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'venv' \
    --exclude '*.deb' \
    ./ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/

# 3. Execute Deployment on Remote
echo "🔥 Triggering Remote Deployment..."
# Using sudo to ensure Docker permissions
ssh -o StrictHostKeyChecking=no -t $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_DIR && sudo bash scripts/deploy_shadow.sh"

echo "==================================================="
echo "   ✅ Sync & Deploy Signal Sent!"
echo "==================================================="
