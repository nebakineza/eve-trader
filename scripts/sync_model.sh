#!/bin/bash
set -e

# Sync Model from Training PC (WSL2) to Debian Server
# Run this on the WSL2 machine after training.

DEBIAN_HOST="jellyfin.towerhouse.lan"
DEBIAN_USER="seb"
DEBIAN_PATH="~/nebakineza/eve-trader"

echo "=========================================="
echo "   Model Sync: WSL2 -> Debian Server      "
echo "=========================================="

if [ ! -f "oracle_v1.pth" ]; then
    echo "❌ Error: oracle_v1.pth not found. Run training first."
    exit 1
fi

echo "📤 Uploading oracle_v1.pth..."
scp oracle_v1.pth $DEBIAN_USER@$DEBIAN_HOST:$DEBIAN_PATH/oracle_v1.pth

echo "📤 Uploading oracle_v1.onnx..."
scp oracle_v1.onnx $DEBIAN_USER@$DEBIAN_HOST:$DEBIAN_PATH/oracle_v1.onnx

echo "♻️  Restarting Oracle Service on Remote..."
ssh $DEBIAN_USER@$DEBIAN_HOST "cd $DEBIAN_PATH && docker-compose restart oracle"

echo "✅ Sync Complete. Oracle updated."
