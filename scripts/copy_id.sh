#!/bin/bash
# One-time setup script to enable passwordless SSH to the Debian server
echo "🔑 Installing SSH Key (id_ed25519) to Debian Host (192.168.14.105)..."
echo "⚠️  You will be prompted for the 'seb' user password one last time."

ssh-copy-id -i ~/.ssh/id_ed25519.pub -o StrictHostKeyChecking=no seb@192.168.14.105

echo "✅ Key installed. You can now execute deployment scripts without passwords."