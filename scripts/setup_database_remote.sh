#!/bin/bash
set -e

# Run this on the Production Debian Server (192.168.14.105)
echo "============================================="
echo "   Database & Network Security Setup (Debian)"
echo "============================================="

# 1. Configure Host Firewall (UFW)
# Allow connections from the Training PC
TRAINING_IP="192.168.10.108"

if command -v ufw &> /dev/null; then
    echo "🛡️  Configuring UFW..."
    sudo ufw allow from $TRAINING_IP to any port 5432 proto tcp comment 'Allow Eve DB from Training PC'
    echo "✅ UFW rule added for $TRAINING_IP"
else
    echo "⚠️  UFW not found. Ensure firewall allows TCP port 5432 from $TRAINING_IP."
fi

# 2. Postgres Configuration Notes
# Since we are using Docker, the 'pg_hba.conf' and 'postgresql.conf' are inside the container.
# The default TimescaleDB container listens on '*' and 'Trusts' connections inside the network 
# (which Docker handles via port mapping).
# 
# However, if strict SCRAM-SHA-256 is required by your local security policy:
# You would need to mount a custom config. 

echo "--------------------------------------------------------"
echo "NOTE: Docker Container Configuration"
echo "The 'db' service in docker-compose.yml maps port 5432:5432."
echo "Running: docker-compose exec db cat /var/lib/postgresql/data/pg_hba.conf"
echo ""
echo "To strictly enforce 'host all all $TRAINING_IP/24 scram-sha-256':"
echo "1. Create a custom pg_hba.conf in ./config/postgres/"
echo "2. Mount it in docker-compose.yml volumes."
echo "--------------------------------------------------------"

# For immediate capability, we assume the Docker default + UFW is sufficient.
# To enforce the user's specific request for internal pg_hba modification without mounting:
# We can script an injection into the running container once simple deployment is done.

echo "⚠️  Waiting for DB container to be healthy (if running)..."
if docker ps | grep -q eve-trader-db; then
    echo "🔧 Injecting permissive pg_hba rule for LAN (192.168.10.108)..."
    # Appending rule specifically for the Training PC
    docker exec eve-trader-db bash -c "echo 'host all all 192.168.10.108/32 scram-sha-256' >> /var/lib/postgresql/data/pg_hba.conf"
    
    # Reload config (Running as postgres user)
    docker exec -u postgres eve-trader-db pg_ctl reload -D /var/lib/postgresql/data
    echo "✅ pg_hba.conf updated and reloaded."
else
    echo "❌ DB Container not running. Please run './scripts/deploy_shadow.sh' first."
fi
