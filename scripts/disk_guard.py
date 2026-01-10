#!/usr/bin/env python3
import shutil
import subprocess
import logging
import sys

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/eve_disk_guard.log")
    ]
)
logger = logging.getLogger("DiskGuard")

THRESHOLD_WARNING = 75.0  # %
THRESHOLD_CRITICAL = 85.0 # %
TARGET_PATH = "/"  # Root filesystem

def get_disk_usage(path):
    total, used, free = shutil.disk_usage(path)
    percent_used = (used / total) * 100
    return percent_used, free // (2**30) # Free in GB

def run_docker_prune(all_resources=False):
    """Runs docker system prune."""
    cmd = ["docker", "system", "prune", "-f"]
    if all_resources:
        cmd.append("-a") # Remove unused images not just dangling
    
    logger.info(f"Running cleanup command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info("Cleanup Result:\n" + result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Cleanup failed: {e.stderr}")

def main():
    logger.info("Checking disk usage...")
    percent_used, free_gb = get_disk_usage(TARGET_PATH)
    
    logger.info(f"Disk Usage: {percent_used:.2f}% (Free: {free_gb} GB)")

    # 1. First Line of Defense: Prune Build Cache (Safest)
    # The user specifically mentioned "multiple docker builds" causing the issue.
    # We should aggressively prune the builder cache.
    try:
        logger.info("Pruning Docker builder cache > 12h...")
        subprocess.run(["docker", "builder", "prune", "-f", "--filter", "until=12h"], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # 2. Re-check and Escalation
    percent_used, free_gb = get_disk_usage(TARGET_PATH)
    
    if percent_used > THRESHOLD_CRITICAL:
        logger.warning(f"CRITICAL: Disk usage is {percent_used:.2f}%! Initiating AGGRESSIVE cleanup.")
        # Aggressive: Remove all unused images, dangling volumes, networks
        run_docker_prune(all_resources=True)
        subprocess.run(["docker", "volume", "prune", "-f"], check=False)
        
    elif percent_used > THRESHOLD_WARNING:
        logger.warning(f"WARNING: Disk usage is {percent_used:.2f}%. Initiating standard cleanup.")
        # Standard: Remove dangling images and stopped containers
        run_docker_prune(all_resources=False)
    else:
        logger.info("Disk usage is acceptable.")

if __name__ == "__main__":
    main()
