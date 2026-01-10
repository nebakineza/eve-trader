import json
import os
from pathlib import Path

CONFIG_PATH = Path("/app/config.json")
LOCAL_CONFIG_PATH = Path("config.json")

def load_config():
    path = CONFIG_PATH if CONFIG_PATH.exists() else LOCAL_CONFIG_PATH
    if not path.exists():
        # Fallback defaults
        return {
            "model": {"d_coefficient": 0.4, "features": ["OBI", "Z_Score"]},
            "risk": {"z_score_threshold": 1.96}
        }
    with open(path, "r") as f:
        return json.load(f)

GLOBAL_CONFIG = load_config()
