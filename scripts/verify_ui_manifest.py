#!/usr/bin/env python3
"""
scripts/verify_ui_manifest.py

Uses the Visual Cortex to verify that the Game Client windows align with
the authoritative ui_manifest.json.

Run on the Host (192.168.14.105).
"""

import json
import os
import sys
import logging
import cv2
import numpy as np

# Path Hygiene
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from nexus.automaton.visual_cortex import VisualCortex
except ImportError:
    print("FATAL: Could not import VisualCortex.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ManifestVerify] - %(levelname)s - %(message)s')
logger = logging.getLogger("ManifestVerify")

def load_manifest():
    path = os.path.join(current_dir, "ui_manifest.json")
    if not os.path.exists(path):
        logger.error(f"Manifest not found at {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)

def verify_region(vc, frame, window_name, config):
    region = config.get("region") # [x, y, w, h]
    expected_text = config.get("ocr_check_text")
    
    if not region or not expected_text:
        return False

    x, y, w, h = region
    
    # Crop the frame
    # Ensure bounds
    fh, fw = frame.shape[:2]
    x = max(0, min(x, fw))
    y = max(0, min(y, fh))
    w = min(w, fw - x)
    h = min(h, fh - y)
    
    roi = frame[y:y+h, x:x+w]
    
    # Enhance
    processed = vc.enhance_for_ocr(roi)
    
    # OCR
    text = vc.run_ocr(processed)
    
    logger.info(f"Checking {window_name}... Expected: '{expected_text}', Found: '{text}'")
    
    # Loose match
    if expected_text.lower() in text.lower():
        return True
    return False

def main():
    manifest = load_manifest()
    if not manifest:
        sys.exit(1)

    vc = VisualCortex()
    logger.info("Capturing Frame...")
    frame = vc.capture_frame()
    
    if frame is None:
        logger.error("Failed to capture frame.")
        sys.exit(1)
        
    results = {}
    
    for name, config in manifest["windows"].items():
        if "ocr_check_text" in config:
            passed = verify_region(vc, frame, name, config)
            results[name] = passed
            if passed:
                logger.info(f"✅ {name.upper()} ALIGNED")
            else:
                logger.warning(f"❌ {name.upper()} MISALIGNED")

    # Summary
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    print("-" * 30)
    print(f"Manifest Verification: {success_count}/{total_count}")
    print("-" * 30)
    
    if success_count == total_count:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
