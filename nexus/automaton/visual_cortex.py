#!/usr/bin/env python3
"""
nexus/automaton/visual_cortex.py

High-performance frame buffer analysis using iGPU (OpenCL) and OCR.
Replacing simple color masking with state-aware UI detection.
"""

import cv2
import numpy as np
import subprocess
import os
import logging
import sys
import time

# Attempt to configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - [VisualCortex] - %(levelname)s - %(message)s')
logger = logging.getLogger("VisualCortex")

class VisualCortex:
    def __init__(self, display=":0"):
        self.display = display
        
        # 1. Enable OpenCL for iGPU acceleration
        if cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)
            logger.info(f"OpenCL Acceleration: ENABLED (Device: {cv2.ocl.Device.getDefault().name()})")
        else:
            logger.warning("OpenCL Acceleration: UNAVAILABLE - Falling back to CPU")

        # Define Launcher States
        self.STATE_PLAY_READY = "STATE_PLAY_READY"
        self.STATE_UPDATE_REQUIRED = "STATE_UPDATE_REQUIRED"
        self.STATE_VERIFYING = "STATE_VERIFYING"
        self.STATE_UNKNOWN = "STATE_UNKNOWN"

    def capture_frame(self):
        """
        Captures the X11 screen frame buffer utilizing scrot.
        Returns: numpy array (BGR)
        """
        env = os.environ.copy()
        env["DISPLAY"] = self.display
        try:
            # scrot to stdout is fast and avoids disk I/O
            cmd = ["scrot", "-z", "-"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            stdout, stderr = proc.communicate(timeout=5)

            if proc.returncode != 0:
                logger.error(f"Frame capture failed: {stderr.decode()}")
                return None

            # Decode from memory
            buf = np.frombuffer(stdout, np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.error(f"Critical capture error: {e}")
            return None

    def enhance_for_ocr(self, frame):
        """
        Applies high-contrast filtering to prepare for OCR.
        Converts to Grayscale -> Thresholding.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply standard thresholding (Otsu's binarization is usually good for text)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def run_ocr(self, img_region):
        """
        Runs Tesseract OCR on a specific image region.
        """
        try:
            # Encode as PNG for transmission to tesseract stdin
            success, encoded_img = cv2.imencode('.png', img_region)
            if not success:
                return ""
            
            # PSM 7: Treat as single text line
            cmd = ["tesseract", "stdin", "stdout", "--psm", "7"] 
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = proc.communicate(input=encoded_img.tobytes(), timeout=3)
            return out.decode().strip()
        except FileNotFoundError:
            logger.error("Tesseract not found. Install tesseract-ocr.")
            return ""
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""

    def detect_play_button(self, hsv_frame):
        """
        Detects the Teal 'PLAY NOW' button with strict color filtering.
        """
        # Teal (Play Button): #00FFEE -> BGR(238, 255, 0)
        # HSV check needed.
        lower_teal = np.array([80, 100, 100])
        upper_teal = np.array([100, 255, 255])
        
        teal_mask = cv2.inRange(hsv_frame, lower_teal, upper_teal)
        return teal_mask

    def locate_button(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 500:
            return None
            
        x, y, w, h = cv2.boundingRect(largest)
        logger.info(f"Contour Features - Area: {area}, Size: {w}x{h}, Pos: {x},{y}")
        
        # Sanity check: Button shouldn't be the whole screen
        # Assuming 1080p, a button might be 300x80 or so.
        # If w > 1500 or h > 900, it's likely a background/false positive.
        if w > 1500 or h > 900:
            logger.warning("Contour too large - ignoring (likely background).")
            return None
            
        return (x, y, w, h) # Return Rect for OCR

    def analyze_state(self, region_of_interest=None):
        """
        Main entry point for state detection.
        Returns: (STATE_ENUM, (x, y) coordinates or None)
        """
        frame = self.capture_frame()
        if frame is None:
            return self.STATE_UNKNOWN, None
            
        # Debug: Save last frame
        cv2.imwrite(os.path.expanduser("~/eve-trader/logs/debug_frame.jpg"), frame)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. PURGE YELLOW HALLUCINATION: Only Scan for Teal
        teal_mask = self.detect_play_button(hsv)
        
        play_rect = self.locate_button(teal_mask)
        
        if play_rect:
            x, y, w, h = play_rect
            # OCR Verification - "Template Match" equivalent via Text
            # Crop the button area
            roi = frame[y:y+h, x:x+w]
            
            # Pre-process for OCR
            processed = self.enhance_for_ocr(roi)
            text = self.run_ocr(processed).upper()
            
            logger.info(f"Visual Cortex: OCR Scan of Teal Blob -> '{text}'")
            
            if "PLAY" in text or "NOW" in text or "LAUNCH" in text:
                logger.info("Visual Cortex: CONFIRMED PLAY BUTTON (Teal + Text Match)")
                return self.STATE_PLAY_READY, (x + w//2, y + h//2)
            else:
                logger.warning(f"Visual Cortex: Teal Blob rejected (Text mismatch: '{text}')")
                
        # 2. Strict Ignore of Updates (Task: Purge Yellow Hallucination)
        # We assume updates are manual or handled differently now to avoid false positives.
        
        return self.STATE_UNKNOWN, None

    def trigger_potato_mode(self):
        """
        NOTE: Direct injection has been removed.
        This method is kept for API compatibility but is now a no-op.
        """
        logger.warning("Potato Mode trigger requested, but direct injection is disabled.")
        return False

if __name__ == "__main__":
    vc = VisualCortex()
    state, coords = vc.analyze_state()
    print(f"State: {state}, Coords: {coords}")
