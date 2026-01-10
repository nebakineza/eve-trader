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


def _repo_root() -> str:
    # nexus/automaton/visual_cortex.py -> repo root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class VisualCortex:
    def __init__(self, display=":0"):
        self.display = display
        self._last_ocr_fallback_ts = 0.0
        self.capture_region = None  # (x, y, w, h) in root coords
        
        # 1. Enable OpenCL for iGPU acceleration (best-effort).
        # Note: OpenCL probing can raise on some hosts/drivers; never let that
        # disable Play Now detection.
        try:
            if cv2.ocl.haveOpenCL():
                try:
                    cv2.ocl.setUseOpenCL(True)
                    device_name = None
                    try:
                        device_name = cv2.ocl.Device.getDefault().name()
                    except Exception:
                        device_name = None

                    if device_name:
                        logger.info(f"OpenCL Acceleration: ENABLED (Device: {device_name})")
                    else:
                        logger.info("OpenCL Acceleration: ENABLED")
                except Exception as e:
                    try:
                        cv2.ocl.setUseOpenCL(False)
                    except Exception:
                        pass
                    logger.warning("OpenCL Acceleration: ERROR (%s) - Falling back to CPU", e)
            else:
                logger.info("OpenCL Acceleration: UNAVAILABLE - Falling back to CPU")
        except Exception as e:
            logger.warning("OpenCL Acceleration: PROBE ERROR (%s) - Falling back to CPU", e)
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass

        # Define Launcher States
        self.STATE_PLAY_READY = "STATE_PLAY_READY"
        self.STATE_UPDATE_REQUIRED = "STATE_UPDATE_REQUIRED"
        self.STATE_VERIFYING = "STATE_VERIFYING"
        self.STATE_UNKNOWN = "STATE_UNKNOWN"

    def set_capture_region(self, region):
        """Set capture region as (x, y, w, h) in root coordinates, or None for full screen."""
        self.capture_region = region

    def _apply_capture_offset(self, coords):
        if coords is None:
            return None
        if self.capture_region is None:
            return coords
        ox, oy, _, _ = self.capture_region
        x, y = coords
        return (int(ox) + int(x), int(oy) + int(y))

    def capture_frame(self):
        """
        Captures the X11 screen frame buffer utilizing scrot.
        Returns: numpy array (BGR)
        """
        env = os.environ.copy()
        env["DISPLAY"] = self.display
        try:
            # scrot to stdout is fast and avoids disk I/O
            cmd = ["scrot", "-z"]
            if self.capture_region is not None:
                x, y, w, h = self.capture_region
                # scrot -a x,y,w,h captures an absolute region.
                cmd += ["-a", f"{int(x)},{int(y)},{int(w)},{int(h)}"]
            cmd += ["-"]
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

    def enhance_for_ocr_fast(self, frame):
        """More robust OCR preprocessing for UI text (adaptive threshold)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        white_ratio = float(np.count_nonzero(thresh == 255)) / float(thresh.size)
        if white_ratio < 0.10 or white_ratio > 0.90:
            thresh = cv2.bitwise_not(thresh)
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
            cmd = ["tesseract", "stdin", "stdout", "--psm", "7", "-l", "eng"]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                out, _ = proc.communicate(input=encoded_img.tobytes(), timeout=6)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                logger.warning("OCR timed out")
                return ""
            return out.decode().strip()
        except FileNotFoundError:
            logger.error("Tesseract not found. Install tesseract-ocr.")
            return ""
        except Exception as e:
            logger.error(f"OCR Error: {e}")
            return ""

    def run_ocr_tsv(self, img_region, *, timeout_seconds: int = 6, psm: int = 7):
        """Run Tesseract OCR and return TSV rows for word-level bounding boxes."""
        try:
            success, encoded_img = cv2.imencode('.png', img_region)
            if not success:
                return []

            # TSV gives per-word bounding boxes.
            cmd = ["tesseract", "stdin", "stdout", "--psm", str(int(psm)), "-l", "eng", "tsv"]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:
                out, _ = proc.communicate(input=encoded_img.tobytes(), timeout=int(timeout_seconds))
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except Exception:
                    pass
                logger.warning("OCR TSV timed out after %ss", timeout_seconds)
                return []
            text = out.decode("utf-8", errors="ignore")
            lines = [l for l in text.splitlines() if l.strip()]
            if len(lines) <= 1:
                return []

            # Header:
            # level page_num block_num par_num line_num word_num left top width height conf text
            rows = []
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) < 12:
                    continue
                try:
                    level = int(parts[0])
                    block_num = int(parts[2])
                    par_num = int(parts[3])
                    line_num = int(parts[4])
                    word_num = int(parts[5])
                    left = int(parts[6])
                    top = int(parts[7])
                    width = int(parts[8])
                    height = int(parts[9])
                    conf = float(parts[10])
                    word = parts[11].strip()
                except Exception:
                    continue

                if not word:
                    continue

                rows.append(
                    {
                        "level": level,
                        "block": block_num,
                        "par": par_num,
                        "line": line_num,
                        "word_num": word_num,
                        "left": left,
                        "top": top,
                        "width": width,
                        "height": height,
                        "conf": conf,
                        "text": word,
                    }
                )
            return rows
        except FileNotFoundError:
            logger.error("Tesseract not found. Install tesseract-ocr.")
            return []
        except Exception as e:
            logger.error(f"OCR TSV Error: {e}")
            return []

    def detect_play_button(self, hsv_frame):
        """
        Detects the Teal 'PLAY NOW' button with strict color filtering.
        """
        # Teal/Cyan UI elements in the launcher can vary with theme/lighting.
        # Use a broader hue range and let contour heuristics + ROI do the heavy lifting.
        # OpenCV hue range is [0..179].
        lower_teal = np.array([60, 80, 80])
        upper_teal = np.array([115, 255, 255])

        teal_mask = cv2.inRange(hsv_frame, lower_teal, upper_teal)

        # Reduce noise and avoid massive connected regions.
        kernel = np.ones((5, 5), np.uint8)
        teal_mask = cv2.morphologyEx(teal_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        teal_mask = cv2.morphologyEx(teal_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return teal_mask

    def locate_button(self, mask, *, debug: bool = False):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        mask_h, mask_w = mask.shape[:2]
        candidates = []

        # Heuristics tuned for a wide button in a cropped ROI.
        # Keep thresholds permissive and let OCR confirm the label.
        min_area = max(400, int((mask_w * mask_h) * 0.0005))
        max_area = int((mask_w * mask_h) * 0.35)
        min_w = max(60, int(mask_w * 0.05))
        max_w = int(mask_w * 0.95)
        min_h = max(18, int(mask_h * 0.04))
        max_h = max(60, int(mask_h * 0.40))

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < min_w or w > max_w or h < min_h or h > max_h:
                continue

            aspect = (w / float(h)) if h else 0.0
            if aspect < 2.0 or aspect > 12.0:
                continue

            candidates.append((area, x, y, w, h))

        if not candidates:
            if debug:
                # Emit a quick hint about what was found so we can tune thresholds.
                def _desc(c):
                    a = float(cv2.contourArea(c))
                    x, y, w, h = cv2.boundingRect(c)
                    aspect = (w / float(h)) if h else 0.0
                    return a, x, y, w, h, aspect

                ranked = sorted((_desc(c) for c in contours), key=lambda t: t[0], reverse=True)
                logger.info(
                    "Debug: %d contours (mask=%dx%d). Top=%s",
                    len(contours),
                    mask_w,
                    mask_h,
                    "; ".join(
                        f"area={a:.0f} rect={w}x{h}@{x},{y} aspect={aspect:.2f}"
                        for a, x, y, w, h, aspect in ranked[:5]
                    ),
                )
            return None

        area, x, y, w, h = max(candidates, key=lambda t: t[0])
        logger.info(f"Contour Features - Area: {area}, Size: {w}x{h}, Pos: {x},{y}")
        return (x, y, w, h)  # Return Rect for OCR

    def analyze_state(self, region_of_interest=None):
        """
        Main entry point for state detection.
        Returns: (STATE_ENUM, (x, y) coordinates or None)
        """
        frame = self.capture_frame()
        if frame is None:
            return self.STATE_UNKNOWN, None

        debug_enabled = os.getenv("VISUAL_CORTEX_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        debug_dir = os.getenv("VISUAL_CORTEX_DEBUG_DIR", os.path.join(_repo_root(), "logs"))
        if debug_enabled:
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except Exception:
                debug_enabled = False

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Default ROI: lower-center portion of the launcher, where the Play button
        # typically resides. This avoids selecting full-screen teal-like backgrounds.
        fh, fw = frame.shape[:2]
        if region_of_interest is None:
            # For the EVE launcher, PLAY NOW is typically center-ish, slightly right.
            x1 = int(fw * 0.40)
            x2 = int(fw * 0.98)
            y1 = int(fh * 0.18)
            y2 = int(fh * 0.78)
        else:
            x1, y1, x2, y2 = region_of_interest
            x1 = int(max(0, min(fw - 1, x1)))
            x2 = int(max(0, min(fw, x2)))
            y1 = int(max(0, min(fh - 1, y1)))
            y2 = int(max(0, min(fh, y2)))
            if x2 <= x1 or y2 <= y1:
                return self.STATE_UNKNOWN, None

        hsv_roi = hsv[y1:y2, x1:x2]

        roi_color = frame[y1:y2, x1:x2]
        rh, rw = roi_color.shape[:2]

        # OCR window inside ROI: center-bottom, where the Play button label typically sits.
        ocr_x1 = int(rw * 0.15)
        ocr_x2 = int(rw * 0.85)
        ocr_y1 = int(rh * 0.15)
        ocr_y2 = int(rh * 0.98)
        roi_ocr = roi_color[ocr_y1:ocr_y2, ocr_x1:ocr_x2]

        # 1. PURGE YELLOW HALLUCINATION: Only Scan for Teal
        teal_mask = self.detect_play_button(hsv_roi)

        if debug_enabled:
            try:
                cv2.imwrite(os.path.join(debug_dir, "visual_cortex_frame.jpg"), frame)
                cv2.imwrite(os.path.join(debug_dir, "visual_cortex_roi.jpg"), roi_color)
                cv2.imwrite(os.path.join(debug_dir, "visual_cortex_teal_mask.png"), teal_mask)
                cv2.imwrite(os.path.join(debug_dir, "visual_cortex_ocr_window.jpg"), roi_ocr)
                nonzero = int(cv2.countNonZero(teal_mask))
                total = int(teal_mask.shape[0] * teal_mask.shape[1])
                if self.capture_region is not None:
                    ox, oy, ow, oh = self.capture_region
                    logger.info(
                        "Debug: capture_region=%d,%d %dx%d ROI=(%d,%d)-(%d,%d) teal_mask_nonzero=%d/%d",
                        ox,
                        oy,
                        ow,
                        oh,
                        x1,
                        y1,
                        x2,
                        y2,
                        nonzero,
                        total,
                    )
                else:
                    logger.info("Debug: ROI=(%d,%d)-(%d,%d) teal_mask_nonzero=%d/%d", x1, y1, x2, y2, nonzero, total)
            except Exception as e:
                logger.warning("Debug image write failed: %s", e)

        play_rect = self.locate_button(teal_mask, debug=debug_enabled)
        
        if play_rect:
            x, y, w, h = play_rect
            # Convert ROI-local coordinates back to full-frame.
            x = x + x1
            y = y + y1
            # OCR Verification - "Template Match" equivalent via Text
            # Crop the button area
            roi = frame[y:y+h, x:x+w]
            
            # Pre-process for OCR
            processed = self.enhance_for_ocr(roi)
            text = self.run_ocr(processed).upper()
            
            logger.info(f"Visual Cortex: OCR Scan of Teal Blob -> '{text}'")
            
            if "PLAY" in text or "NOW" in text or "LAUNCH" in text:
                logger.info("Visual Cortex: CONFIRMED PLAY BUTTON (Teal + Text Match)")
                return self.STATE_PLAY_READY, self._apply_capture_offset((x + w//2, y + h//2))
            else:
                logger.warning(f"Visual Cortex: Teal Blob rejected (Text mismatch: '{text}')")

            # If OCR is flaky/slow, fall back to a strict fill+shape heuristic.
            # This is safe-ish because we already restricted to the expected ROI.
            try:
                rx = int(x - x1)
                ry = int(y - y1)
                sub = teal_mask[ry : ry + h, rx : rx + w]
                if sub.size:
                    fill = float(cv2.countNonZero(sub)) / float(w * h)
                    if debug_enabled:
                        logger.info("Debug: teal_candidate fill_ratio=%.2f", fill)
                    if (w >= 160 and h >= 35 and (w / float(h)) >= 2.5 and fill >= 0.35):
                        logger.info("Visual Cortex: CONFIRMED PLAY BUTTON (Teal Fill Match)")
                        return self.STATE_PLAY_READY, self._apply_capture_offset((x + w // 2, y + h // 2))
            except Exception:
                pass

        # 1a. Color-agnostic fallback: find any saturated/bright "button-like" rectangles
        # in the ROI, then OCR-confirm that the rectangle itself contains PLAY/LAUNCH.
        try:
            sat = hsv_roi[:, :, 1]
            val = hsv_roi[:, :, 2]
            generic_mask = ((sat >= 170) & (val >= 170)).astype(np.uint8) * 255
            kernel = np.ones((5, 5), np.uint8)
            generic_mask = cv2.morphologyEx(generic_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            generic_mask = cv2.morphologyEx(generic_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(generic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                roi_h, roi_w = hsv_roi.shape[:2]
                rects = []
                for c in contours:
                    area = float(cv2.contourArea(c))
                    if area < 600 or area > (roi_w * roi_h * 0.25):
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    if w < 80 or h < 22 or w > int(roi_w * 0.98) or h > int(roi_h * 0.60):
                        continue
                    aspect = (w / float(h)) if h else 0.0
                    if aspect < 2.0 or aspect > 14.0:
                        continue
                    rects.append((area, x, y, w, h))

                if debug_enabled:
                    logger.info("Debug: generic_mask contours=%d rect_candidates=%d", len(contours), len(rects))

                for area, x, y, w, h in sorted(rects, key=lambda t: t[0], reverse=True)[:5]:
                    # Convert ROI-local to full-frame coords
                    fx = x1 + x
                    fy = y1 + y
                    crop = frame[fy : fy + h, fx : fx + w]
                    if crop.size == 0:
                        continue
                    processed = self.enhance_for_ocr(crop)
                    text = self.run_ocr(processed).upper()
                    if debug_enabled:
                        logger.info("Debug: generic_button_candidate area=%.0f rect=%dx%d@%d,%d ocr='%s'", area, w, h, fx, fy, text)
                    if "PLAY" in text or "LAUNCH" in text:
                        logger.info("Visual Cortex: CONFIRMED PLAY BUTTON (Generic Color + Text Match)")
                        return self.STATE_PLAY_READY, self._apply_capture_offset((fx + w // 2, fy + h // 2))
        except Exception as e:
            if debug_enabled:
                logger.warning("Debug: generic button scan failed: %s", e)

        # 1b. OCR fallback: search the ROI for the words PLAY/NOW and click the word center.
        # This helps when the launcher theme/button color isn't reliably teal.
        now_ts = time.time()
        if now_ts - float(self._last_ocr_fallback_ts) < 10.0:
            return self.STATE_UNKNOWN, None
        self._last_ocr_fallback_ts = now_ts

        orh, orw = roi_ocr.shape[:2]

        # Downscale for OCR speed; map boxes back to full-res coordinates.
        scale = 0.5 if max(orw, orh) >= 450 else 1.0
        if scale != 1.0:
            roi_small = cv2.resize(roi_ocr, (int(orw * scale), int(orh * scale)), interpolation=cv2.INTER_AREA)
        else:
            roi_small = roi_ocr

        processed_roi = cv2.cvtColor(roi_small, cv2.COLOR_BGR2GRAY)
        processed_roi = cv2.normalize(processed_roi, None, 0, 255, cv2.NORM_MINMAX)
        if debug_enabled:
            try:
                cv2.imwrite(os.path.join(debug_dir, "visual_cortex_ocr_processed.png"), processed_roi)
            except Exception:
                pass
        rows = self.run_ocr_tsv(processed_roi, timeout_seconds=6, psm=6)
        if rows:
            if debug_enabled:
                preview = [
                    f"{r.get('text','').strip()}({r.get('conf', -1):.0f})"
                    for r in rows
                    if r.get("text")
                ][:12]
                logger.info("Debug: OCR TSV rows=%d preview=%s", len(rows), " ".join(preview))
            # Prefer high-confidence words on the same line: PLAY + NOW.
            norm = [
                {
                    **r,
                    "u": r["text"].strip().upper(),
                }
                for r in rows
                if r.get("conf", -1) >= 50
            ]

            plays = [r for r in norm if r["u"] in {"PLAY", "LAUNCH"} or "PLAY" in r["u"]]
            nows = [r for r in norm if r["u"] == "NOW" or "NOW" in r["u"]]

            def center(r):
                return (r["left"] + r["width"] // 2, r["top"] + r["height"] // 2)

            # Try strict match: PLAY and NOW on same (block,par,line)
            for p in plays:
                for n in nows:
                    if (p["block"], p["par"], p["line"]) == (n["block"], n["par"], n["line"]):
                        cx, cy = center(p)
                        if scale != 1.0:
                            cx = int(cx / scale)
                            cy = int(cy / scale)
                        logger.info("Visual Cortex: CONFIRMED PLAY via OCR (PLAY+NOW)")
                        return self.STATE_PLAY_READY, self._apply_capture_offset((x1 + ocr_x1 + cx, y1 + ocr_y1 + cy))

            # Looser: PLAY alone.
            if plays:
                best = max(plays, key=lambda r: (r.get("conf", 0.0), r.get("width", 0)))
                cx, cy = center(best)
                if scale != 1.0:
                    cx = int(cx / scale)
                    cy = int(cy / scale)
                logger.info("Visual Cortex: CONFIRMED PLAY via OCR (PLAY)")
                return self.STATE_PLAY_READY, self._apply_capture_offset((x1 + ocr_x1 + cx, y1 + ocr_y1 + cy))
                
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
