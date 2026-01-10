import time
import logging
import os
import glob
from .hardware_bridge import ArduinoHID
from .visual_cortex import VisualCortex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Supervisor")

class AutomatonSupervisor:
    def __init__(self, watch_dir="/tmp/shots"):
        self.watch_dir = watch_dir
        self.hid = ArduinoHID()
        self.vision = VisualCortex()

    def get_latest_screenshot(self):
        files = glob.glob(os.path.join(self.watch_dir, "*.jpg"))
        if not files: return None
        return max(files, key=os.path.getctime)

    def run_cycle(self):
        shot_path = self.get_latest_screenshot()
        if not shot_path:
            logger.debug("No visual input")
            return

        image = self.vision.load_image(shot_path)
        if image is None: return

        # 1. Check health
        if self.vision.is_screen_white(image):
            logger.warning("Visual Anomaly: White Screen detected.")
            # Action: Maybe ESC? Or wait?
            return

        # 2. Login Logic (Example)
        # Check for "Connect" button
        # btn_loc = self.vision.find_template(image, "templates/connect_btn.png")
        # if btn_loc:
        #     logger.info(f"Found Connect Button at {btn_loc}")
        #     self.hid.move_mouse(*btn_loc)
        #     self.hid.click()

        # 3. OCR Checks
        # text = self.vision.detect_text(image, roi=(600, 400, 200, 50))
        # if "Character Information" in text:
        #    pass

    def loop(self):
        logger.info("Starting Automaton Supervisor Loop")
        while True:
            self.run_cycle()
            time.sleep(1)

if __name__ == "__main__":
    supervisor = AutomatonSupervisor()
    supervisor.loop()
