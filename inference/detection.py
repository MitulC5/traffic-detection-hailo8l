"""
detection.py — GStreamer-based real-time detection pipeline for Hailo-8L
Uses the Hailo GStreamer detection app framework with a custom callback
for real-time traffic object detection on Raspberry Pi 5.
This is a basic pipeline which inherits the original detection_pipeline from hailo-apps.
"""


from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import csv, datetime
from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# ------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# ------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        file_exists = Path('detections_log.csv').exists()
        self.log_file = open('detections_log.csv', mode='a', newline='')
        self.writer = csv.writer(self.log_file)
        if not file_exists:
            self.writer.writerow(["Timestamp", "Label", "Confidence", "X", "Y", "Width", "Height"])

# ------------------------------------------------------------------------------
# User-defined callback function
# ------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    string_to_print += f"Total detections: {len(detections)}\n"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        user_data.writer.writerow([
                timestamp, 
                label, 
                f"{confidence:.2f}", 
                bbox.xmin(), bbox.ymin(), bbox.width(), bbox.height()
            ])
        # Ensure it actually gets written to disk immediately
        
        string_to_print += (
            f"Detection: {label}, conf={confidence:.2f}, "
            f"bbox=({bbox.xmin():.3f},{bbox.ymin():.3f},{bbox.width():.3f},{bbox.height():.3f})\n"
        )
    user_data.log_file.flush() 
    print(string_to_print)
    return Gst.PadProbeReturn.OK

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    if env_file.exists():
        os.environ["HAILO_ENV_FILE"] = str(env_file)
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
