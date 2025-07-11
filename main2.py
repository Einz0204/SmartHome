# face_runner.py

import cv2
import time
from facefunction import load_facedata, process_frame, save_log, save_unknown_image

class FaceRecognizer:
    def __init__(self):
        print(" Loading face data...")
        self.known_encodings, self.known_names = load_facedata()
        self.state = {
            "elapsed": 0,
            "last_time": time.time(),
            "last_name": "No face",
            "recognized": False
        }
        self.last_frame = None
        self.current_name = "No face"
        self.cap = cv2.VideoCapture(0)
        self.finished = False

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.finished = True
            return None, "Camera error"

        self.last_frame = frame.copy()
        processed_frame, self.current_name = process_frame(
            frame, self.known_encodings, self.known_names, self.state)

        # 判斷是否成功辨識
        if self.state["recognized"] and self.current_name not in ["Unknown", "No face"] and self.state["elapsed"] >= 3:
            print(" Face recognized: ", self.current_name)
            save_log(self.current_name, "Success")
            self.finished = True

        # Timeout 未知臉
        elif self.current_name == "Unknown" and self.state["elapsed"] >= 3:
            print(" Unknown face timeout.")
            save_log("Unknown", "Unknown")
            if self.last_frame is not None:
                save_unknown_image(self.last_frame)
            self.finished = True

        return processed_frame, self.current_name

    def release(self):
        self.cap.release()
