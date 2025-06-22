import cv2
import moondream as md
from PIL import Image
import uuid
import time
import json
import os
import csv
from datetime import datetime

#---------------csv Output----------------------
LOG_FILE = "lost_items_log.csv"
logged_ids = set()
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "object_id", "type", "frames_present", "frame_number"])

# -------------------- Tracker Class --------------------
class LostItemTracker:
    def __init__(self, iou_threshold=0.5, linger_threshold=30):
        self.objects = {}
        self.iou_threshold = iou_threshold
        self.linger_threshold = linger_threshold

    def _iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        if inter_area == 0:
            return 0
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter_area / float(box1_area + box2_area - inter_area)

    def update(self, frame_num, detections_by_type, frame_shape):
        h, w = frame_shape[:2]
        for obj_type, boxes in detections_by_type.items():
            for box in boxes:
                x1 = int(box["x_min"] * w)
                y1 = int(box["y_min"] * h)
                x2 = int(box["x_max"] * w)
                y2 = int(box["y_max"] * h)
                new_box = [x1, y1, x2, y2]

                matched = False
                for obj_id, obj in self.objects.items():
                    if obj["type"] == obj_type:
                        iou = self._iou(obj["box"], new_box)
                        if iou > self.iou_threshold:
                            obj["box"] = new_box
                            obj["last_seen"] = frame_num
                            obj["frames_present"] += 1
                            matched = True
                            break

                if not matched:
                    new_id = str(uuid.uuid4())[:8]
                    self.objects[new_id] = {
                        "type": obj_type,
                        "box": new_box,
                        "last_seen": frame_num,
                        "frames_present": 1,
                        "prompt_pending": False,
                        "is_lost": False
                    }

        # Clean up old objects
        self.objects = {
            k: v for k, v in self.objects.items()
            if frame_num - v["last_seen"] <= self.linger_threshold
        }

        # Set prompt flag if necessary
        for obj in self.objects.values():
            if obj["frames_present"] == self.linger_threshold and not obj["is_lost"]:
                obj["prompt_pending"] = True

        return {
            k: v for k, v in self.objects.items()
            if v["frames_present"] >= 5
        }

# -------------------- Moondream API --------------------
class MoondreamRotator:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.index = 0
        self.model = self._create_model()

    def _create_model(self):
        return md.vl(api_key=self.api_keys[self.index])

    def next_model(self):
        self.index = (self.index + 1) % len(self.api_keys)
        print(f"Switching to API Key #{self.index + 1}")
        self.model = self._create_model()

    def detect(self, image, obj):
        try:
            return self.model.detect(image, obj)["objects"]
        except Exception as e:
            if "429" in str(e):
                print(f"[ERROR] 429 Too Many Requests for '{obj}'")
                self.next_model()
                return self.detect(image, obj)
            print(f"[ERROR] Detecting '{obj}': {e}")
            return []

    def ask(self, image, prompt):
        return self.model.ask(image, prompt)

# -------------------- Config --------------------
API_KEYS = [
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIyZWE3MWEwNC03NTk1LTQ2MjktOTkxOC1hOGJlYmVjNzQ3YTQiLCJvcmdfaWQiOiJ0ZVdaRE9wdXFYVmwzeUNGajNDM1dKcHJEb1N6UFVXWSIsImlhdCI6MTc1MDUzMzA4OSwidmVyIjoxfQ.elvF8OGtF0XJOv2Ptzj06dw3T-K2IzFxqn64XkuVTJc",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI0OTRlNjhlZS0yNjU2LTQ0MjEtOGMwNy04MjFkODQxZDQyNjEiLCJvcmdfaWQiOiJ0ZVdaRE9wdXFYVmwzeUNGajNDM1dKcHJEb1N6UFVXWSIsImlhdCI6MTc1MDUyNDAwNSwidmVyIjoxfQ.LfJHkVhBoaGQQQpj-En5f0s0XB-5e4yMkNt0VQCy_wA",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJhMzNjOGM2MC1kM2FmLTQ4YjgtYTBmNy1jY2M5MjczMmE2YzEiLCJvcmdfaWQiOiIwSUU5RTlEbE5GazhDOGF5VnNOUGpzZ2RkOE4wTE5qbyIsImlhdCI6MTc1MDUxNzcyNSwidmVyIjoxfQ.4D4Hez-8i0UR_V7Qgtzlvmrgqv2eG6i4pghF1XHPsmg"

]
VIDEO_SOURCE = "your_input_video.mp4"
OUTPUT_VIDEO = "output_with_lost_items.mp4"
FRAME_SKIP = 1
CAPTION_INTERVAL = 30
LOST_OBJECTS = ["backpack"]
rotator = MoondreamRotator(API_KEYS)
tracker = LostItemTracker()

# -------------------- Utility Functions --------------------
def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def detect_all_objects(image):
    detections = {}
    for obj_type in LOST_OBJECTS:
        results = rotator.detect(image, obj_type)
        if results:
            detections[obj_type] = results
    return detections

def draw_detections(frame, detections_by_type, tracked_objects):
    h, w, _ = frame.shape
    for obj_type, boxes in detections_by_type.items():
        for box in boxes:
            x1 = int(box["x_min"] * w)
            y1 = int(box["y_min"] * h)
            x2 = int(box["x_max"] * w)
            y2 = int(box["y_max"] * h)
            color = (0, 255, 255)
            label = obj_type
            for obj_id, tracked in tracked_objects.items():
                if tracked["type"] == obj_type:
                    tracked_box = tracked["box"]
                    if tracker._iou([x1, y1, x2, y2], tracked_box) > 0.5:
                        if tracked.get("is_lost"):
                            color = (0, 0, 255)
                            label += f" (LOST - ID:{obj_id})"
                        else:
                            label += f" (ID:{obj_id})"
                        break
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

def log_lost_item(obj_id, obj_type, frames_present, frame_number):
    if obj_id not in logged_ids:
        logged_ids.add(obj_id)
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                obj_id,
                obj_type,
                frames_present,
                frame_number
            ])

# -------------------- Main --------------------
def main():
    cap = cv2.VideoCapture('/content/test2.mp4')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            pil_img = frame_to_pil(frame)
            detections = detect_all_objects(pil_img)
            tracked_objects = tracker.update(frame_count, detections, frame.shape)

            for obj_id, obj in tracked_objects.items():
                if obj.get("prompt_pending"):
                    prompt = f"Is the {obj['type']} which is labelled in the image held by someone? Answer in one word."
                    try:
                        response = rotator.ask(pil_img, prompt)
                        answer = response.get("answer", "").strip().lower()
                        if answer == "no":
                            obj["is_lost"] = True
                            log_lost_item(obj_id, obj["type"], obj["frames_present"], frame_count)
                        else:
                            obj["frames_present"] = 0
                    except Exception as e:
                        print(f"Prompt Error: {e}")
                    obj["prompt_pending"] = False

            frame = draw_detections(frame, detections, tracked_objects)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    print("✅ Processing complete. Output saved to", OUTPUT_VIDEO)

