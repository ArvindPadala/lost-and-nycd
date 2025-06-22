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

# Initialize log file if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "object_id", "type", "frames_present", "frame_number"])


# -------------------- Tracker Class --------------------
class LostItemTracker:
    def __init__(self, iou_threshold=0.5, linger_threshold=5):
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
                        "frames_present": 1
                    }

        # Clean up old objects
        self.objects = {
            k: v for k, v in self.objects.items()
            if frame_num - v["last_seen"] <= self.linger_threshold
        }

        return {
            k: v for k, v in self.objects.items()
            if v["frames_present"] >= self.linger_threshold
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

# -------------------- Config --------------------
API_KEYS = [
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiIyZWE3MWEwNC03NTk1LTQ2MjktOTkxOC1hOGJlYmVjNzQ3YTQiLCJvcmdfaWQiOiJ0ZVdaRE9wdXFYVmwzeUNGajNDM1dKcHJEb1N6UFVXWSIsImlhdCI6MTc1MDUzMzA4OSwidmVyIjoxfQ.elvF8OGtF0XJOv2Ptzj06dw3T-K2IzFxqn64XkuVTJc",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiI0OTRlNjhlZS0yNjU2LTQ0MjEtOGMwNy04MjFkODQxZDQyNjEiLCJvcmdfaWQiOiJ0ZVdaRE9wdXFYVmwzeUNGajNDM1dKcHJEb1N6UFVXWSIsImlhdCI6MTc1MDUyNDAwNSwidmVyIjoxfQ.LfJHkVhBoaGQQQpj-En5f0s0XB-5e4yMkNt0VQCy_wA",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJhMzNjOGM2MC1kM2FmLTQ4YjgtYTBmNy1jY2M5MjczMmE2YzEiLCJvcmdfaWQiOiIwSUU5RTlEbE5GazhDOGF5VnNOUGpzZ2RkOE4wTE5qbyIsImlhdCI6MTc1MDUxNzcyNSwidmVyIjoxfQ.4D4Hez-8i0UR_V7Qgtzlvmrgqv2eG6i4pghF1XHPsmg"
]  # Replace with your real keys
rotator = MoondreamRotator(API_KEYS)
tracker = LostItemTracker(linger_threshold=30)

VIDEO_SOURCE = 0
FRAME_SKIP = 1
CAPTION_INTERVAL = 30
LOST_OBJECTS = ["backpack", "wallet", "phone", "bottle"]

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

def draw_detections(frame, detections_by_type, tracked_objects, caption=None):
    h, w, _ = frame.shape
    count = 0

    for obj_type, boxes in detections_by_type.items():
        for box in boxes:
            x1 = int(box["x_min"] * w)
            y1 = int(box["y_min"] * h)
            x2 = int(box["x_max"] * w)
            y2 = int(box["y_max"] * h)

            color = (0, 255, 255)  # Default: Yellow
            label = obj_type

            # Match detection box to tracker
            for obj_id, tracked in tracked_objects.items():
                if tracked["type"] == obj_type:
                    tracked_box = tracked["box"]
                    iou = tracker._iou([x1, y1, x2, y2], tracked_box)
                    if iou > 0.5:
                        frames_seen = tracked["frames_present"]
                        if frames_seen >= 30:
                            color = (255, 0, 0)  # Blue for potentially lost
                        else:
                            color = (0, 0, 255)  # Red for normal lingering
                        label += f" (ID: {obj_id})"
                        break

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            count += 1

    # Summary text
    cv2.putText(frame, f"Lost items detected: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Optional: Caption
    if caption:
        wrapped = caption[:100] + ("..." if len(caption) > 100 else "")
        cv2.putText(frame, wrapped, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

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
        print(f"Logged lost item: {obj_type} (ID: {obj_id}) seen for {frames_present} frames")

# -------------------- Main Loop --------------------

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    frame_count = 0
    print("Lost item detection running. Press 'q' to quit.")

    if not cap.isOpened():
        print("Error: Could not open webcam/video.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error.")
                break

            caption = None
            if frame_count % FRAME_SKIP == 0:
                pil_img = frame_to_pil(frame)
                detections = detect_all_objects(pil_img)
                tracked_objects = tracker.update(frame_count, detections, frame.shape)

                # Log potentially lost items (seen ≥ 30 frames)
                for obj_id, obj in tracked_objects.items():
                    if obj["frames_present"] >= 30:
                        print(f"Potentially lost: {obj['type']} (ID: {obj_id}) for {obj['frames_present']} frames")
                        log_lost_item(obj_id, obj["type"], obj["frames_present"], frame_count)

                # Generate caption occasionally
                if frame_count % CAPTION_INTERVAL == 0:
                    try:
                        caption_dict = rotator.model.caption(pil_img)
                        caption = caption_dict.get("caption", "")
                        print(f"Scene Story: {caption}")
                    except Exception as e:
                        print(f"Caption Error: {e}")

                if detections:
                    print(f"[FRAME {frame_count}] Detected Now: {sum(len(v) for v in detections.values())} objects")
                    print(f"Tracker currently managing {len(tracked_objects)} lingering objects: {[f'{v['type']} (ID: {k})' for k,v in tracked_objects.items()]}")

                if tracked_objects:
                    print(f"Lingering items: {[f'{v['type']} (ID: {k})' for k,v in tracked_objects.items()]}")

                frame = draw_detections(frame, detections, tracked_objects, caption)

            cv2.imshow("Lost Item Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\n Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if tracker.objects:
            print("Final lingering items before shutdown:")
            for k, v in tracker.objects.items():
                print(f"→ {v['type']} (ID: {k}), seen for {v['frames_present']} frames")

            # Optional JSON backup
            with open("final_lingering_summary.json", "w") as f:
                json.dump(tracker.objects, f, indent=2)

        # Save final frame with blue boxes if there are potentially lost items
        if 'frame' in locals() and tracker.objects:
            output_path = "final_lost_items_frame.jpg"
            cv2.imwrite(output_path, frame)
            print(f"Final frame with potentially lost items saved to {output_path}")

        print("Stream ended.")

if __name__ == "__main__":
    main()
