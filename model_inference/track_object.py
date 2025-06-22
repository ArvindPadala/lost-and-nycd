import cv2
import moondream as md
from PIL import Image
import json
import numpy as np
from sort import sort  

# === CONFIG ===
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJrZXlfaWQiOiJhMzNjOGM2MC1kM2FmLTQ4YjgtYTBmNy1jY2M5MjczMmE2YzEiLCJvcmdfaWQiOiIwSUU5RTlEbE5GazhDOGF5VnNOUGpzZ2RkOE4wTE5qbyIsImlhdCI6MTc1MDUxNzcyNSwidmVyIjoxfQ.4D4Hez-8i0UR_V7Qgtzlvmrgqv2eG6i4pghF1XHPsmg"
VIDEO_SOURCE = "/content/test2.mp4"
FRAME_SKIP = 3
#LOST_OBJECTS = ["backpack", "wallet", "phone", "glove", "umbrella", "bottle"]
LOST_OBJECTS = ["backpack"]
OUTPUT_VIDEO = "tracked_output.mp4"
OUTPUT_LOG = "tracked_detections.json"

# Initialize Moondream
model = md.vl(api_key=API_KEY)

# Initialize SORT tracker
tracker = sort()

def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def detect_all_objects(pil_img, width, height):
    all_dets = []
    for obj_type in LOST_OBJECTS:
        try:
            detections = model.detect(pil_img, obj_type).get("objects", [])
            for det in detections:
                x1 = det["x_min"] * width
                y1 = det["y_min"] * height
                x2 = det["x_max"] * width
                y2 = det["y_max"] * height
                all_dets.append({
                    "label": obj_type,
                    "bbox": [x1, y1, x2, y2]
                })
        except Exception as e:
            print(f"[ERROR] {obj_type}: {e}")
    return all_dets

def draw_tracked(frame, tracked_objs):
    for obj in tracked_objs:
        x1, y1, x2, y2, obj_id = map(int, obj[:5])
        label = f"ID {obj_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return frame

def match_labels(tracked_objs, detections):
    assignments = []
    for trk in tracked_objs:
        tx1, ty1, tx2, ty2, obj_id = trk
        best_label = None
        best_iou = 0
        for det in detections:
            dx1, dy1, dx2, dy2 = det["bbox"]
            iou = compute_iou((tx1, ty1, tx2, ty2), (dx1, dy1, dx2, dy2))
            if iou > best_iou:
                best_iou = iou
                best_label = det["label"]
        assignments.append({
            "id": int(obj_id),
            "label": best_label,
            "bbox": {
                "x_min": tx1, "y_min": ty1,
                "x_max": tx2, "y_max": ty2
            }
        })
    return assignments

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    logs = []
    frame_count = 0

    if not cap.isOpened():
        print("❌ Cannot open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps / FRAME_SKIP, (width, height))

    print("🔍 Tracking started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            pil_img = frame_to_pil(frame)
            detections = detect_all_objects(pil_img, width, height)

            # Prepare for SORT: [x1, y1, x2, y2, confidence]
            sort_input = np.array([det["bbox"] + [1.0] for det in detections])
            tracked = tracker.update(sort_input)

            # Draw on frame
            frame = draw_tracked(frame, tracked)

            # Match labels to tracked boxes
            matched = match_labels(tracked, detections)

            logs.append({
                "frame": frame_count,
                "timestamp_sec": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                "tracked_objects": matched
            })

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(OUTPUT_LOG, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"✅ Video saved: {OUTPUT_VIDEO}")
    print(f"📄 Log saved: {OUTPUT_LOG}")