
import time
import cv2
import torch
import numpy as np

from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection
)

# ---------------- SETTINGS ----------------

MODEL_ID = "IDEA-Research/grounding-dino-tiny"

CAMERA_INDEX = 0
DETECT_EVERY_N_FRAMES = 5

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

# Add or remove descriptions to suit your project.
PROMPT = (
    "person. mobile phone. laptop. keyboard. mouse. "
    "water bottle. cup. book. pen. bag. chair. "
    "table. remote control. headphones. shoe. "
    "toy. fruit. electronic device. unknown object."
)

# ------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Loading Grounding DINO model...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    MODEL_ID
).to(device)
model.eval()

print("Model loaded!")

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(1)

if not camera.isOpened():
    raise RuntimeError(
        "Webcam unavailable. Check the camera connection and permissions."
    )

# Current target and detection results
detections = []
target_box = None
target_label = None
target_score = 0.0

frame_count = 0
last_detection_time = 0.0
fps = 0.0


def iou(box_a, box_b):
    """Intersection over Union between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def detect_objects(frame):
    """Run open-vocabulary detection on one webcam frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    inputs = processor(
        images=pil_image,
        text=PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs)

    result = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(frame.shape[0], frame.shape[1])]
    )[0]

    found = []

    for box, score, label in zip(
        result["boxes"],
        result["scores"],
        result["text_labels"]
    ):
        x1, y1, x2, y2 = map(int, box.tolist())

        x1 = max(0, min(x1, frame.shape[1] - 1))
        x2 = max(0, min(x2, frame.shape[1] - 1))
        y1 = max(0, min(y1, frame.shape[0] - 1))
        y2 = max(0, min(y2, frame.shape[0] - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        found.append({
            "box": (x1, y1, x2, y2),
            "label": str(label),
            "score": float(score)
        })

    # Highest-confidence candidate first
    found.sort(key=lambda obj: obj["score"], reverse=True)
    return found


try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Could not read webcam frame.")
            break

        frame_count += 1
        now = time.time()

        # Run the expensive AI model every few frames
        if frame_count % DETECT_EVERY_N_FRAMES == 1:
            try:
                detections = detect_objects(frame)
                last_detection_time = now

                # Automatically choose one object initially.
                if target_box is None and detections:
                    selected = detections[0]
                    target_box = selected["box"]
                    target_label = selected["label"]
                    target_score = selected["score"]

                # Keep following the closest matching detection.
                elif target_box is not None and detections:
                    best = max(
                        detections,
                        key=lambda obj: iou(
                            target_box, obj["box"]
                        )
                    )

                    if iou(target_box, best["box"]) >= 0.20:
                        target_box = best["box"]
                        target_label = best["label"]
                        target_score = best["score"]

                    else:
                        # Previous target disappeared.
                        # Automatically select the strongest detection.
                        selected = detections[0]
                        target_box = selected["box"]
                        target_label = selected["label"]
                        target_score = selected["score"]

                elif not detections:
                    target_box = None
                    target_label = None
                    target_score = 0.0

            except Exception as exc:
                print("Detection error:", exc)

        # Draw only the focused object
        if target_box is not None:
            x1, y1, x2, y2 = target_box

            cv2.rectangle(
                frame, (x1, y1), (x2, y2),
                (0, 255, 0), 3
            )

            label = (
                f"{target_label}: {target_score:.2f}"
                if target_label
                else "Unknown Object"
            )

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No confident object detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Status information
        cv2.putText(
            frame,
            f"Objects found: {len(detections)}",
            (15, frame.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "N: Next object | R: Reset focus | ESC: Exit",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Display approximate processing FPS
        fps = 1.0 / max(time.time() - now, 0.001)

        cv2.imshow("AI Open-Vocabulary Object Focus", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key in (ord("n"), ord("N")) and detections:
            # Cycle to the next detected object
            if target_box is not None:
                current = max(
                    range(len(detections)),
                    key=lambda i: iou(
                        target_box, detections[i]["box"]
                    )
                )
                next_index = (current + 1) % len(detections)
            else:
                next_index = 0

            selected = detections[next_index]
            target_box = selected["box"]
            target_label = selected["label"]
            target_score = selected["score"]

        elif key in (ord("r"), ord("R")):
            target_box = None
            target_label = None
            target_score = 0.0

finally:
    camera.release()
    cv2.destroyAllWindows()
    print("Application closed.")
