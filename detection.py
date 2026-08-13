from ultralytics import YOLO
from PIL import Image, ImageDraw
import os


# =========================================================
# YOLO MODEL PATH
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "best.pt"
)


# =========================================================
# LOAD MODEL
# =========================================================

print("Loading YOLO model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# =========================================================
# OBJECT DETECTION
# =========================================================

def detect_objects(image, confidence=0.10):

    # Make sure input is a PIL RGB image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    # Run YOLO
    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    detections = []

    # Copy image for drawing
    annotated_image = image.copy()

    draw = ImageDraw.Draw(annotated_image)

    # =====================================================
    # PROCESS RESULTS
    # =====================================================

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            conf = float(
                box.conf[0].item()
            )

            coordinates = box.xyxy[0].tolist()

            x1, y1, x2, y2 = map(
                int,
                coordinates
            )

            # Get class name
            class_name = model.names[class_id]

            # Store detection
            detections.append(
                {
                    "class": class_name,
                    "confidence": round(conf, 2),
                    "box": [x1, y1, x2, y2]
                }
            )

            # Draw bounding box
            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=4
            )

            # Draw label
            label = f"{class_name} {conf:.2f}"

            draw.text(
                (x1, max(0, y1 - 20)),
                label,
                fill="red"
            )

    return detections, annotated_image
