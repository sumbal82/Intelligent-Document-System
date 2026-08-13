from ultralytics import YOLO
from PIL import Image, ImageDraw
import os


# =========================================================
# YOLO MODEL PATH
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best.pt"
)


# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading YOLO model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# =========================================================
# OBJECT DETECTION FUNCTION
# =========================================================

def detect_objects(image, confidence=0.10):

    # -----------------------------------------------------
    # Convert input to PIL Image
    # -----------------------------------------------------

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    # -----------------------------------------------------
    # Run YOLO detection
    # -----------------------------------------------------

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False
    )

    # -----------------------------------------------------
    # Prepare output
    # -----------------------------------------------------

    detections = []
    annotated_image = image.copy()

    draw = ImageDraw.Draw(annotated_image)

    # -----------------------------------------------------
    # Process YOLO results
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            # Class ID
            class_id = int(box.cls[0].item())

            # Confidence
            conf = float(box.conf[0].item())

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Class name
            class_name = model.names[class_id]

            # -------------------------------------------------
            # Save detection
            # -------------------------------------------------

            detections.append({
                "class": class_name,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2]
            })

            # -------------------------------------------------
            # Draw bounding box
            # -------------------------------------------------

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=4
            )

            # -------------------------------------------------
            # Draw label
            # -------------------------------------------------

            label = f"{class_name} {conf:.2f}"

            draw.text(
                (x1, max(0, y1 - 20)),
                label,
                fill="red"
            )

    # -----------------------------------------------------
    # Return detections + annotated image
    # -----------------------------------------------------

    return detections, annotated_image
