from ultralytics import YOLO
import cv2
import os

# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO model...")

# best.pt is in the same folder as detection.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO model not found: {MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully.")


# ============================================================
# OBJECT DETECTION
# ============================================================

def detect_objects(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    results = model(
        image_path,
        conf=0.25,
        verbose=False
    )

    detections = []

    # --------------------------------------------------------
    # Read detections
    # --------------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            try:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                class_name = model.names.get(
                    class_id,
                    f"class_{class_id}"
                )

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                detections.append(
                    {
                        "class": str(class_name),
                        "confidence": round(
                            confidence,
                            4
                        ),
                        "bbox": [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                    }
                )

            except Exception as error:

                print(
                    "YOLO detection error:",
                    error
                )

    # --------------------------------------------------------
    # Annotated image
    # --------------------------------------------------------

    if results:

        annotated_image = results[0].plot()

    else:

        image = cv2.imread(image_path)
        annotated_image = image

    return (
        detections,
        annotated_image
    )
