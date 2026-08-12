import os
import streamlit as st
from ultralytics import YOLO


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best.pt"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


# ============================================================
# OBJECT DETECTION
# ============================================================

def detect_objects(image):

    if image is None:
        raise ValueError("Image is empty.")

    yolo_model = get_model()

    results = yolo_model.predict(
        source=image,
        conf=0.25,
        iou=0.45,
        verbose=False
    )

    detections = []

    if results:

        result = results[0]

        if result.boxes is not None:

            for box in result.boxes:

                try:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    class_name = yolo_model.names.get(
                        class_id,
                        f"class_{class_id}"
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    detections.append({
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
                    })

                except Exception as error:

                    print(
                        "YOLO detection error:",
                        error
                    )

        annotated_image = result.plot()

    else:

        annotated_image = image.copy()

    return detections, annotated_image
