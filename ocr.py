import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image


@st.cache_resource
def load_ocr():
    return easyocr.Reader(
        ['en'],
        gpu=False,
        verbose=False
    )


reader = load_ocr()


def prepare_image(image):

    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid image format.")

    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
    else:
        image_bgr = image

    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    height, width = gray.shape

    if width < 1200:
        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )

    return enhanced


def extract_text(image):

    if isinstance(image, Image.Image):
        original = np.array(
            image.convert("RGB")
        )

    elif isinstance(image, np.ndarray):
        original = image

    else:
        raise ValueError(
            "Unsupported image format."
        )

    enhanced = prepare_image(original)

    original_results = reader.readtext(
        original,
        detail=1,
        paragraph=False,
        text_threshold=0.6,
        low_text=0.3,
        link_threshold=0.4,
        mag_ratio=1.2
    )

    enhanced_results = reader.readtext(
        enhanced,
        detail=1,
        paragraph=False,
        text_threshold=0.6,
        low_text=0.3,
        link_threshold=0.4,
        mag_ratio=1.2
    )

    texts = []

    for result in original_results + enhanced_results:

        if len(result) < 3:
            continue

        detected_text = str(
            result[1]
        ).strip()

        try:
            confidence = float(result[2])
        except Exception:
            confidence = 0.0

        if detected_text and confidence >= 0.40:
            texts.append(
                (
                    detected_text,
                    confidence
                )
            )

    final_text = []
    seen = set()

    for text, confidence in texts:

        key = text.lower().strip()

        if key not in seen:
            seen.add(key)
            final_text.append(text)

    return "\n".join(final_text).strip()
