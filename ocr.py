import cv2
import numpy as np
import easyocr
import streamlit as st


# ============================================================
# EASYOCR MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def get_reader():
    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_for_ocr(image):

    if image is None:
        raise ValueError("Image is empty.")

    image = np.asarray(image)

    if image.ndim == 2:

        gray = image

    elif image.ndim == 3:

        if image.shape[2] == 4:
            image = image[:, :, :3]

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        raise ValueError(
            "Invalid image format for OCR."
        )

    # Noise reduction
    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text(image):

    processed = preprocess_for_ocr(image)

    # Model loads only when OCR is actually requested
    reader = get_reader()

    results = reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        mag_ratio=1.5
    )

    text_lines = []

    for item in results:

        if len(item) < 3:
            continue

        try:

            detected_text = str(
                item[1]
            ).strip()

            confidence = float(
                item[2]
            )

            if (
                detected_text
                and confidence >= 0.20
            ):

                text_lines.append(
                    detected_text
                )

        except Exception:
            continue

    text = "\n".join(
        text_lines
    )

    return text, results
