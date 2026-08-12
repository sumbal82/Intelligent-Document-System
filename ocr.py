import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image


# ============================================================
# LOAD EASY OCR ONLY ONCE
# ============================================================

@st.cache_resource
def load_ocr():

    return easyocr.Reader(
        ['en'],
        gpu=False,
        verbose=False
    )


reader = load_ocr()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def prepare_image(image):

    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if not isinstance(image, np.ndarray):
        raise ValueError("Invalid image format.")

    # RGB -> BGR
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )
    else:
        image_bgr = image

    # Grayscale
    gray = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2GRAY
    )

    # Upscale only small images
    height, width = gray.shape

    if width < 1200:

        scale = 2

        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Light noise reduction
    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    # Back to RGB
    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )

    return enhanced


# ============================================================
# OCR
# ============================================================

def extract_text(image):

    try:

        # ----------------------------------------------------
        # Convert image
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # PREPARE IMAGE
        # ----------------------------------------------------

        enhanced = prepare_image(
            original
        )


        # ----------------------------------------------------
        # OCR ORIGINAL IMAGE
        # ----------------------------------------------------

        original_results = reader.readtext(
            original,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=1.2
        )


        # ----------------------------------------------------
        # OCR ENHANCED IMAGE
        # ----------------------------------------------------

        enhanced_results = reader.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=1.2
        )


        # ----------------------------------------------------
        # COLLECT RESULTS
        # ----------------------------------------------------

        texts = []

        for result in (
            original_results +
            enhanced_results
        ):

            if len(result) < 3:
                continue

            detected_text = str(
                result[1]
            ).strip()

            try:

                confidence = float(
                    result[2]
                )

            except Exception:

                confidence = 0.0


            if (
                detected_text
                and confidence >= 0.40
            ):

                texts.append(
                    (
                        detected_text,
                        confidence
                    )
                )


        # ----------------------------------------------------
        # REMOVE DUPLICATES
        # ----------------------------------------------------

        final_text = []

        seen = set()

        for text, confidence in texts:

            key = text.lower().strip()

            if key not in seen:

                seen.add(key)

                final_text.append(
                    text
                )


        # ----------------------------------------------------
        # FINAL TEXT
        # ----------------------------------------------------

        return "\n".join(
            final_text
        ).strip()


    except Exception as e:

        raise Exception(
            f"OCR failed: {str(e)}"
        )
