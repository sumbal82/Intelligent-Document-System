import numpy as np
from PIL import Image
import streamlit as st


@st.cache_resource
def load_ocr_model():
    import easyocr

    print("Loading EasyOCR model...")

    reader = easyocr.Reader(
        ['en'],
        gpu=False,
        verbose=True
    )

    print("EasyOCR loaded successfully")

    return reader


def extract_text(image):
    """
    Extract text from the complete image using EasyOCR.
    """

    reader = load_ocr_model()

    # PIL Image -> NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    results = reader.readtext(
        image,
        detail=1,
        paragraph=False
    )

    extracted_lines = []

    for result in results:

        if len(result) < 3:
            continue

        text = result[1]
        confidence = result[2]

        if confidence >= 0.25 and text.strip():
            extracted_lines.append(text.strip())

    return "\n".join(extracted_lines)
