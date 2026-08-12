
import cv2
import easyocr
import streamlit as st


@st.cache_resource
def get_reader():

    return easyocr.Reader(
        ["en"],
        gpu=False,
        verbose=False
    )


def preprocess_for_ocr(image):

    if image is None:
        raise ValueError("Image is empty")

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def extract_text(image):

    processed = preprocess_for_ocr(image)

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

        if len(item) >= 3:

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

    text = "\n".join(text_lines)

    return text, results
