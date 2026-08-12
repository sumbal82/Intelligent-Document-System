import cv2
import numpy as np
import easyocr


_reader = None


def get_reader():
    global _reader

    if _reader is None:
        _reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )

    return _reader


def preprocess_for_ocr(image):
    """
    Image -> grayscale -> denoise -> contrast enhancement
    """

    if image is None:
        raise ValueError("Image is empty")

    # BGR image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Remove small noise
    gray = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def extract_text(image):
    """
    Extract text from an image.

    Returns:
        text: complete OCR text
        results: detailed OCR detections
    """

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
            detected_text = str(item[1]).strip()
            confidence = float(item[2])

            # Ignore extremely weak detections
            if detected_text and confidence >= 0.20:
                text_lines.append(detected_text)

    text = "\n".join(text_lines)

    return text, results
