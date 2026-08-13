import easyocr
import numpy as np
from PIL import Image


print("Loading EasyOCR model...")

# English OCR
reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("EasyOCR loaded successfully")


def extract_text(image):
    """
    Extract text from the complete image using EasyOCR.
    """

    # PIL Image -> NumPy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    # EasyOCR
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

        # Ignore extremely low-confidence results
        if confidence >= 0.25 and text.strip():
            extracted_lines.append(text.strip())

    return "\n".join(extracted_lines)
