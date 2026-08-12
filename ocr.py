import easyocr
import numpy as np
import cv2
from PIL import Image

# ============================================================
# EASY OCR MODEL - LAZY LOADING
# ============================================================

reader = None


def get_reader():
    global reader

    if reader is None:
        reader = easyocr.Reader(
            ["en"],
            gpu=False,
            verbose=False
        )

    return reader


# ============================================================
# IMAGE PREPROCESSING FOR OCR
# ============================================================

def prepare_image(image):
    """
    Prepare image to improve OCR readability.
    """

    # PIL Image -> NumPy
    if isinstance(image, Image.Image):
        image = np.array(image)

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

    # Upscale smaller images
    height, width = gray.shape

    if width < 1600:
        gray = cv2.resize(
            gray,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC
        )

    # Improve contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Small noise reduction
    enhanced = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    # Grayscale -> RGB
    enhanced = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )

    return enhanced


# ============================================================
# OCR TEXT EXTRACTION
# ============================================================

def extract_text(image):

    try:

        # ----------------------------------------------------
        # Load EasyOCR only when OCR is actually needed
        # ----------------------------------------------------

        reader_model = get_reader()

        # ----------------------------------------------------
        # Convert input image
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
        # Prepare enhanced image
        # ----------------------------------------------------

        enhanced = prepare_image(
            original
        )

        # ----------------------------------------------------
        # OCR original image
        # ----------------------------------------------------

        original_results = reader_model.readtext(
            original,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=1.5
        )

        # ----------------------------------------------------
        # OCR enhanced image
        # ----------------------------------------------------

        enhanced_results = reader_model.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=1.5
        )

        # ----------------------------------------------------
        # Collect results
        # ----------------------------------------------------

        texts = []

        for result in original_results + enhanced_results:

            if len(result) >= 3:

                detected_text = str(
                    result[1]
                ).strip()

                confidence = float(
                    result[2]
                )

                if (
                    detected_text
                    and confidence >= 0.35
                ):
                    texts.append(
                        (
                            detected_text,
                            confidence
                        )
                    )

        # ----------------------------------------------------
        # Remove duplicates
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
        # Final text
        # ----------------------------------------------------

        return "\n".join(
            final_text
        ).strip()

    except Exception as e:

        raise Exception(
            f"OCR failed: {str(e)}"
        )
