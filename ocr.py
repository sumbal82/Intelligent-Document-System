import easyocr
import numpy as np
import cv2
from PIL import Image


# ============================================================
# EASY OCR MODEL
# ============================================================

reader = easyocr.Reader(
    ['en'],
    gpu=False,
    verbose=False
)


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

    # Upscale image
    height, width = gray.shape

    if width < 1600:
        scale = 2
        gray = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
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

    # Convert back to RGB
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
        # Convert input
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
        # OCR on original image
        # ----------------------------------------------------

        original_results = reader.readtext(
            original,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=1.5
        )


        # ----------------------------------------------------
        # OCR on enhanced image
        # ----------------------------------------------------

        enhanced_results = reader.readtext(
            enhanced,
            detail=1,
            paragraph=False,
            text_threshold=0.6,
            low_text=0.3,
            link_threshold=0.4,
            mag_ratio=1.5
        )


        # ----------------------------------------------------
        # Collect text
        # ----------------------------------------------------

        texts = []

        for result in original_results:

            if len(result) >= 3:

                detected_text = str(
                    result[1]
                ).strip()

                confidence = float(
                    result[2]
                )

                if detected_text and confidence >= 0.35:

                    texts.append(
                        (
                            detected_text,
                            confidence
                        )
                    )


        for result in enhanced_results:

            if len(result) >= 3:

                detected_text = str(
                    result[1]
                ).strip()

                confidence = float(
                    result[2]
                )

                if detected_text and confidence >= 0.35:

                    texts.append(
                        (
                            detected_text,
                            confidence
                        )
                    )


        # ----------------------------------------------------
        # Remove duplicate detections
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

        text = "\n".join(
            final_text
        )

        return text.strip()


    except Exception as e:

        raise Exception(
            f"OCR failed: {str(e)}"
        )
