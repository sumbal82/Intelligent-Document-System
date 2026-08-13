from PIL import Image, ImageEnhance


def preprocess_image(image):
    """
    Prepare image for OCR while preserving text information.
    """

    image = image.convert("RGB")

    # Improve contrast slightly
    image = ImageEnhance.Contrast(image).enhance(1.3)

    # Improve sharpness slightly
    image = ImageEnhance.Sharpness(image).enhance(1.5)

    return image
