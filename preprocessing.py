import cv2
from PIL import Image, ImageEnhance


def preprocess_image(image_path):

    if not image_path:
        raise ValueError("Image path is empty.")

    # ---------------------------------------------------------
    # Read image
    # ---------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found or could not be read: {image_path}"
        )

    # ---------------------------------------------------------
    # Convert BGR -> RGB
    # ---------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ---------------------------------------------------------
    # Resize small images
    # ---------------------------------------------------------

    height, width = image.shape[:2]

    max_side = max(width, height)

    if max_side < 1200:

        scale = 1200 / max_side

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC
        )

    # ---------------------------------------------------------
    # Convert to PIL
    # ---------------------------------------------------------

    processed_image = Image.fromarray(
        image
    ).convert("RGB")

    # ---------------------------------------------------------
    # Mild enhancement
    # ---------------------------------------------------------

    processed_image = ImageEnhance.Contrast(
        processed_image
    ).enhance(1.10)

    processed_image = ImageEnhance.Sharpness(
        processed_image
    ).enhance(1.15)

    return processed_image
