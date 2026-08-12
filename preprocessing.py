import cv2
from PIL import Image, ImageEnhance, ImageFilter


def preprocess_image(image_path):

    # ---------------------------------------------------------
    # Read image
    # ---------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # ---------------------------------------------------------
    # Convert BGR -> RGB
    # ---------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # ---------------------------------------------------------
    # Resize only when image is very small
    # ---------------------------------------------------------

    height, width = image.shape[:2]

    max_side = max(
        width,
        height
    )

    if max_side < 1200:

        scale = 1200 / max_side

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_CUBIC
        )

    # ---------------------------------------------------------
    # PIL RGB
    # ---------------------------------------------------------

    processed_image = Image.fromarray(
        image
    ).convert("RGB")

    # ---------------------------------------------------------
    # Mild enhancement
    # IMPORTANT:
    # Do NOT convert every image to black/white.
    # ---------------------------------------------------------

    processed_image = ImageEnhance.Contrast(
        processed_image
    ).enhance(1.10)

    processed_image = ImageEnhance.Sharpness(
        processed_image
    ).enhance(1.15)

    return processed_image