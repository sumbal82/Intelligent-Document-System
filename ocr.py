import easyocr
import numpy as np
from PIL import Image

reader = easyocr.Reader(['en'], gpu=False)

def extract_text(image):
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)

        results = reader.readtext(image, detail=0, paragraph=True)

        text = "\n".join(results)

        return text.strip()

    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")