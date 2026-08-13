from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading SmolVLM...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

print(f"SmolVLM loaded on: {device}")


def answer_image_question(image, question):

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    image = image.convert("RGB")

    question = str(question).strip()

    if not question:
        return "Please enter a question about the image."

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image"
                },
                {
                    "type": "text",
                    "text": question
                }
            ]
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=80
        )

    answer = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    answer = answer.strip()

    return answer
