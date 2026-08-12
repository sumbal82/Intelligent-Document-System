from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)
from PIL import Image
import torch

# ============================================================
# LOAD SMOLVLM ONCE
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading SmolVLM...")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME
)

model.to(device)
model.eval()

print(f"SmolVLM loaded on {device}")


# ============================================================
# RESIZE IMAGE FOR FASTER VLM
# ============================================================

def prepare_vlm_image(image):

    image = image.convert("RGB")

    max_size = 1024

    width, height = image.size

    if max(width, height) > max_size:

        scale = max_size / max(width, height)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    return image


# ============================================================
# YOLO INFORMATION
# ============================================================

def format_detections(detections):

    if not detections:
        return "No objects were detected."

    lines = []

    for obj in detections:

        name = str(
            obj.get("class", "Unknown")
        )

        confidence = obj.get(
            "confidence",
            0
        )

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        lines.append(
            f"- {name} ({confidence:.2f})"
        )

    return "\n".join(lines)


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer):

    if not answer:
        return ""

    answer = str(answer).strip()

    prefixes = [
        "Assistant:",
        "assistant:",
        "Answer:",
        "answer:"
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

    if "Assistant:" in answer:

        answer = answer.split(
            "Assistant:"
        )[-1].strip()

    if "assistant:" in answer:

        answer = answer.split(
            "assistant:"
        )[-1].strip()

    return answer.strip()


# ============================================================
# ASK VLM
# ============================================================

def ask_vlm(
    image,
    question,
    document_text="",
    detections=None
):

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        if isinstance(image, str):

            image = Image.open(
                image
            ).convert("RGB")

        elif isinstance(image, Image.Image):

            image = image.convert("RGB")

        else:

            return None

        # Resize only for VLM
        image = prepare_vlm_image(image)


        # ----------------------------------------------------
        # OCR CONTEXT
        # ----------------------------------------------------

        if document_text:

            ocr_context = str(
                document_text
            ).strip()

            if len(ocr_context) > 2500:

                ocr_context = (
                    ocr_context[:2500]
                    + "..."
                )

        else:

            ocr_context = (
                "No readable text was detected."
            )


        # ----------------------------------------------------
        # YOLO CONTEXT
        # ----------------------------------------------------

        yolo_context = format_detections(
            detections
        )


        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        question = str(
            question
        ).strip()

        if not question:

            return "Please enter a question."


        # ----------------------------------------------------
        # SHORT PROMPT
        # ----------------------------------------------------

        instruction = f"""
Answer the question about this image.

Look carefully at the image.
Use OCR text when useful.
Use detected objects when useful.
Answer only what can be determined from the image.
Do not invent information.
Keep the answer short.

OCR:
{ocr_context}

Objects:
{yolo_context}

Question:
{question}
"""


        # ----------------------------------------------------
        # CHAT TEMPLATE
        # ----------------------------------------------------

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image"
                    },
                    {
                        "type": "text",
                        "text": instruction
                    }
                ]
            }
        ]


        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )


        # ----------------------------------------------------
        # PROCESS IMAGE
        # ----------------------------------------------------

        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )


        # ----------------------------------------------------
        # MOVE TO DEVICE
        # ----------------------------------------------------

        inputs = {
            key: value.to(device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )


        # ----------------------------------------------------
        # REMOVE INPUT TOKENS
        # ----------------------------------------------------

        if "input_ids" in inputs:

            input_length = (
                inputs["input_ids"].shape[1]
            )

            generated_tokens = (
                output[:, input_length:]
            )

        else:

            generated_tokens = output


        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        answer = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]


        # ----------------------------------------------------
        # CLEAN
        # ----------------------------------------------------

        answer = clean_answer(
            answer
        )


        if not answer:

            return (
                "The answer is not clearly "
                "visible in the image."
            )

        return answer


    except Exception as error:

        print(
            "SmolVLM error:",
            error
        )

        return None


# ============================================================
# PUBLIC FUNCTION
# ============================================================

def answer_question(
    image,
    question,
    document_text="",
    detections=None
):

    question = str(
        question
    ).strip()

    if not question:

        return "Please enter a question."

    answer = ask_vlm(
        image,
        question,
        document_text,
        detections
    )

    if answer:

        return answer

    return (
        "I could not determine a reliable "
        "answer from the uploaded image."
    )
