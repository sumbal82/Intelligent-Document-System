from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)
from PIL import Image
import torch


# ============================================================
# LOAD SMOLVLM
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("Loading SmolVLM...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)
model.eval()

print(f"SmolVLM loaded on {device}")


# ============================================================
# YOLO INFORMATION
# ============================================================

def format_detections(detections):

    if not detections:
        return "No objects were detected."

    lines = []

    for obj in detections:

        name = obj.get("class", "Unknown")
        confidence = obj.get("confidence", 0)

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

    answer = answer.strip()

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

    if answer.startswith("User:"):
        answer = answer[
            len("User:"):
        ].strip()

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
        # IMAGE
        # ----------------------------------------------------

        if isinstance(image, str):

            image = Image.open(
                image
            ).convert("RGB")

        elif isinstance(image, Image.Image):

            image = image.convert("RGB")

        else:

            return None


        # ----------------------------------------------------
        # OCR CONTEXT
        # ----------------------------------------------------

        if document_text:

            ocr_context = str(
                document_text
            ).strip()

            # Prevent huge OCR text from making
            # every question unnecessarily slow.
            if len(ocr_context) > 4000:
                ocr_context = ocr_context[:4000]

        else:

            ocr_context = (
                "No reliable OCR text was extracted."
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

            return (
                "Please enter a question."
            )


        # ----------------------------------------------------
        # SHORT INSTRUCTION
        # ----------------------------------------------------

        instruction = f"""
Answer the user's question about the uploaded image.

Use the image as the primary source.

Use OCR when the question asks about written text.
Use object detections when useful.

Do not guess.
If something is not visible, say that it is not clearly visible.
Give only a concise answer.

OCR:
{ocr_context}

Detected objects:
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
        # PROCESS IMAGE + TEXT
        # ----------------------------------------------------

        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )


        # ----------------------------------------------------
        # MOVE TO CPU/GPU
        # ----------------------------------------------------

        inputs = {
            key: value.to(device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # GENERATE ANSWER
        # ----------------------------------------------------

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False
            )


        # ----------------------------------------------------
        # ONLY NEW TOKENS
        # ----------------------------------------------------

        if "input_ids" in inputs:

            input_length = (
                inputs["input_ids"]
                .shape[1]
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
                "The information is not clearly "
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

        return (
            "Please enter a question."
        )


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
