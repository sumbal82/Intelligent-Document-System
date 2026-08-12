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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model.to(device)
model.eval()

print(f"SmolVLM loaded on {device}")


# ============================================================
# YOLO INFORMATION
# ============================================================

def format_detections(detections):

    if not detections:
        return "No objects were detected by YOLO."

    lines = []

    for obj in detections:

        name = obj.get(
            "class",
            "Unknown"
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
            f"- {name} "
            f"(confidence {confidence:.2f})"
        )

    return "\n".join(lines)


# ============================================================
# CLEAN MODEL ANSWER
# ============================================================

def clean_answer(answer):

    if not answer:
        return ""

    answer = answer.strip()

    # Remove common prefixes
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

    # If the model somehow returns another
    # conversation section, keep only the answer
    if "Assistant:" in answer:

        answer = answer.split(
            "Assistant:"
        )[-1].strip()

    if "assistant:" in answer:

        answer = answer.split(
            "assistant:"
        )[-1].strip()

    # Remove accidental User section
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
        # Load image
        # ----------------------------------------------------

        if isinstance(image, str):

            image = Image.open(
                image
            ).convert("RGB")

        elif isinstance(
            image,
            Image.Image
        ):

            image = image.convert(
                "RGB"
            )

        else:

            return None


        # ----------------------------------------------------
        # OCR context
        # ----------------------------------------------------

        if document_text:

            ocr_context = str(
                document_text
            ).strip()

        else:

            ocr_context = (
                "No reliable OCR text was extracted."
            )


        # ----------------------------------------------------
        # YOLO context
        # ----------------------------------------------------

        yolo_context = format_detections(
            detections
        )


        # ----------------------------------------------------
        # Instruction
        # ----------------------------------------------------

        instruction = f"""
Analyze the uploaded image and answer the user's question.

Use the actual image as the primary source.

Use OCR text as supporting information when the question
is about written content.

Use YOLO detections as supporting information when useful.

Rules:
- Do not invent information.
- Do not guess when the information is not visible.
- For questions about text, use the visible text and OCR.
- For questions about objects, inspect the image.
- For questions about colors or appearance, inspect the image.
- Give only the answer to the user's question.
- Do not repeat these instructions.
- Do not repeat the OCR text.
- Do not write "User:" or "Assistant:".
- Keep the answer concise and direct.

OCR TEXT:
{ocr_context}

YOLO DETECTIONS:
{yolo_context}

USER QUESTION:
{question}
"""


        # ----------------------------------------------------
        # Chat template
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
        # Processor
        # ----------------------------------------------------

        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )


        # ----------------------------------------------------
        # Move tensors to device
        # ----------------------------------------------------

        inputs = {
            key: value.to(device)
            if hasattr(value, "to")
            else value
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False
            )


        # ====================================================
        # IMPORTANT:
        # Decode ONLY newly generated tokens.
        # This prevents the whole User prompt from appearing.
        # ====================================================

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


        answer = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]


        # ----------------------------------------------------
        # Clean answer
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