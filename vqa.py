import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"

_device = None
_processor = None
_model = None


# ============================================================
# LOAD MODEL ONLY ONCE
# ============================================================

def get_model():

    global _device
    global _processor
    global _model

    if _model is None:

        _device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        _processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME
        )

        _model = _model.to(_device)
        _model.eval()

    return _processor, _model, _device


# ============================================================
# PREPARE IMAGE
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
# FORMAT YOLO DETECTIONS
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
        # MODEL
        # ----------------------------------------------------

        processor, model, device = get_model()

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

        image = prepare_vlm_image(image)

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        question = str(
            question
        ).strip()

        if not question:

            return "Please enter a question."

        # ----------------------------------------------------
        # OCR CONTEXT
        # ----------------------------------------------------

        if document_text:

            ocr_context = str(
                document_text
            ).strip()

            # Prevent huge prompts
            if len(ocr_context) > 4000:

                ocr_context = (
                    ocr_context[:4000]
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
        # INSTRUCTION
        # ----------------------------------------------------

        instruction = f"""
Answer the user's question about the uploaded image.

Rules:
1. Look carefully at the image.
2. Use the OCR text when the question is about written text.
3. Use the detected objects when useful.
4. Do not invent information.
5. If the answer cannot be determined from the image, say so.
6. Give a short and direct answer.

OCR TEXT:
{ocr_context}

DETECTED OBJECTS:
{yolo_context}

QUESTION:
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
        # PROCESS
        # ----------------------------------------------------

        inputs = processor(
            text=prompt,
            images=[image],
            return_tensors="pt"
        )

        # ----------------------------------------------------
        # DEVICE
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
                max_new_tokens=40,
                do_sample=False
            )

        # ----------------------------------------------------
        # REMOVE PROMPT TOKENS
        # ----------------------------------------------------

        if "input_ids" in inputs:

            input_length = (
                inputs["input_ids"].shape[1]
            )

            generated_tokens = output[
                :,
                input_length:
            ]

        else:

            generated_tokens = output

        # ----------------------------------------------------
        # DECODE
        # ----------------------------------------------------

        answer = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

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
