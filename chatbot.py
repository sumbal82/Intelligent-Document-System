import re

# Extracted text file from Step 1
TEXT_FILE = "input/extracted_text.txt"

# Load document text
with open(TEXT_FILE, "r", encoding="utf-8") as file:
    document_text = file.read()

# Split document into lines
lines = [
    line.strip()
    for line in document_text.splitlines()
    if line.strip()
]


def answer_question(question):

    question_lower = question.lower().strip()

    # -----------------------------------------
    # TITLE QUESTIONS
    # -----------------------------------------
    title_words = [
        "title",
        "document title",
        "name of the document",
        "document name"
    ]

    if any(word in question_lower for word in title_words):

        # Usually the first meaningful line is the document title
        if lines:
            return lines[0]

    # -----------------------------------------
    # GENERAL "WHAT IS WRITTEN" QUESTIONS
    # -----------------------------------------
    if (
        "what is written" in question_lower
        or "what text" in question_lower
        or "text on the document" in question_lower
    ):
        return "\n".join(lines[:5])

    # -----------------------------------------
    # SIGNATURE / DIRECTOR QUESTIONS
    # -----------------------------------------
    if "signature" in question_lower or "sign" in question_lower:

        for line in lines:
            if "sign" in line.lower() or "signature" in line.lower():
                return line

    # -----------------------------------------
    # NORMAL KEYWORD SEARCH
    # -----------------------------------------

    keywords = re.findall(r"[a-zA-Z0-9]+", question_lower)

    stop_words = {
        "what", "is", "the", "a", "an", "of", "to",
        "who", "when", "where", "which", "tell", "me",
        "about", "this", "that", "was", "are", "do",
        "does", "did", "how", "why", "can", "you"
    }

    keywords = [
        word for word in keywords
        if word not in stop_words
    ]

    results = []

    for line in lines:

        line_lower = line.lower()

        score = 0

        for keyword in keywords:
            if keyword in line_lower:
                score += 1

        if score > 0:
            results.append((score, line))

    # Sort by relevance
    results.sort(key=lambda x: x[0], reverse=True)

    if results:
        return "\n".join(
            line for score, line in results[:3]
        )

    return "Sorry, this information was not found in the document."


# =========================================
# CHATBOT INTERFACE
# =========================================

print("=" * 50)
print("          DOCUMENT CHATBOT")
print("=" * 50)

print("\nChatbot is ready!")
print("Ask questions about your document.")
print("Type 'exit' to close.\n")

while True:

    question = input("You: ")

    if question.lower().strip() == "exit":
        print("Chatbot: Goodbye!")
        break

    answer = answer_question(question)

    print("Chatbot:", answer)