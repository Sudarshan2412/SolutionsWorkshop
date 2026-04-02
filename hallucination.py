"""
WORKSHOP DEMO — Hallucination Baseline (Raw LLM Call)
======================================================
Concept: Ask the LLM directly without retrieval.

This file intentionally does NOT use:
- document loading
- vector store
- retriever
- RetrievalQA

Use this as the "before RAG" demo to show that the model can sound
confident even when it is not grounded in your handbook.
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


# Same model family used across the workshop, but called directly.
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)


def run_raw_llm_call(question: str) -> str:
    """Send a question directly to the model with no retrieval context."""
    response = llm.invoke(question)
    return response.content


if __name__ == "__main__":
    default_question = "What is the backlog exam fee at my college?"

    print("RAW LLM CALL (NO RETRIEVAL)")
    print("=" * 40)
    print(
        "This answer is ungrounded. The model may produce a plausible but incorrect answer.\n"
    )

    typed_question = input(
        "Enter a question (or press Enter to use the demo question): "
    ).strip()
    question = typed_question or default_question

    print(f"\nQuestion: {question}\n")

    answer = run_raw_llm_call(question)

    print("Answer:")
    print(answer)
