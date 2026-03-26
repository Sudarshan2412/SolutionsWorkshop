"""
app.py — Chat Dashboard (fully provided, no blanks)
====================================================
This is the frontend. It imports the RAG chain you built in
5_rag_chain.py and wraps it in a Gradio chat interface.

Run this file AFTER completing all 5 workshop files:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import importlib
from dotenv import load_dotenv

get_rag_chain = importlib.import_module("5_rag_chain").get_rag_chain

load_dotenv()

PDF_PATH = "handbook.pdf"

print("Initialising RAG chain — this may take a minute on first run...")
print("(The embedding model is being downloaded and the PDF is being indexed)\n")

chain = get_rag_chain(PDF_PATH)

if chain is None:
    print("\nWARNING: RAG chain is not set up yet.")
    print("Complete the YOUR CODE HERE section in 5_rag_chain.py first, then re-run app.py.\n")


def chat(message: str, history: list) -> str:
    """
    Called by Gradio on every user message.
    Passes the message to the RAG chain and returns the answer.
    """
    if chain is None:
        return "The RAG chain is not set up yet. Please complete 5_rag_chain.py first."
    result = chain.invoke({"query": message})
    return result["result"]


demo = gr.ChatInterface(
    fn=chat,
    title="Ask Your College Handbook",
    description=(
        "This chatbot answers questions using your college's actual handbook. "
        "It retrieves relevant sections and generates grounded answers — "
        "no hallucinations, no made-up rules."
    ),
    examples=[
        "What is the minimum attendance required to sit for exams?",
        "How many backlogs am I allowed before I am detained?",
        "What is the process to apply for a re-evaluation?",
        "What are the rules for using the college hostel?",
    ],
)

if __name__ == "__main__":
    demo.launch()
