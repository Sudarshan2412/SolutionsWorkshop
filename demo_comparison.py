"""
DEMO — Hands-on Session: LangChain Impact
==========================================

Frontend runbook + live comparison for workshop delivery.
Run:
    python demo_comparison.py
Then open http://localhost:7860
"""

import importlib
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()


def workshop_guide() -> str:
    return """
WORKSHOP STRUCTURE
======================================================================
PART A: PARTICIPANT MODE
  python 1_hello_chain.py
  python 2_memory.py
  python 3_loader.py
  python 4_retriever.py
  python 5_rag_chain.py

PART B: SOLUTION MODE
  python solutions/1_hello_chain_solution.py
  python solutions/2_memory_solution.py
  python solutions/3_loader_solution.py
  python solutions/4_retriever_solution.py
  python solutions/5_rag_chain_solution.py

PART C: APP MODE
    python app.py

APP CAPABILITIES
======================================================================
- Multi-source RAG answers from handbook + pet guide.
- Image upload is supported directly in chat.

HANDBOOK EXAMPLE QUERIES
======================================================================
- What is the minimum attendance required to sit for exams?
- What is the policy on late assignment submissions?
- How is the final grade calculated in the handbook?
- What is the process to apply for leave?

PET GUIDE EXAMPLE QUERIES
======================================================================
- What are common health issues for cats?
- How should I train my dog?
- What's the best diet for dogs?

FLOW
======================================================================
1) Show no-memory behavior (model gets turns independently)
2) Show memory behavior (ConversationBufferMemory)
3) Optionally run one handbook RAG smoke check
4) Continue with app.py and demo text/image upload
""".strip()


def run_live_memory_comparison() -> str:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=256,
    )

    turn1 = "Hi! My name is Arjun and I study at RVCE Bangalore."
    turn2 = "What college did I just mention?"

    lines = ["LIVE COMPARISON: WITHOUT MEMORY vs WITH MEMORY", "=" * 70, ""]

    lines.append("[WITHOUT MEMORY] Independent LLM calls")
    lines.append(f"Turn 1 (user): {turn1}")
    no_mem_1 = llm.invoke(turn1)
    lines.append(f"Turn 1 (ai): {no_mem_1.content}")
    lines.append(f"Turn 2 (user): {turn2}")
    no_mem_2 = llm.invoke(turn2)
    lines.append(f"Turn 2 (ai): {no_mem_2.content}")
    lines.append("")

    lines.append("[WITH MEMORY] ConversationChain + ConversationBufferMemory")
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    lines.append(f"Turn 1 (user): {turn1}")
    mem_1 = conversation.predict(input=turn1)
    lines.append(f"Turn 1 (ai): {mem_1}")
    lines.append(f"Turn 2 (user): {turn2}")
    mem_2 = conversation.predict(input=turn2)
    lines.append(f"Turn 2 (ai): {mem_2}")
    lines.append("")
    lines.append("Summary: second block should be more context-aware.")

    return "\n".join(lines)


def run_optional_rag_smoke_check() -> str:
    """
    Optional smoke check for handbook retrieval after participants complete File 5.
    """
    try:
        get_rag_chain = importlib.import_module("5_rag_chain").get_rag_chain
        chain = get_rag_chain("handbook.pdf")
        question = "What is the minimum attendance required to sit for exams?"
        result = chain.invoke({"query": question})

        lines = [
            "OPTIONAL RAG SMOKE CHECK",
            "=" * 70,
            f"Question: {question}",
            "",
            "Answer:",
            result.get("result", ""),
            "",
            "Sources:",
        ]

        for i, doc in enumerate(result.get("source_documents", [])[:3], 1):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            lines.append(f"{i}. {source} (page {page})")

        return "\n".join(lines)
    except Exception as exc:
        return (
            "OPTIONAL RAG SMOKE CHECK\n"
            + "=" * 70
            + "\nCould not run RAG smoke check.\n"
            + f"Reason: {exc}\n"
            + "Complete File 5 or ensure GROQ_API_KEY is configured before retrying."
        )


def run_demo(run_rag_smoke_check: bool) -> tuple[str, str, str]:
    guide_text = workshop_guide()
    try:
        live_text = run_live_memory_comparison()
    except Exception as exc:
        live_text = (
            "Live comparison could not run.\n"
            f"Reason: {exc}\n"
            "Check GROQ_API_KEY in .env and try again."
        )

    rag_text = (
        run_optional_rag_smoke_check()
        if run_rag_smoke_check
        else "Optional RAG smoke check skipped. Enable the checkbox and run again to execute it."
    )

    return guide_text, live_text, rag_text


demo = gr.Interface(
    fn=run_demo,
    inputs=gr.Checkbox(
        label="Run optional handbook RAG smoke check",
        value=False,
    ),
    outputs=[
        gr.Textbox(label="Workshop Guide", lines=28),
        gr.Textbox(label="Live Demo Output", lines=22),
        gr.Textbox(label="Optional RAG Smoke Check", lines=14),
    ],
    title="Workshop Demo Comparison",
    description=(
        "Runbook + live no-memory vs memory comparison, with an optional handbook RAG smoke check."
    ),
)


if __name__ == "__main__":
    demo.launch()
