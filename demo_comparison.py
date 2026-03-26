"""
DEMO — Hands-on Session: LangChain Impact
==========================================

Simple frontend version of demo comparison.
Run:
    python demo_comparison.py
Then open http://localhost:7860
"""

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

FLOW
======================================================================
1) Show no-memory behavior (model gets turns independently)
2) Show memory behavior (ConversationBufferMemory)
3) Continue with RAG files and then app.py
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


def run_demo() -> tuple[str, str]:
    guide_text = workshop_guide()
    try:
        live_text = run_live_memory_comparison()
    except Exception as exc:
        live_text = (
            "Live comparison could not run.\n"
            f"Reason: {exc}\n"
            "Check GROQ_API_KEY in .env and try again."
        )
    return guide_text, live_text


demo = gr.Interface(
    fn=run_demo,
    inputs=None,
    outputs=[
        gr.Textbox(label="Workshop Guide", lines=18),
        gr.Textbox(label="Live Demo Output", lines=22),
    ],
    title="LangChain Demo Comparison",
    description="Simple frontend for workshop guide + live no-memory vs memory comparison.",
)


if __name__ == "__main__":
    demo.launch()
