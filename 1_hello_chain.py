"""
WORKSHOP FILE 1 — Chains
========================
Concept: LangChain Expression Language (LCEL)

The most fundamental LangChain pattern — connecting a prompt template
and an LLM using the | (pipe) operator to form a chain.

After completing this file you will understand:
  - What a PromptTemplate is and why it's useful
  - How to connect components using the | operator
  - How to invoke a chain and read the output
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

# --- LLM setup (provided — no need to change this) ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)

# --- Prompt template (provided — no need to change this) ---
prompt = PromptTemplate.from_template(
    "Explain {topic} in 2-3 sentences, like I'm a first year engineering student."
)

# ============================================================
# YOUR CODE HERE
# ------------------------------------------------------------
# Connect the prompt and the llm into a chain using the | operator
# The chain should: take a topic → format the prompt → send to LLM
#
# Hint: chain = prompt | llm
# ============================================================

chain = None  # replace this line

# ============================================================

# --- Test your chain (provided — run this to check your work) ---
if chain is None:
    print("Chain is not set up yet. Complete the YOUR CODE HERE section above.")
else:
    print("Running chain...\n")
    result = chain.invoke({"topic": "what a vector database is"})
    print("Result:")
    print(result.content)
