"""
WORKSHOP FILE 2 — Memory
========================
Concept: ConversationBufferMemory

By default, every LLM call is stateless — it remembers nothing.
LangChain's memory system fixes this by storing conversation history
and injecting it into each new prompt automatically.

After completing this file you will understand:
  - Why LLMs have no memory by default
  - How ConversationBufferMemory stores chat history
  - How ConversationChain uses memory behind the scenes
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

# --- LLM setup (provided) ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)

# ============================================================
# YOUR CODE HERE
# ------------------------------------------------------------
# Create a memory object that stores the full conversation history.
# The memory will be passed into ConversationChain below.
#
# Hint: ConversationBufferMemory(return_messages=True)
#
# Why return_messages=True? It keeps history as a list of message
# objects (HumanMessage, AIMessage) rather than a plain string —
# which works better with chat-style LLMs.
# ============================================================
memory = ConversationBufferMemory(return_messages=True)

# memory = None

# ============================================================

# --- Chain setup (provided — uses the memory you created above) ---
if memory is None:
    print("Memory is not set up yet. Complete the YOUR CODE HERE section above.")
else:
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    # --- Two-turn conversation to prove memory works ---
    print("Turn 1:")
    response1 = conversation.predict(
        input="Hi! My name is Arjun and I study at RVCE Bangalore."
    )
    print(f"AI: {response1}\n")

    print("Turn 2:")
    response2 = conversation.predict(
        input="What college did I just mention?"
    )
    print(f"AI: {response2}\n")

    print("Turn 3:")
    response3 = conversation.predict(
        input="And what's my name?"
    )
    print(f"AI: {response3}\n")

    # Show what's stored in memory
    print("--- What's stored in memory ---")
    for msg in memory.chat_memory.messages:
        role = "You" if msg.type == "human" else "AI"
        print(f"{role}: {msg.content[:80]}...")
