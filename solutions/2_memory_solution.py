"""
SOLUTION — File 2: Memory
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationBufferMemory

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)

# SOLUTION: create memory object
memory = ConversationBufferMemory(return_messages=True)

conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

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

print("--- What's stored in memory ---")
for msg in memory.chat_memory.messages:
    role = "You" if msg.type == "human" else "AI"
    print(f"{role}: {msg.content[:80]}...")
