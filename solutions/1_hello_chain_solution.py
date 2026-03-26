"""
SOLUTION — File 1: Chains
"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)

prompt = PromptTemplate.from_template(
    "Explain {topic} in 2-3 sentences, like I'm a first year engineering student."
)

# SOLUTION: connect prompt and llm with the pipe operator
chain = prompt | llm

print("Running chain...\n")
result = chain.invoke({"topic": "what a vector database is"})
print("Result:")
print(result.content)
