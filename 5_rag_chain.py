"""
WORKSHOP FILE 5 — Full RAG Chain (The Final Boss)
==================================================
Concept: RetrievalQA

This is where everything comes together. RetrievalQA is a LangChain
chain that:
  1. Takes a user question
  2. Runs it through the retriever (file 4) to fetch relevant chunks
  3. Injects those chunks into a prompt as context
  4. Sends the prompt to the LLM
  5. Returns a grounded answer based only on the document

This is the answer to the hallucination demo — the same question that
the LLM got wrong at the start of the session will now be answered
correctly from the actual handbook.

After completing this file you will understand:
  - How RetrievalQA chains a retriever and LLM together
  - What "grounded" answers mean and why they matter
  - How the full RAG pipeline flows end to end
"""

from dotenv import load_dotenv
import importlib
from langchain_groq import ChatGroq  
from langchain_classic.chains import RetrievalQA 

build_retriever = importlib.import_module("4_retriever").build_retriever
load_dotenv()

PDF_PATH = "cats_and_dogs_notebook.pdf"

# --- LLM setup (provided) ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)

def get_rag_chain(pdf_path: str):
    """
    Build and return the full RAG chain.
    This function is imported by app.py to power the chat UI.
    """

    print("Building retriever from PDF...")
    retriever = build_retriever(pdf_path)
    print("Retriever ready.\n")

    # ============================================================
    # YOUR CODE HERE
    # ------------------------------------------------------------
    # Create a RetrievalQA chain that connects the retriever and LLM.
    #
    # RetrievalQA.from_chain_type() takes:
    #   llm        → the language model that generates the answer
    #   retriever  → fetches the relevant chunks for each question
    #   return_source_documents=True → also return which chunks were used
    #                                  (useful for debugging and trust)
    #
    # Hint:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    # ============================================================

    # qa_chain = None 

    # ============================================================

    return qa_chain


# --- Run this file directly to test your work ---
if __name__ == "__main__":
    chain = get_rag_chain(PDF_PATH)

    if chain is None:
        print("RAG chain not created yet. Complete the YOUR CODE HERE section above.")
    else:
        # This is the same question asked during the hallucination demo!
        question = "What is the minimum attendance required to sit for exams?"

        print(f"Question: {question}\n")
        print("Thinking...\n")

        result = chain.invoke({"query": question})

        print("Answer:")
        print(result["result"])

        print("\n--- Source chunks used to generate this answer ---")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\n[Source {i}] page {doc.metadata.get('page', '?')}:")
            print(doc.page_content[:250])
