"""
SOLUTION — File 5: Full RAG Chain
"""

from dotenv import load_dotenv
import importlib
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

build_retriever = importlib.import_module("4_retriever").build_retriever

load_dotenv()

PDF_PATH = "handbook.pdf"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
)


def get_rag_chain(pdf_path: str):
    print("Building retriever from PDF...")
    retriever = build_retriever(pdf_path)
    print("Retriever ready.\n")

    # SOLUTION: assemble the full RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain


if __name__ == "__main__":
    chain = get_rag_chain(PDF_PATH)

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
