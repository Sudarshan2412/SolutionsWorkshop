"""
SOLUTION — File 4: Vector Store and Retriever
"""

from dotenv import load_dotenv
import importlib
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

load_and_split = importlib.import_module("3_loader_solution").load_and_split

load_dotenv()

PDF_PATH = "handbook.pdf"

class LocalMiniLMEmbeddings(Embeddings):
    """Small wrapper so Chroma can use a local sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


embeddings = LocalMiniLMEmbeddings()


def build_retriever(pdf_path: str):
    print("Loading and splitting PDF...")
    chunks = load_and_split(pdf_path)
    print(f"{len(chunks)} chunks ready for embedding.\n")

    # SOLUTION: create vector store and retriever
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


if __name__ == "__main__":
    retriever = build_retriever(PDF_PATH)

    test_query = "What is the minimum attendance required?"
    print(f"Test query: '{test_query}'\n")

    results = retriever.invoke(test_query)

    print(f"Retrieved {len(results)} chunks:\n")
    for i, doc in enumerate(results, 1):
        print(f"[Chunk {i}] (page {doc.metadata.get('page', '?')})")
        print(doc.page_content[:300])
        print()
