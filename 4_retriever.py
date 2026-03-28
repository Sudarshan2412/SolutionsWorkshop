"""
WORKSHOP FILE 4 — Vector Store and Retriever
=============================================
Concept: Local Embeddings + Chroma + Retriever

This is the core of RAG. Each chunk from file 3 gets converted into
a vector (a list of numbers that captures its meaning). These vectors
are stored in ChromaDB. At query time, the user's question is also
embedded into a vector, and the chunks closest in meaning are retrieved.

This is semantic search — not keyword matching. "attendance rule" will
retrieve chunks about "minimum presence requirement" even if the words
don't match exactly.

After completing this file you will understand:
  - What embeddings are and why they enable semantic search
  - How Chroma stores and queries vectors locally
  - What a retriever is and what k=3 means
"""

from dotenv import load_dotenv
import importlib
from collections.abc import Sequence
import os
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

load_and_split = importlib.import_module("3_loader").load_and_split

load_dotenv()

PDF_PATH = "handbook.pdf"

# --- Embedding model setup (provided) ---
# This model runs locally — no API calls, no rate limits.
# It converts text into 384-dimensional vectors.
class LocalMiniLMEmbeddings(Embeddings):
    """Small wrapper so Chroma can use a local sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


embeddings = LocalMiniLMEmbeddings()


def _normalise_pdf_paths(pdf_path: str | Sequence[str]) -> list[str]:
    if isinstance(pdf_path, str):
        return [pdf_path]
    return [path for path in pdf_path if isinstance(path, str) and path.strip()]


def build_retriever(pdf_path: str | Sequence[str]):
    """
    Load, split, embed, and store document chunks.
    Returns a retriever that can fetch the top-k relevant chunks
    for any query.
    """

    pdf_paths = _normalise_pdf_paths(pdf_path)
    if not pdf_paths:
        raise ValueError("No PDF paths provided to build_retriever.")

    all_chunks = []
    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF not found: {path}")

        print(f"Loading and splitting PDF: {path}")
        chunks = load_and_split(path)
        all_chunks.extend(chunks)

    print(
        f"{len(all_chunks)} chunks ready for embedding from {len(pdf_paths)} PDF(s).\n"
    )

    # ============================================================
    # YOUR CODE HERE
    # ------------------------------------------------------------
    # 1. Create a Chroma vector store from the chunks.
    #    Pass in the chunks and the embeddings model.
    #    Chroma will embed every chunk and store the vectors locally.
    #
    # 2. Call .as_retriever() on the vector store.
    #    search_kwargs={"k": 3} means: return the 3 most relevant chunks.
    #
    # Hint:
    # vectorstore = Chroma.from_documents(?, embedding=?)
    # ============================================================

    vectorstore = None  
    retriever = None   

    # ============================================================

    return retriever


# --- Run this file directly to test your work ---
if __name__ == "__main__":
    retriever = build_retriever(PDF_PATH)

    if retriever is None:
        print("Retriever not created yet. Complete the YOUR CODE HERE section above.")
    else:
        test_query = "What is the minimum attendance required?"
        print(f"Test query: '{test_query}'\n")

        results = retriever.invoke(test_query)

        print(f"Retrieved {len(results)} chunks:\n")
        for i, doc in enumerate(results, 1):
            print(f"[Chunk {i}] (page {doc.metadata.get('page', '?')})")
            print(doc.page_content[:300])
            print()
