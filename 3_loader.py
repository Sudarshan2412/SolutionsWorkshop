"""
WORKSHOP FILE 3 — Document Loading and Splitting
=================================================
Concept: PyPDFLoader + RecursiveCharacterTextSplitter

An LLM can't read an entire PDF in one shot — the context window
is limited. LangChain solves this by loading the PDF and splitting
it into small overlapping chunks that can each be embedded and searched.

After completing this file you will understand:
  - How LangChain loads PDFs as a list of Document objects
  - Why we split documents into chunks before embedding
  - What chunk_size and chunk_overlap mean and why they matter
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "handbook.pdf"


def load_and_split(pdf_path: str):
    """
    Load a PDF and split it into chunks ready for embedding.
    Returns a list of Document objects (chunks).
    """

    # ============================================================
    # YOUR CODE HERE — Part 1: Load the PDF
    # ------------------------------------------------------------
    # Use PyPDFLoader to load the PDF file.
    # Call .load() on it to get a list of Document objects
    # (one Document per page).
    #
    # Hint:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # ============================================================

    # loader = None  
    # docs = None    

    # ============================================================
    # YOUR CODE HERE — Part 2: Split into chunks
    # ------------------------------------------------------------
    # Create a RecursiveCharacterTextSplitter and split the docs.
    #
    # chunk_size=500    → each chunk is at most 500 characters
    # chunk_overlap=50  → consecutive chunks share 50 characters
    #                     (overlap helps the LLM answer questions
    #                      that span a chunk boundary)
    #
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    # ============================================================

    # splitter = None  
    # chunks = None  

    # ============================================================

    return chunks


# --- Run this file directly to test your work ---
if __name__ == "__main__":
    chunks = load_and_split(PDF_PATH)

    if chunks is None:
        print("Chunks not created yet. Complete the YOUR CODE HERE sections above.")
    else:
        print(f"Loaded PDF: {PDF_PATH}")
        print(f"Total chunks created: {len(chunks)}\n")
        print("--- First chunk ---")
        print(chunks[0].page_content)
        print("\n--- Metadata of first chunk ---")
        print(chunks[0].metadata)
