"""
SOLUTION — File 3: Document Loading and Splitting
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "handbook.pdf"


def load_and_split(pdf_path: str):
    # SOLUTION Part 1: load the PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # SOLUTION Part 2: split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)

    return chunks


if __name__ == "__main__":
    chunks = load_and_split(PDF_PATH)
    print(f"Loaded PDF: {PDF_PATH}")
    print(f"Total chunks created: {len(chunks)}\n")
    print("--- First chunk ---")
    print(chunks[0].page_content)
    print("\n--- Metadata of first chunk ---")
    print(chunks[0].metadata)
