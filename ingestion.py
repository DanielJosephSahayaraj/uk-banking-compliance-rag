import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retriever import ensure_collection, store_chunks
from config import COLLECTION_NAME

# ── Settings ──
DOCUMENTS_FOLDER = "documents"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 200


def load_pdfs_from_folder(folder: str) -> list:
    """Load all PDFs from the documents folder."""
    all_documents = []
    folder_path = Path(folder)

    if not folder_path.exists():
        print(f"Folder '{folder}' not found!")
        return []

    pdf_files = list(folder_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in '{folder}'!")
        return []

    print(f"Found {len(pdf_files)} PDF(s) to load...")

    for pdf_path in pdf_files:
        try:
            print(f"Loading: {pdf_path.name}")
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            # ── Add source metadata to each page ──
            for doc in documents:
                doc.metadata["source_file"] = pdf_path.name
                doc.metadata["document_type"] = get_document_type(pdf_path.name)

            all_documents.extend(documents)
            print(f"  → Loaded {len(documents)} pages from {pdf_path.name}")

        except Exception as e:
            print(f"  → Failed to load {pdf_path.name}: {e}")

    print(f"\nTotal pages loaded: {len(all_documents)}")
    return all_documents


def get_document_type(filename: str) -> str:
    """Tag document with its regulatory type."""
    filename = filename.lower()

    if "consumer_duty" in filename or "ps22" in filename:
        return "FCA Consumer Duty"
    elif "financial_crime" in filename or "aml" in filename:
        return "FCA Financial Crime"
    elif "business_plan" in filename:
        return "FCA Business Plan"
    elif "pra" in filename or "prudential" in filename:
        return "PRA Rulebook"
    elif "basel" in filename:
        return "Basel III"
    elif "gdpr" in filename:
        return "GDPR"
    else:
        return "FCA General"


def chunk_documents(documents: list) -> tuple[list[str], list[dict]]:
    """Split documents into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)

    texts = []
    metadata = []

    for chunk in chunks:
        texts.append(chunk.page_content)
        metadata.append({
            "source_file": chunk.metadata.get("source_file", "unknown"),
            "document_type": chunk.metadata.get("document_type", "unknown"),
            "page": chunk.metadata.get("page", 0)
        })

    print(f"Created {len(texts)} chunks from {len(documents)} pages")
    return texts, metadata


def ingest_documents(folder: str = DOCUMENTS_FOLDER):
    """Full ingestion pipeline — load, chunk, embed, store."""

    print("\n" + "="*50)
    print("UK BANKING COMPLIANCE RAG — DOCUMENT INGESTION")
    print("="*50 + "\n")

    # ── Step 1: Ensure Qdrant collection exists ──
    print("Step 1: Setting up Qdrant collection...")
    ensure_collection()

    # ── Step 2: Load PDFs ──
    print("\nStep 2: Loading PDFs...")
    documents = load_pdfs_from_folder(folder)

    if not documents:
        print("No documents to ingest. Add PDFs to the documents folder.")
        return

    # ── Step 3: Chunk documents ──
    print("\nStep 3: Chunking documents...")
    texts, metadata = chunk_documents(documents)

    # ── Step 4: Store in Qdrant ──
    print("\nStep 4: Embedding and storing in Qdrant...")
    print("(This may take a few minutes for large documents...)")

    # Store in batches to avoid memory issues
    batch_size = 50
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]
        store_chunks(batch_texts, batch_meta)
        batch_num = (i // batch_size) + 1
        print(f"  → Batch {batch_num}/{total_batches} stored")

    print("\n" + "="*50)
    print(f"INGESTION COMPLETE!")
    print(f"Total chunks stored: {len(texts)}")
    print(f"Collection: {COLLECTION_NAME}")
    print("="*50 + "\n")


if __name__ == "__main__":
    ingest_documents()