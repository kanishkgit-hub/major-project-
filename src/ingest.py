import hashlib

import chromadb
import fitz
import requests

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MEDICAL_KNOWLEDGE_DIR,
    OLLAMA_BASE_URL,
)


def create_chunks(text):
    """Split extracted PDF text into overlapping chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def get_embedding(text):
    """Generate an embedding through the local Ollama API."""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    embeddings = response.json()["embeddings"]

    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings.")

    return embeddings[0]


def get_disease_category(pdf_path):
    """Identify the category from the parent folder name."""

    return pdf_path.parent.name.lower()


def ingest_documents():
    """Extract, embed, and index all medical PDFs."""

    if not MEDICAL_KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Medical directory not found: {MEDICAL_KNOWLEDGE_DIR}"
        )

    pdf_files = list(
        MEDICAL_KNOWLEDGE_DIR.rglob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            "No PDFs found inside medical_knowledge."
        )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_indexed = 0

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            with fitz.open(pdf_path) as document:
                for page_number, page in enumerate(
                    document, start=1
                ):
                    text = page.get_text("text")

                    chunks = create_chunks(text)

                    for chunk_number, chunk in enumerate(
                        chunks, start=1
                    ):
                        # Stable IDs prevent duplicate chunks
                        # when ingestion is rerun.
                        chunk_hash = hashlib.sha256(
                            chunk.encode("utf-8")
                        ).hexdigest()[:16]

                        document_id = (
                            f"{pdf_path.relative_to(MEDICAL_KNOWLEDGE_DIR)}"
                            f"|page={page_number}"
                            f"|chunk={chunk_number}"
                            f"|hash={chunk_hash}"
                        )

                        embedding = get_embedding(chunk)

                        collection.upsert(
                            ids=[document_id],
                            documents=[chunk],
                            embeddings=[embedding],
                            metadatas=[{
                                "source": str(
                                    pdf_path.relative_to(
                                        MEDICAL_KNOWLEDGE_DIR
                                    )
                                ),
                                "page": page_number,
                                "disease": get_disease_category(
                                    pdf_path
                                ),
                            }],
                        )

                        total_indexed += 1

                        if total_indexed % 10 == 0:
                            print(
                                f"Indexed {total_indexed} chunks..."
                            )

        except Exception as exc:
            print(
                f"ERROR processing {pdf_path.name}: {exc}"
            )
            raise

    print("\nIngestion complete!")
    print(f"Chunks indexed in this run: {total_indexed}")
    print(f"Total chunks in database: {collection.count()}")


if __name__ == "__main__":
    ingest_documents()