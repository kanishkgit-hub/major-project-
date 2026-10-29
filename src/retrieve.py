import re
import requests
import chromadb
import numpy as np

from src.config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)


# --------------------------------------------------
# 1. CONNECT TO CHROMADB
# --------------------------------------------------

def get_collection():
    """Connect to the existing medical knowledge database."""

    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    return client.get_collection(
        name=CHROMA_COLLECTION_NAME
    )


# --------------------------------------------------
# 2. GENERATE QUERY EMBEDDING
# --------------------------------------------------

def get_embedding(text: str) -> list[float]:
    """Generate an embedding using local Ollama."""

    response = requests.post(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()
    embeddings = data.get("embeddings")

    if not embeddings or not embeddings[0]:
        raise RuntimeError(
            "Ollama returned no embeddings. "
            "Check the embedding model in src/config.py."
        )

    return embeddings[0]


# --------------------------------------------------
# 3. NORMALIZE CATEGORY VALUES
# --------------------------------------------------

def normalize_category(value) -> str:
    """
    Normalize category strings.

    Examples:
        'diabetes' -> 'diabetes'
        'Diabetes' -> 'diabetes'
        ' diabetes ' -> 'diabetes'
    """

    if value is None:
        return ""

    return re.sub(
        r"[^a-z]",
        "",
        str(value).casefold().strip(),
    )


# --------------------------------------------------
# 4. RETRIEVE RELEVANT MEDICAL PASSAGES
# --------------------------------------------------

def retrieve_documents(
    question: str,
    top_k: int = 4,
    disease: str | None = None,
) -> list[dict]:
    """
    Retrieve medical passages using cosine similarity.

    Category filtering happens in Python.
    ChromaDB metadata filters are not used.

    Returns dictionaries compatible with rag.py.
    """

    if not question or not question.strip():
        return []

    collection = get_collection()
    total_chunks = collection.count()

    if total_chunks == 0:
        print("ERROR: The ChromaDB collection is empty.")
        return []

    print(f"Database chunks: {total_chunks}")

    requested_category = (
        normalize_category(disease)
        if disease
        else None
    )

    if requested_category and requested_category not in {
        "diabetes",
        "heart",
        "kidney",
    }:
        raise ValueError(
            "Disease must be diabetes, heart, kidney, or None."
        )

    # Retrieve ALL stored records.
    # This avoids the problem of retrieving only the top 10
    # records before applying the disease category filter.

    records = collection.get(
        limit=total_chunks,
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ],
    )

    ids = records.get("ids") or []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    stored_embeddings = records.get("embeddings")

    print(f"Records loaded: {len(ids)}")

    if not ids or not stored_embeddings is not None:
        raise RuntimeError(
            "No stored embeddings were returned by ChromaDB. "
            "Check the collection and its stored embeddings."
        )

    if len(documents) != len(ids):
        raise RuntimeError(
            "The number of documents does not match the number of IDs."
        )

    if len(metadatas) != len(ids):
        raise RuntimeError(
            "The number of metadata records does not match the IDs."
        )

    # Count categories from the actual stored records.
    category_counts = {}

    for metadata in metadatas:
        metadata = metadata or {}

        category = normalize_category(
            metadata.get("disease")
        )

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    print("Category counts:", category_counts)

    # Filter records BEFORE calculating similarity.
    candidates = []

    for record_id, document, metadata, vector in zip(
        ids,
        documents,
        metadatas,
        stored_embeddings,
    ):
        metadata = metadata or {}

        category = normalize_category(
            metadata.get("disease")
        )

        if requested_category:
            if category != requested_category:
                continue

        if not document or vector is None:
            continue

        candidates.append({
            "id": record_id,
            "text": document,
            "source": metadata.get(
                "source", "Unknown source"
            ),
            "page": metadata.get(
                "page", "Unknown"
            ),
            "disease": category,
            "embedding": vector,
        })

    print("Requested category:", requested_category)
    print("Matching candidates:", len(candidates))

    if not candidates:
        print(
            "No matching records found. "
            "Inspect the category counts above."
        )
        return []

    # Generate the query embedding.
    query_vector = np.asarray(
        get_embedding(question),
        dtype=np.float32,
    )

    # Prepare document vectors.
    document_vectors = np.asarray(
        [item["embedding"] for item in candidates],
        dtype=np.float32,
    )

    if document_vectors.ndim != 2:
        raise RuntimeError(
            "Stored embeddings have an unexpected shape."
        )

    if document_vectors.shape[1] != query_vector.shape[0]:
        raise RuntimeError(
            "Embedding dimensions do not match. "
            f"Query: {query_vector.shape[0]}, "
            f"Documents: {document_vectors.shape[1]}. "
            "Check whether the same embedding model was "
            "used during ingestion and retrieval."
        )

    # Calculate cosine similarity.
    query_norm = np.linalg.norm(query_vector)

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1,
    )

    if query_norm == 0:
        raise RuntimeError(
            "The query embedding has zero magnitude."
        )

    similarities = (
        document_vectors @ query_vector
    ) / (
        np.maximum(document_norms, 1e-12) * query_norm
    )

    # Higher cosine similarity means a closer match.
    ranked_indices = np.argsort(-similarities)

    selected = []

    for index in ranked_indices[:max(1, top_k)]:
        item = candidates[int(index)]

        # Convert cosine similarity into cosine distance.
        similarity = float(similarities[index])

        selected.append({
            "text": item["text"],
            "source": item["source"],
            "page": item["page"],
            "disease": item["disease"],
            "distance": 1.0 - similarity,
        })

    print(f"Returning {len(selected)} passages.")

    return selected


# --------------------------------------------------
# 5. INTERACTIVE TEST
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CDSS MEDICAL DOCUMENT RETRIEVAL")
    print("=" * 60)

    question = input(
        "\nAsk a question about your medical documents: "
    ).strip()

    disease = input(
        "Category (diabetes/heart/kidney, "
        "or Enter for all): "
    ).strip().lower()

    if disease not in ("", "diabetes", "heart", "kidney"):
        print("Invalid category. Searching all documents.")
        disease = ""

    try:
        results = retrieve_documents(
            question=question,
            disease=disease or None,
            top_k=5,
        )

        print("\n" + "=" * 60)
        print("RETRIEVED RESULTS")
        print("=" * 60)

        if not results:
            print("No matching passages found.")

        for index, result in enumerate(results, start=1):
            print(f"\n--- RESULT {index} ---")
            print("Source:", result["source"])
            print("Page:", result["page"])
            print("Category:", result["disease"])
            print("Distance:", round(result["distance"], 4))
            print("Text:", result["text"][:500])

    except requests.RequestException as error:
        print("\nOllama connection error:", error)

    except Exception as error:
        print("\nRetrieval error:", error)
        raise