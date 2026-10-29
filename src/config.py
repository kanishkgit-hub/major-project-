from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Medical documents directory
MEDICAL_KNOWLEDGE_DIR = PROJECT_ROOT / "medical_knowledge"

# ChromaDB storage directory
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Ollama local API
OLLAMA_BASE_URL = "http://localhost:11434"

# AI models
EMBEDDING_MODEL = "nomic-embed-text"
GENERATION_MODEL = "llama3.2:3b"

# ChromaDB collection
CHROMA_COLLECTION_NAME = "cdss_medical_knowledge_v2"

# Retrieval settings
TOP_K_RESULTS = 4

# Text chunking settings
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200