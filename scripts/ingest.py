"""Build (or rebuild) the vector database from the handbook PDF.

Usage:  python -m scripts.ingest
"""

from __future__ import annotations

from app.config import get_settings
from app.embeddings import SentenceTransformerEmbedder
from app.rag import ingest_handbook
from app.vector_store import VectorStore


def main() -> None:
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    count = ingest_handbook(
        settings.handbook_path, embedder, store, settings.chunk_size, settings.chunk_overlap
    )
    print(f"Stored {count} chunks from {settings.handbook_path} in {settings.chroma_dir}")


if __name__ == "__main__":
    main()
