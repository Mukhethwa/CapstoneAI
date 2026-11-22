"""Build (or rebuild) the vector database from the handbook PDF and the ZAIO website.

Usage:  python -m scripts.ingest [--recrawl]
"""

from __future__ import annotations

import argparse
import logging

from app.config import get_settings
from app.embeddings import SentenceTransformerEmbedder
from app.rag import build_knowledge_base, load_or_crawl_website
from app.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recrawl", action="store_true",
                        help="crawl the website again instead of using the saved snapshot")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    if args.recrawl and settings.website_snapshot.exists():
        settings.website_snapshot.unlink()
    web_pages = load_or_crawl_website(
        settings.website_snapshot, settings.website_url, settings.website_max_pages
    )
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    counts = build_knowledge_base(
        settings.handbook_path, web_pages, embedder, store,
        settings.chunk_size, settings.chunk_overlap,
    )
    print(f"Stored {counts['handbook']} handbook chunks and {counts['website']} website chunks "
          f"in {settings.chroma_dir}")


if __name__ == "__main__":
    main()
