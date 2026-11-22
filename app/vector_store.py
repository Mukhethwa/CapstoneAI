"""ChromaDB vector store holding handbook and website chunks in a single collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.document_processor import Chunk, source_label


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    similarity: float  # cosine similarity, higher is more relevant
    source: str = "handbook"
    page: int | None = None
    url: str | None = None
    title: str | None = None

    @property
    def label(self) -> str:
        return source_label(self.source, self.page, self.url)


def _metadata(chunk: Chunk) -> dict:
    """Chroma metadata cannot hold None, so only set the fields a chunk actually has."""
    meta: dict = {"source": chunk.source}
    if chunk.page is not None:
        meta["page"] = chunk.page
    if chunk.url:
        meta["url"] = chunk.url
    if chunk.title:
        meta["title"] = chunk.title
    return meta


class VectorStore:
    def __init__(self, persist_dir: str | Path | None, collection_name: str = "knowledge_base"):
        if persist_dir is None:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._name = collection_name
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self._name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def count_by_source(self) -> dict[str, int]:
        metadatas = self._collection.get(include=["metadatas"])["metadatas"] or []
        return dict(Counter(m.get("source", "unknown") for m in metadatas))

    def reset(self) -> None:
        """Delete every stored chunk (used when rebuilding the knowledge base)."""
        self._client.delete_collection(self._name)
        self._collection = self._get_collection()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], batch_size: int = 500) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            self._collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=embeddings[start:start + batch_size],
                metadatas=[_metadata(c) for c in batch],
            )

    def query(
        self, embedding: list[float], top_k: int = 6, source: str | None = None
    ) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks, optionally from one source only."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                text=doc,
                similarity=1.0 - float(dist),
                source=meta.get("source", "handbook"),
                page=int(meta["page"]) if "page" in meta else None,
                url=meta.get("url"),
                title=meta.get("title"),
            )
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
