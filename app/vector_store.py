"""ChromaDB vector store holding the handbook chunks and their embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from app.document_processor import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page: int
    similarity: float  # cosine similarity, higher is more relevant


class VectorStore:
    def __init__(self, persist_dir: str | Path | None, collection_name: str = "handbook"):
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

    def reset(self) -> None:
        """Delete every stored chunk (used when re-ingesting the handbook)."""
        self._client.delete_collection(self._name)
        self._collection = self._get_collection()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[{"page": c.page} for c in chunks],
        )

    def query(self, embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(text=doc, page=int(meta["page"]), similarity=1.0 - float(dist))
            for doc, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]
