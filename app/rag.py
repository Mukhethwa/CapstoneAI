"""Part 1 + 2: ingest the handbook into the vector store, then retrieve and answer questions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import NOT_FOUND_MESSAGE
from app.document_processor import chunk_pages, load_pdf
from app.embeddings import Embedder
from app.llm import LLM
from app.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class AskResult:
    answer: str
    source: str | None
    found: bool
    context: list[RetrievedChunk] = field(default_factory=list)


def format_source(pages: list[int]) -> str | None:
    unique = sorted(set(pages))
    if not unique:
        return None
    if len(unique) == 1:
        return f"Page {unique[0]}"
    return "Pages " + ", ".join(str(p) for p in unique)


def ingest_handbook(
    pdf_path: str | Path,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> int:
    """Load, chunk, embed and store the handbook. Returns the number of chunks stored."""
    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.reset()
    store.add_chunks(chunks, embeddings)
    logger.info("Ingested %d chunks from %d pages of %s", len(chunks), len(pages), pdf_path)
    return len(chunks)


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        top_k: int = 4,
        min_similarity: float = 0.25,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Embed the question and return the most relevant chunks above the similarity floor."""
        query_embedding = self.embedder.embed_query(question)
        results = self.store.query(query_embedding, self.top_k)
        return [r for r in results if r.similarity >= self.min_similarity]

    def ask(self, question: str) -> AskResult:
        context = self.retrieve(question)
        if not context:
            return AskResult(answer=NOT_FOUND_MESSAGE, source=None, found=False)

        reply = self.llm.generate(question, context)
        if not reply.found or not reply.answer:
            return AskResult(answer=NOT_FOUND_MESSAGE, source=None, found=False, context=context)

        # Only trust page numbers that were actually in the retrieved context.
        retrieved_pages = {c.page for c in context}
        pages = [p for p in reply.pages if p in retrieved_pages] or [context[0].page]
        return AskResult(
            answer=reply.answer, source=format_source(pages), found=True, context=context
        )
