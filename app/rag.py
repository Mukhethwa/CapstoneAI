"""Build the combined knowledge base, then retrieve across both sources and answer questions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.config import NOT_FOUND_MESSAGE
from app.document_processor import HANDBOOK, WEBSITE, chunk_pages, chunk_web_pages, load_pdf
from app.embeddings import Embedder
from app.llm import LLM
from app.vector_store import RetrievedChunk, VectorStore
from app.web_crawler import WebPage, WebsiteCrawler, load_pages, save_pages

logger = logging.getLogger(__name__)


@dataclass
class AskResult:
    answer: str
    source: str | None  # the most appropriate single source
    found: bool
    sources: list[str] = field(default_factory=list)  # every source the answer used
    context: list[RetrievedChunk] = field(default_factory=list)


def unique_labels(chunks: list[RetrievedChunk]) -> list[str]:
    labels: list[str] = []
    for chunk in chunks:
        if chunk.label not in labels:
            labels.append(chunk.label)
    return labels


def build_knowledge_base(
    pdf_path: str | Path,
    web_pages: list[WebPage],
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> dict[str, int]:
    """Chunk, embed and store the handbook and website pages in one collection.

    Returns the number of chunks stored per source.
    """
    handbook_pages = load_pdf(pdf_path)
    chunks = chunk_pages(handbook_pages, chunk_size, chunk_overlap)
    chunks += chunk_web_pages(web_pages, chunk_size, chunk_overlap)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.reset()
    store.add_chunks(chunks, embeddings)
    counts = {
        HANDBOOK: sum(c.source == HANDBOOK for c in chunks),
        WEBSITE: sum(c.source == WEBSITE for c in chunks),
    }
    logger.info(
        "Stored %d handbook chunks (%d pages) and %d website chunks (%d pages)",
        counts[HANDBOOK], len(handbook_pages), counts[WEBSITE], len(web_pages),
    )
    return counts


def load_or_crawl_website(snapshot: str | Path, url: str, max_pages: int) -> list[WebPage]:
    """Use the saved website snapshot, crawling the site first if there is none."""
    try:
        return load_pages(snapshot)
    except FileNotFoundError:
        logger.info("No website snapshot at %s, crawling %s", snapshot, url)
        pages = WebsiteCrawler(url, max_pages=max_pages).crawl()
        save_pages(pages, snapshot)
        return pages


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: LLM,
        top_k: int = 6,
        min_similarity: float = 0.3,
        per_source: int = 2,
    ):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.per_source = per_source

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Embed the question and search each knowledge source.

        The website has far more chunks than the handbook, so a single global search can push
        relevant handbook passages out of the results. Each source is therefore searched on its
        own and the best ``per_source`` matches from every source are always kept (if they pass
        the similarity floor); the remaining slots go to the best matches overall.
        """
        query_embedding = self.embedder.embed_query(question)
        ranked: dict[str, list[RetrievedChunk]] = {}
        for source in (HANDBOOK, WEBSITE):
            hits = self.store.query(query_embedding, self.top_k, source=source)
            ranked[source] = [h for h in hits if h.similarity >= self.min_similarity]

        picked = [hit for hits in ranked.values() for hit in hits[: self.per_source]]
        rest = sorted((h for hits in ranked.values() for h in hits if h not in picked),
                      key=lambda h: h.similarity, reverse=True)
        picked += rest[: max(self.top_k - len(picked), 0)]
        return sorted(picked, key=lambda h: h.similarity, reverse=True)

    def ask(self, question: str) -> AskResult:
        context = self.retrieve(question)
        if not context:
            return AskResult(answer=NOT_FOUND_MESSAGE, source=None, found=False)

        reply = self.llm.generate(question, context)
        if not reply.found or not reply.answer:
            return AskResult(answer=NOT_FOUND_MESSAGE, source=None, found=False, context=context)

        # The model cites excerpts by number (1-based); ignore numbers that don't exist.
        cited = [context[n - 1] for n in reply.sources if 1 <= n <= len(context)]
        labels = unique_labels(cited or context[:1])
        return AskResult(
            answer=reply.answer, source=labels[0], found=True, sources=labels, context=context
        )
