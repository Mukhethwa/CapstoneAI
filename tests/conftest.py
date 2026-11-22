"""Shared test fixtures: lightweight fakes so tests run offline and fast."""

from __future__ import annotations

import hashlib
import math
import re
import uuid

import pytest

from app.document_processor import Chunk
from app.llm import LLMAnswer
from app.rag import RAGPipeline
from app.vector_store import RetrievedChunk, VectorStore

SAMPLE_CHUNKS = [
    Chunk(id="c1", page=16, text="FEES Total Fees: R 38,950. Fees must be paid in full before orientation day."),
    Chunk(id="c2", page=11, text="LIVE CLASSES We will have live classes every Thursday 6 pm to 8 pm and Tuesday 9 am to 11 am."),
    Chunk(id="c3", page=5, text="HARDWARE REQUIREMENTS At least 4 or 8 GB of RAM is required for your laptop."),
]


class HashEmbedder:
    """Deterministic bag-of-words embedding: similar wording -> similar vectors."""

    dim = 256

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeLLM:
    """Answers with the first retrieved chunk, or reports not found when told to."""

    def __init__(self, found: bool = True, pages: list[int] | None = None):
        self.found = found
        self.pages = pages
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, question: str, context: list[RetrievedChunk]) -> LLMAnswer:
        self.calls.append((question, context))
        if not self.found:
            return LLMAnswer(found=False)
        pages = self.pages if self.pages is not None else [context[0].page]
        return LLMAnswer(found=True, answer=context[0].text, pages=pages)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def store(embedder) -> VectorStore:
    vs = VectorStore(None, collection_name=f"test-{uuid.uuid4().hex}")
    vs.add_chunks(SAMPLE_CHUNKS, embedder.embed_documents([c.text for c in SAMPLE_CHUNKS]))
    return vs


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def pipeline(embedder, store, fake_llm) -> RAGPipeline:
    return RAGPipeline(embedder, store, fake_llm, top_k=2, min_similarity=0.2)
