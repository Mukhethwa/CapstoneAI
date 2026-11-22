import json

import pytest

from app.config import NOT_FOUND_MESSAGE
from app.document_processor import HANDBOOK, WEBSITE
from app.llm import LLMError, build_prompt, parse_answer
from app.rag import RAGPipeline, build_knowledge_base
from app.vector_store import RetrievedChunk, VectorStore
from app.web_crawler import WebPage

from tests.conftest import FakeLLM


def test_vector_store_searches_both_sources(store, embedder):
    handbook = store.query(embedder.embed_query("total fees orientation"), top_k=1)
    website = store.query(embedder.embed_query("refund policy cancel deposit"), top_k=1)
    assert handbook[0].source == HANDBOOK and handbook[0].page == 16
    assert website[0].source == WEBSITE and website[0].url == "https://www.zaio.io/refund-policy"
    assert website[0].title == "Refund Policy"


def test_vector_store_can_filter_by_source(store, embedder):
    results = store.query(embedder.embed_query("fees"), top_k=4, source=WEBSITE)
    assert results and all(r.source == WEBSITE for r in results)


def test_vector_store_counts_chunks_per_source(store):
    assert store.count_by_source() == {HANDBOOK: 2, WEBSITE: 2}


def test_retrieved_chunk_labels():
    assert RetrievedChunk("t", 0.9, HANDBOOK, page=18).label == "Student Handbook - Page 18"
    assert RetrievedChunk("t", 0.9, WEBSITE, url="https://www.zaio.io/x").label == "https://www.zaio.io/x"


def test_retrieve_keeps_best_matches_from_each_source(embedder):
    """A small source must not be crowded out by a much larger one."""
    from app.document_processor import Chunk

    chunks = [Chunk(id="h", source=HANDBOOK, page=16, text="bootcamp fees are R 38,950")]
    chunks += [Chunk(id=f"w{i}", source=WEBSITE, url=f"https://www.zaio.io/{i}",
                     text=f"bootcamp fees and payment plans option {i} fees bootcamp")
               for i in range(10)]
    store = VectorStore(None, collection_name="kb-test-diversity")
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    pipeline = RAGPipeline(embedder, store, FakeLLM(), top_k=3, min_similarity=0.1, per_source=1)
    context = pipeline.retrieve("bootcamp fees payment plans")
    assert len(context) == 3
    assert {c.source for c in context} == {HANDBOOK, WEBSITE}
    assert [c.similarity for c in context] == sorted((c.similarity for c in context), reverse=True)


def test_retrieve_filters_out_irrelevant_chunks(pipeline):
    assert pipeline.retrieve("zebra quantum volcano") == []


def test_ask_answers_from_handbook_with_page_source(pipeline, fake_llm):
    result = pipeline.ask("What are the total fees?")
    assert result.found is True
    assert "R 38,950" in result.answer
    assert result.source == "Student Handbook - Page 16"
    assert result.sources == ["Student Handbook - Page 16"]
    assert fake_llm.calls[0][0] == "What are the total fees?"


def test_ask_answers_from_website_with_url_source(pipeline):
    result = pipeline.ask("What is the refund policy if I cancel?")
    assert result.found is True
    assert result.source == "https://www.zaio.io/refund-policy"


def test_ask_returns_all_cited_sources_in_order(embedder, store):
    pipeline = RAGPipeline(embedder, store, FakeLLM(sources=[2, 1, 2]), top_k=3, min_similarity=0.0)
    result = pipeline.ask("What are the total fees?")
    context_labels = [c.label for c in result.context]
    assert result.sources == [context_labels[1], context_labels[0]]
    assert result.source == context_labels[1]


def test_ask_ignores_excerpt_numbers_the_llm_invents(embedder, store):
    pipeline = RAGPipeline(embedder, store, FakeLLM(sources=[99]), top_k=3, min_similarity=0.2)
    result = pipeline.ask("What are the total fees?")
    assert result.source == "Student Handbook - Page 16"


def test_ask_returns_fallback_without_calling_llm_when_nothing_relevant(pipeline, fake_llm):
    result = pipeline.ask("zebra quantum volcano")
    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.source is None and result.sources == []
    assert fake_llm.calls == []


def test_ask_returns_fallback_when_llm_says_context_is_insufficient(embedder, store):
    pipeline = RAGPipeline(embedder, store, FakeLLM(found=False), top_k=3, min_similarity=0.2)
    result = pipeline.ask("What are the total fees?")
    assert result.found is False
    assert result.answer == "I could not find that information in the available knowledge base."


def test_build_knowledge_base_stores_both_sources(embedder, tmp_path, monkeypatch):
    from app import rag
    from app.document_processor import Page

    monkeypatch.setattr(rag, "load_pdf", lambda path: [Page(1, "Handbook intro text."),
                                                       Page(2, "More handbook text.")])
    pages = [WebPage("https://www.zaio.io/a", "A", "Website text about courses.")]
    store = VectorStore(None, collection_name="kb-test-build")
    counts = build_knowledge_base("unused.pdf", pages, embedder, store)
    assert counts == {HANDBOOK: 2, WEBSITE: 1}
    assert store.count_by_source() == {HANDBOOK: 2, WEBSITE: 1}
    # Rebuilding replaces the old content instead of duplicating it.
    build_knowledge_base("unused.pdf", pages, embedder, store)
    assert store.count() == 3


def test_build_prompt_numbers_excerpts_and_names_their_sources():
    prompt = build_prompt("Q?", [
        RetrievedChunk("handbook ctx", 0.9, HANDBOOK, page=3),
        RetrievedChunk("web ctx", 0.8, WEBSITE, url="https://www.zaio.io/x"),
    ])
    assert "[1] Source: Student Handbook - Page 3" in prompt
    assert "[2] Source: https://www.zaio.io/x" in prompt
    assert "Q?" in prompt


def test_parse_answer_handles_plain_and_fenced_json():
    payload = {"found": True, "answer": " R 38,950 ", "sources": [1]}
    assert parse_answer(json.dumps(payload)).answer == "R 38,950"
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert parse_answer(fenced).sources == [1]


def test_parse_answer_rejects_garbage():
    with pytest.raises(LLMError):
        parse_answer("not json at all")
