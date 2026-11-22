import json

import pytest

from app.config import NOT_FOUND_MESSAGE
from app.llm import LLMError, build_prompt, parse_answer
from app.rag import RAGPipeline, format_source
from app.vector_store import RetrievedChunk

from tests.conftest import FakeLLM


def test_vector_store_returns_most_similar_chunk_first(store, embedder):
    results = store.query(embedder.embed_query("How much are the total fees?"), top_k=3)
    assert results[0].page == 16
    assert results[0].similarity >= results[-1].similarity


def test_retrieve_filters_out_irrelevant_chunks(pipeline):
    assert pipeline.retrieve("zebra quantum volcano") == []


def test_ask_answers_from_retrieved_context(pipeline, fake_llm):
    result = pipeline.ask("What are the total fees?")
    assert result.found is True
    assert "R 38,950" in result.answer
    assert result.source == "Page 16"
    question, context = fake_llm.calls[0]
    assert question == "What are the total fees?"
    assert context and context[0].page == 16


def test_ask_returns_not_found_without_calling_llm_when_nothing_relevant(pipeline, fake_llm):
    result = pipeline.ask("zebra quantum volcano")
    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.source is None
    assert fake_llm.calls == []


def test_ask_returns_not_found_when_llm_says_context_is_insufficient(embedder, store):
    pipeline = RAGPipeline(embedder, store, FakeLLM(found=False), top_k=2, min_similarity=0.2)
    result = pipeline.ask("What are the total fees?")
    assert result.found is False
    assert result.answer == NOT_FOUND_MESSAGE


def test_ask_ignores_pages_the_llm_invents(embedder, store):
    pipeline = RAGPipeline(embedder, store, FakeLLM(pages=[99]), top_k=2, min_similarity=0.2)
    result = pipeline.ask("What are the total fees?")
    assert result.source == "Page 16"


@pytest.mark.parametrize(
    "pages,expected",
    [([], None), ([12], "Page 12"), ([17, 16, 16], "Pages 16, 17")],
)
def test_format_source(pages, expected):
    assert format_source(pages) == expected


def test_build_prompt_includes_question_and_page_labels():
    prompt = build_prompt("Q?", [RetrievedChunk(text="ctx", page=3, similarity=0.9)])
    assert "[Page 3]" in prompt and "ctx" in prompt and "Q?" in prompt


def test_parse_answer_handles_plain_and_fenced_json():
    payload = {"found": True, "answer": " R 38,950 ", "pages": [16]}
    assert parse_answer(json.dumps(payload)).answer == "R 38,950"
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    assert parse_answer(fenced).pages == [16]


def test_parse_answer_rejects_garbage():
    with pytest.raises(LLMError):
        parse_answer("not json at all")
