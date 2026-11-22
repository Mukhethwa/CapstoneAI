"""GeminiLLM retry / fallback behaviour, tested with a fake client (no network)."""

import json

import pytest
from google.genai import errors

from app.llm import GeminiLLM, LLMError
from app.vector_store import RetrievedChunk

CONTEXT = [RetrievedChunk("Total fees: R 38,950", 0.9, "handbook", page=16)]
GOOD_REPLY = json.dumps({"found": True, "answer": "R 38,950", "sources": [1]})


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, behaviour):
        self.behaviour = behaviour  # model name -> list of results (Exception or reply text)
        self.calls: list[str] = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        result = self.behaviour[model].pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


class FakeClient:
    def __init__(self, behaviour):
        self.models = FakeModels(behaviour)


def overloaded():
    return errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def make_llm(behaviour, **kwargs) -> GeminiLLM:
    llm = GeminiLLM("", "primary", fallback_models=["backup"], backoff_seconds=0, **kwargs)
    llm._client = FakeClient(behaviour)
    return llm


def test_missing_api_key_raises_llm_error():
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM("").generate("q", CONTEXT)


def test_retries_a_temporary_failure_then_succeeds():
    llm = make_llm({"primary": [overloaded(), GOOD_REPLY]})
    answer = llm.generate("fees?", CONTEXT)
    assert answer.found and answer.answer == "R 38,950" and answer.sources == [1]
    assert llm._client.models.calls == ["primary", "primary"]


def test_falls_back_to_next_model_when_primary_stays_overloaded():
    llm = make_llm({"primary": [overloaded()] * 3, "backup": [GOOD_REPLY]}, max_attempts=3)
    assert llm.generate("fees?", CONTEXT).answer == "R 38,950"
    assert llm._client.models.calls == ["primary"] * 3 + ["backup"]


def test_raises_when_every_model_is_unavailable():
    llm = make_llm({"primary": [overloaded()] * 2, "backup": [overloaded()] * 2}, max_attempts=2)
    with pytest.raises(LLMError, match="503"):
        llm.generate("fees?", CONTEXT)


def test_does_not_retry_permanent_errors():
    bad_key = errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})
    llm = make_llm({"primary": [bad_key]})
    with pytest.raises(LLMError, match="API key not valid"):
        llm.generate("fees?", CONTEXT)
    assert llm._client.models.calls == ["primary"]
