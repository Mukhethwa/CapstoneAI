import pytest
from fastapi.testclient import TestClient

from app.config import NOT_FOUND_MESSAGE
from app.llm import LLMAnswer, LLMError
from app.main import create_app
from app.rag import RAGPipeline


@pytest.fixture
def client(pipeline):
    with TestClient(create_app(pipeline), raise_server_exceptions=False) as test_client:
        yield test_client


def test_ask_returns_answer_and_source(client):
    response = client.post("/ask", json={"question": "What are the total fees?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["found"] is True
    assert body["source"] == "Student Handbook - Page 16"
    assert body["sources"] == ["Student Handbook - Page 16"]
    assert "R 38,950" in body["answer"]


def test_ask_website_question_returns_url_source(client):
    response = client.post("/ask", json={"question": "What courses and bootcamps does ZAIO offer?"})
    assert response.status_code == 200
    assert response.json()["source"] == "https://www.zaio.io/bootcamps"


def test_ask_unknown_question_returns_not_found(client):
    response = client.post("/ask", json={"question": "zebra quantum volcano"})
    assert response.status_code == 200
    assert response.json() == {
        "answer": NOT_FOUND_MESSAGE, "source": None, "sources": [], "found": False
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json": {}},
        {"json": {"question": ""}},
        {"json": {"question": "   "}},
        {"json": {"question": 123}},
        {"json": {"question": "x" * 1001}},
        {"json": ["not", "an", "object"]},
        {"content": "this is not json", "headers": {"content-type": "application/json"}},
    ],
)
def test_invalid_requests_return_json_error(client, kwargs):
    response = client.post("/ask", **kwargs)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "Invalid request"
    assert body["details"]


def test_wrong_method_and_unknown_route_return_json(client):
    assert client.get("/ask").json()["error"] == "Method Not Allowed"
    assert client.get("/nope").status_code == 404
    assert "error" in client.get("/nope").json()


def test_llm_failure_returns_502(embedder, store):
    class BrokenLLM:
        def generate(self, question, context) -> LLMAnswer:
            raise LLMError("quota exceeded")

    pipeline = RAGPipeline(embedder, store, BrokenLLM(), top_k=3, min_similarity=0.2)
    with TestClient(create_app(pipeline)) as client:
        response = client.post("/ask", json={"question": "What are the total fees?"})
    assert response.status_code == 502
    assert response.json()["error"] == "The language model is unavailable"


def test_unexpected_error_returns_500_json(embedder, store):
    class CrashingLLM:
        def generate(self, question, context) -> LLMAnswer:
            raise RuntimeError("boom")

    pipeline = RAGPipeline(embedder, store, CrashingLLM(), top_k=3, min_similarity=0.2)
    with TestClient(create_app(pipeline), raise_server_exceptions=False) as client:
        response = client.post("/ask", json={"question": "What are the total fees?"})
    assert response.status_code == 500
    assert response.json() == {"error": "Internal server error", "details": None}


def test_health_reports_chunk_count(client):
    assert client.get("/health").json() == {
        "status": "ok", "chunks": 4, "by_source": {"handbook": 2, "website": 2}
    }
