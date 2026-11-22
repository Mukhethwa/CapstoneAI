"""Part 3 + 5: FastAPI service exposing POST /ask with JSON in and JSON out (n8n ready)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app.llm import LLMError
from app.rag import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., max_length=1000, examples=["What is the attendance requirement?"])

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class AskResponse(BaseModel):
    answer: str
    source: str | None = Field(None, examples=["Page 12"])
    found: bool


class ErrorResponse(BaseModel):
    error: str
    details: list[str] | None = None


def build_default_pipeline() -> RAGPipeline:
    """Wire up the real components and ingest the handbook if the index is empty."""
    from app.embeddings import SentenceTransformerEmbedder
    from app.llm import GeminiLLM
    from app.rag import ingest_handbook
    from app.vector_store import VectorStore

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    if store.count() == 0:
        logger.info("Vector store is empty, ingesting %s", settings.handbook_path)
        ingest_handbook(
            settings.handbook_path, embedder, store, settings.chunk_size, settings.chunk_overlap
        )
    llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    return RAGPipeline(embedder, store, llm, settings.top_k, settings.min_similarity)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_default_pipeline()
        yield

    app = FastAPI(
        title="Handbook RAG Assistant",
        description="Answers student questions using the bootcamp handbook.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            location = ".".join(str(part) for part in err.get("loc", []) if part != "body")
            details.append(f"{location or 'body'}: {err.get('msg')}")
        body = ErrorResponse(error="Invalid request", details=details)
        return JSONResponse(status_code=422, content=body.model_dump())

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code, content=ErrorResponse(error=str(exc.detail)).model_dump()
        )

    @app.exception_handler(Exception)
    async def unexpected_error(_: Request, exc: Exception):
        logger.exception("Unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500, content=ErrorResponse(error="Internal server error").model_dump()
        )

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok", "chunks": request.app.state.pipeline.store.count()}

    @app.post(
        "/ask",
        response_model=AskResponse,
        responses={422: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    )
    def ask(payload: AskRequest, request: Request):
        try:
            result = request.app.state.pipeline.ask(payload.question)
        except LLMError as exc:
            logger.error("LLM error: %s", exc)
            return JSONResponse(
                status_code=502,
                content=ErrorResponse(error="The language model is unavailable",
                                      details=[str(exc)]).model_dump(),
            )
        return AskResponse(answer=result.answer, source=result.source, found=result.found)

    return app


app = create_app()
