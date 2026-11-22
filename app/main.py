"""FastAPI service exposing POST /ask over the handbook + website knowledge base (n8n ready)."""

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
    question: str = Field(..., max_length=1000, examples=["What courses does ZAIO offer?"])

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class AskResponse(BaseModel):
    answer: str
    source: str | None = Field(
        None, description="Most appropriate source: a ZAIO URL or 'Student Handbook - Page N'",
        examples=["https://www.zaio.io/programmes", "Student Handbook - Page 18"],
    )
    sources: list[str] = Field(default_factory=list, description="Every source the answer used")
    found: bool


class ErrorResponse(BaseModel):
    error: str
    details: list[str] | None = None


def build_default_pipeline() -> RAGPipeline:
    """Wire up the real components and build the knowledge base if the index is empty."""
    from app.embeddings import SentenceTransformerEmbedder
    from app.llm import GeminiLLM
    from app.rag import build_knowledge_base, load_or_crawl_website
    from app.vector_store import VectorStore

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    if store.count() == 0:
        logger.info("Vector store is empty, building the knowledge base")
        web_pages = load_or_crawl_website(
            settings.website_snapshot, settings.website_url, settings.website_max_pages
        )
        build_knowledge_base(
            settings.handbook_path, web_pages, embedder, store,
            settings.chunk_size, settings.chunk_overlap,
        )
    llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, settings.gemini_fallback_models)
    return RAGPipeline(embedder, store, llm, settings.top_k, settings.min_similarity)


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_default_pipeline()
        yield

    app = FastAPI(
        title="ZAIO Knowledge Assistant",
        description="Answers questions using the ZAIO Student Handbook and the ZAIO website.",
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
        store = request.app.state.pipeline.store
        return {"status": "ok", "chunks": store.count(), "by_source": store.count_by_source()}

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
        return AskResponse(
            answer=result.answer, source=result.source, sources=result.sources, found=result.found
        )

    return app


app = create_app()
