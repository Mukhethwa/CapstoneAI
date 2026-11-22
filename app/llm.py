"""Answer generation with Google Gemini, grounded only in retrieved handbook context."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from app.vector_store import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful assistant for students of the Zaio Full-Stack AI Engineer Bootcamp.
Answer the student's question using ONLY the handbook excerpts provided.
Rules:
- Do not use outside knowledge and do not guess.
- If the excerpts do not contain the answer, set "found" to false and leave "answer" empty.
- Otherwise set "found" to true, answer clearly and concisely, and list in "pages" the page
  numbers of the excerpts you actually used."""


class LLMError(RuntimeError):
    """Raised when the language model cannot produce an answer."""


@dataclass
class LLMAnswer:
    found: bool
    answer: str = ""
    pages: list[int] = field(default_factory=list)


class LLM(Protocol):
    def generate(self, question: str, context: list[RetrievedChunk]) -> LLMAnswer: ...


class _AnswerSchema(BaseModel):
    found: bool
    answer: str
    pages: list[int]


def build_prompt(question: str, context: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(f"[Page {c.page}]\n{c.text}" for c in context)
    return f"Handbook excerpts:\n\n{excerpts}\n\nStudent question: {question}"


def parse_answer(raw: str) -> LLMAnswer:
    """Parse the model's JSON reply, tolerating markdown code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = _AnswerSchema.model_validate(json.loads(text))
    except (ValueError, TypeError) as exc:
        raise LLMError(f"Could not parse model response: {raw[:200]!r}") from exc
    return LLMAnswer(found=data.found, answer=data.answer.strip(), pages=data.pages)


RETRYABLE_STATUS = {429, 500, 503}


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        max_attempts: int = 4,
        backoff_seconds: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._client = None
        if api_key:
            from google import genai

            self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: list[RetrievedChunk]) -> LLMAnswer:
        from google.genai import types

        if self._client is None:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")

        from google.genai import errors

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=_AnswerSchema,
        )
        for attempt in range(self.max_attempts):
            try:
                response = self._client.models.generate_content(
                    model=self._model, contents=build_prompt(question, context), config=config
                )
                break
            except errors.APIError as exc:
                # Retry temporary failures (rate limit / overloaded) with exponential backoff.
                if exc.code in RETRYABLE_STATUS and attempt < self.max_attempts - 1:
                    time.sleep(self.backoff_seconds * 2**attempt)
                    continue
                raise LLMError(f"Gemini request failed: {exc}") from exc
            except Exception as exc:  # network errors etc.
                raise LLMError(f"Gemini request failed: {exc}") from exc
        if not response.text:
            raise LLMError("Gemini returned an empty response")
        return parse_answer(response.text)
