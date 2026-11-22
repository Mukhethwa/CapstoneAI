"""Answer generation with Google Gemini, grounded only in the retrieved knowledge-base context."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel

from app.vector_store import RetrievedChunk

SYSTEM_PROMPT = """You are a helpful assistant for ZAIO, a South African tech and business institute.
You answer questions using ONLY the numbered excerpts provided. Excerpts come from two sources:
the ZAIO Student Handbook (for the Jan 2026 Full-Stack AI Engineer Bootcamp) and the ZAIO website.
Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge and do not guess.
- If the excerpts do not contain the answer, set "found" to false and leave "answer" empty.
- Otherwise set "found" to true and answer clearly and concisely.
- If the handbook and the website disagree (for example on fees), say which source says what.
- In "sources" list the numbers of the excerpts you actually used, most important first."""


class LLMError(RuntimeError):
    """Raised when the language model cannot produce an answer."""


@dataclass
class LLMAnswer:
    found: bool
    answer: str = ""
    sources: list[int] = field(default_factory=list)  # 1-based excerpt numbers


class LLM(Protocol):
    def generate(self, question: str, context: list[RetrievedChunk]) -> LLMAnswer: ...


class _AnswerSchema(BaseModel):
    found: bool
    answer: str
    sources: list[int]


def build_prompt(question: str, context: list[RetrievedChunk]) -> str:
    excerpts = "\n\n".join(
        f"[{n}] Source: {c.label}\n{c.text}" for n, c in enumerate(context, start=1)
    )
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def parse_answer(raw: str) -> LLMAnswer:
    """Parse the model's JSON reply, tolerating markdown code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = _AnswerSchema.model_validate(json.loads(text))
    except (ValueError, TypeError) as exc:
        raise LLMError(f"Could not parse model response: {raw[:200]!r}") from exc
    return LLMAnswer(found=data.found, answer=data.answer.strip(), sources=data.sources)


RETRYABLE_STATUS = {429, 500, 503}


class GeminiLLM:
    """Gemini client with retries and fallback models for temporary overloads."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        fallback_models: list[str] | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 2.0,
    ):
        self.models = [model] + [m for m in (fallback_models or []) if m and m != model]
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._client = None
        if api_key:
            from google import genai

            self._client = genai.Client(api_key=api_key)

    def generate(self, question: str, context: list[RetrievedChunk]) -> LLMAnswer:
        if self._client is None:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=_AnswerSchema,
        )
        prompt = build_prompt(question, context)
        last_error: Exception | None = None
        for model in self.models:
            for attempt in range(self.max_attempts):
                try:
                    response = self._client.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                except errors.APIError as exc:
                    last_error = exc
                    if exc.code not in RETRYABLE_STATUS:
                        raise LLMError(f"Gemini request failed: {exc}") from exc
                    # Temporary failure (rate limit / overloaded): back off, then retry.
                    if attempt < self.max_attempts - 1:
                        time.sleep(self.backoff_seconds * 2**attempt)
                    continue
                except Exception as exc:  # network errors etc.
                    raise LLMError(f"Gemini request failed: {exc}") from exc
                if not response.text:
                    raise LLMError("Gemini returned an empty response")
                return parse_answer(response.text)
            # This model stayed unavailable; move on to the next fallback model.
        raise LLMError(f"Gemini request failed: {last_error}")
