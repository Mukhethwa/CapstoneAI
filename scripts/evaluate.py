"""Part 4: run a set of test questions through the RAG pipeline and record the results.

Usage:
    python -m scripts.evaluate                  # full RAG (needs GEMINI_API_KEY)
    python -m scripts.evaluate --retrieval-only # only show which pages were retrieved

Writes docs/TEST_RESULTS.md (and docs/test_results.json).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from app.config import PROJECT_ROOT, get_settings
from app.embeddings import SentenceTransformerEmbedder
from app.llm import LLMError
from app.rag import RAGPipeline, format_source, ingest_handbook
from app.vector_store import VectorStore

# (question, topic, expected page(s) or None when the handbook does not contain the answer)
TEST_QUESTIONS: list[tuple[str, str, list[int] | None]] = [
    ("What are the total fees for the bootcamp?", "Fees", [16]),
    ("When is orientation day?", "Key dates", [7]),
    ("What days and times are the live classes?", "Live classes", [11]),
    ("How much RAM does my laptop need?", "Hardware", [5]),
    ("How much is the final project worth in the final grade?", "Grading", [14]),
    ("When are the tutor support hours?", "Tutor support", [20]),
    ("Which platform is used for communication after orientation day?", "Communication", [18]),
    ("What financing options are available to pay the fees?", "Fees", [16, 17]),
    ("What does a yellow tile on my calendar mean?", "Calendar", [9]),
    ("How much do paid internships pay per month?", "Placement", [24]),
    ("Who is the lead instructor?", "Team", [15]),
    ("What is the highest level tag on Discord?", "Discord levels", [21]),
    ("Which technologies are covered in weeks 16 to 20?", "Sprints", [6]),
    ("What is the attendance requirement?", "Out of scope", None),
    ("What is the bootcamp's refund policy?", "Out of scope", None),
    ("What is the capital of France?", "Out of scope", None),
]


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM call")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="seconds to wait between LLM calls (free-tier rate limits)")
    args = parser.parse_args()

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    if store.count() == 0:
        ingest_handbook(settings.handbook_path, embedder, store,
                        settings.chunk_size, settings.chunk_overlap)

    llm = None
    if not args.retrieval_only:
        from app.llm import GeminiLLM

        llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model)
    pipeline = RAGPipeline(embedder, store, llm, settings.top_k, settings.min_similarity)

    rows = []
    for i, (question, topic, expected) in enumerate(TEST_QUESTIONS):
        context = pipeline.retrieve(question)
        retrieved = format_source([c.page for c in context]) or "none above threshold"
        if args.retrieval_only:
            answer = context[0].text[:200] + "..." if context else "(nothing retrieved)"
            source = format_source([context[0].page]) if context else None
        else:
            if i:
                time.sleep(args.delay)
            try:
                result = pipeline.ask(question)
                answer, source = result.answer, result.source
            except LLMError as exc:
                answer, source = f"(LLM error: {exc})", None
        expected_label = format_source(expected) if expected else "Not in handbook"
        rows.append({
            "question": question, "topic": topic, "expected": expected_label,
            "retrieved": retrieved, "source": source or "N/A", "answer": answer,
        })
        print(f"[{i + 1}/{len(TEST_QUESTIONS)}] {question} -> {source or 'N/A'}")

    out_dir = PROJECT_ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "test_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    mode = "retrieval only" if args.retrieval_only else f"full RAG with {settings.gemini_model}"
    lines = [
        "# Test Results",
        "",
        f"Generated on {date.today().isoformat()} by `python -m scripts.evaluate` ({mode}).",
        f"Embedding model: `{settings.embedding_model}`, top_k={settings.top_k}, "
        f"min_similarity={settings.min_similarity}.",
        "",
        "| # | Topic | Question | Expected | Source returned | Answer |",
        "|---|-------|----------|----------|-----------------|--------|",
    ]
    for n, r in enumerate(rows, start=1):
        lines.append(
            f"| {n} | {r['topic']} | {_escape(r['question'])} | {r['expected']} | "
            f"{r['source']} | {_escape(r['answer'])} |"
        )
    lines += ["", "Pages retrieved for each question (before answer generation):", ""]
    lines += [f"{n}. {r['question']}: {r['retrieved']}" for n, r in enumerate(rows, start=1)]
    Path(out_dir / "TEST_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'TEST_RESULTS.md'}")


if __name__ == "__main__":
    main()
