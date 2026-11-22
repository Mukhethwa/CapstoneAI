"""Capstone Part 4: run test questions through the RAG pipeline and record the results.

Questions cover the Student Handbook, the ZAIO website, both sources together, and questions
that neither source can answer (which must return the fallback message).

Usage:
    python -m scripts.evaluate                  # full RAG (needs GEMINI_API_KEY)
    python -m scripts.evaluate --retrieval-only # only show which sources were retrieved

Writes docs/TEST_RESULTS.md and docs/test_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import date

from app.config import NOT_FOUND_MESSAGE, PROJECT_ROOT, get_settings
from app.embeddings import SentenceTransformerEmbedder
from app.llm import LLMError
from app.rag import RAGPipeline, build_knowledge_base, load_or_crawl_website, unique_labels
from app.vector_store import VectorStore

HANDBOOK, WEBSITE, BOTH, NONE = "Handbook", "Website", "Both", "Unanswerable"
SITE = "https://www.zaio.io"


@dataclass(frozen=True)
class TestCase:
    question: str
    category: str
    expected: tuple[str, ...] = ()  # acceptable sources; empty means "must refuse"


TEST_CASES: list[TestCase] = [
    # Student Handbook
    TestCase("When are the tutor support hours?", HANDBOOK, ("Student Handbook - Page 20",)),
    TestCase("What days and times are the live classes?", HANDBOOK, ("Student Handbook - Page 11",)),
    TestCase("What does a yellow tile on my calendar mean?", HANDBOOK, ("Student Handbook - Page 9",)),
    TestCase("Which platform is used for communication after orientation day?", HANDBOOK,
             ("Student Handbook - Page 18",)),
    TestCase("How much is the final project worth in the final grade?", HANDBOOK,
             ("Student Handbook - Page 14",)),
    TestCase("How much RAM does my laptop need for the bootcamp?", HANDBOOK,
             ("Student Handbook - Page 5",)),
    TestCase("What is the highest level tag on the Discord server?", HANDBOOK,
             ("Student Handbook - Page 21",)),
    # ZAIO website
    TestCase("What courses does ZAIO offer?", WEBSITE,
             (f"{SITE}/", f"{SITE}/programmes", f"{SITE}/bootcamps", f"{SITE}/occupational-certificates",
              f"{SITE}/schools/technology", f"{SITE}/compare", f"{SITE}/2027")),
    TestCase("What is ZAIO's refund policy?", WEBSITE, (f"{SITE}/refund-policy", f"{SITE}/terms")),
    TestCase("When does the next Fullstack AI Engineer bootcamp start?", WEBSITE,
             (f"{SITE}/programmes/fullstack-ai-engineer-bootcamp", f"{SITE}/intakes")),
    TestCase("Where are ZAIO's campuses located?", WEBSITE,
             (f"{SITE}/about", f"{SITE}/", f"{SITE}/contact", f"{SITE}/locations/cape-town",
              f"{SITE}/locations/durban", f"{SITE}/2027")),
    TestCase("Is ZAIO accredited?", WEBSITE, (f"{SITE}/accreditation", f"{SITE}/2027/parents")),
    TestCase("What are the entry requirements for the occupational certificates?", WEBSITE,
             (f"{SITE}/entry-requirements", f"{SITE}/occupational-certificates")),
    TestCase("How do I apply to ZAIO?", WEBSITE, (f"{SITE}/how-to-apply",)),
    TestCase("What is The Weekly Commit?", WEBSITE, (f"{SITE}/newsletter", f"{SITE}/")),
    # Both sources
    TestCase("How much does the Fullstack AI Engineer bootcamp cost?", BOTH,
             ("Student Handbook - Page 16", f"{SITE}/programmes/fullstack-ai-engineer-bootcamp",
              f"{SITE}/tuition-and-financing")),
    TestCase("Which companies can I use to finance my studies?", BOTH,
             ("Student Handbook - Page 16", "Student Handbook - Page 17",
              f"{SITE}/tuition-and-financing", f"{SITE}/programmes/fullstack-ai-engineer-bootcamp")),
    # Neither source
    TestCase("What is the capital of France?", NONE),
    TestCase("Who won the 2022 FIFA World Cup?", NONE),
    TestCase("What is the Wi-Fi password at the Cape Town campus?", NONE),
    TestCase("What will the weather be in Cape Town tomorrow?", NONE),
]


def passed(case: TestCase, found: bool, sources: list[str]) -> bool:
    if not case.expected:
        return not found
    return found and any(s in case.expected for s in sources)


def _escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM call")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="seconds to wait between LLM calls (free-tier rate limits)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.chroma_dir, settings.collection_name)
    if store.count() == 0:
        pages = load_or_crawl_website(settings.website_snapshot, settings.website_url,
                                      settings.website_max_pages)
        build_knowledge_base(settings.handbook_path, pages, embedder, store,
                             settings.chunk_size, settings.chunk_overlap)

    llm = None
    if not args.retrieval_only:
        from app.llm import GeminiLLM

        llm = GeminiLLM(settings.gemini_api_key, settings.gemini_model, settings.gemini_fallback_models)
    pipeline = RAGPipeline(embedder, store, llm, settings.top_k, settings.min_similarity)

    rows = []
    for i, case in enumerate(TEST_CASES):
        context = pipeline.retrieve(case.question)
        retrieved = unique_labels(context)
        if args.retrieval_only:
            found = bool(context)
            sources = retrieved[:1]
            answer = (context[0].text[:200] + "...") if context else NOT_FOUND_MESSAGE
        else:
            if i:
                time.sleep(args.delay)
            try:
                result = pipeline.ask(case.question)
                found, sources, answer = result.found, result.sources, result.answer
            except LLMError as exc:
                found, sources, answer = False, [], f"(LLM error: {exc})"
        ok = passed(case, found, sources)
        rows.append({
            "question": case.question, "category": case.category,
            "expected": list(case.expected) or ["(fallback message)"],
            "retrieved": retrieved, "source": sources[0] if sources else None,
            "sources": sources, "answer": answer, "found": found, "passed": ok,
        })
        print(f"[{i + 1}/{len(TEST_CASES)}] {'PASS' if ok else 'FAIL'} {case.question} -> "
              f"{sources[0] if sources else 'no source'}")

    out_dir = PROJECT_ROOT / "docs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "test_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    total = len(rows)
    score = sum(r["passed"] for r in rows)
    by_category = {}
    for r in rows:
        done, count = by_category.get(r["category"], (0, 0))
        by_category[r["category"]] = (done + r["passed"], count + 1)
    mode = "retrieval only" if args.retrieval_only else f"full RAG with `{settings.gemini_model}`"
    lines = [
        "# Test Results",
        "",
        f"Generated on {date.today().isoformat()} by `python -m scripts.evaluate` ({mode}).",
        f"Embedding model `{settings.embedding_model}`, top_k={settings.top_k}, "
        f"min_similarity={settings.min_similarity}. Knowledge base: "
        + ", ".join(f"{n} {s} chunks" for s, n in store.count_by_source().items()) + ".",
        "",
        f"**{score}/{total} passed.** "
        + ", ".join(f"{c}: {p}/{n}" for c, (p, n) in by_category.items()) + ".",
        "",
        "A test passes when an answerable question is answered with one of the expected sources,",
        "or when an unanswerable question returns the fallback message:",
        f"*\"{NOT_FOUND_MESSAGE}\"*",
        "",
        "| # | Category | Question | Retrieved source | Generated answer | Result |",
        "|---|----------|----------|------------------|------------------|--------|",
    ]
    for n, r in enumerate(rows, start=1):
        source = "<br>".join(r["sources"]) if r["sources"] else "None (refused)"
        lines.append(
            f"| {n} | {r['category']} | {_escape(r['question'])} | {source} | "
            f"{_escape(r['answer'])} | {'Pass' if r['passed'] else 'Fail'} |"
        )
    lines += ["", "## Chunks retrieved before generation", "",
              "Sources of the chunks passed to the model as context, most similar first.", ""]
    for n, r in enumerate(rows, start=1):
        lines.append(f"{n}. **{r['question']}**: " + (", ".join(r["retrieved"]) or "nothing above threshold"))
    (out_dir / "TEST_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{score}/{total} passed. Wrote {out_dir / 'TEST_RESULTS.md'}")


if __name__ == "__main__":
    main()
