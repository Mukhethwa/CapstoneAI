# Loom Demonstration Outline (5–10 minutes)

Before recording: activate the virtual environment, make sure `.env` has `GEMINI_API_KEY`, and run `python -m scripts.ingest` once so the vector database is built.

## 1. Introduction (≈ 30 s)

- What the project is: a RAG assistant that answers from the ZAIO Student Handbook and the ZAIO website.
- It builds on the previous handbook-only practical. Show the git history (`git log --oneline`), where each commit adds one part.

## 2. Knowledge base (≈ 2 min), Part 1

- Open `app/web_crawler.py`: robots.txt, sitemap seeding, same-host links, cleaning (nav/header/footer removed, cards kept on one line, boilerplate lines removed).
- Run `python -m scripts.crawl --max-pages 10` and show the log of crawled pages.
- Open `data/website_pages.json` and show that one page's text is clean.
- Open `app/document_processor.py` (`chunk_pages`, `chunk_web_pages`) and `app/vector_store.py` (`_metadata`): one ChromaDB collection with `source`, `page`, `url` and `title` metadata.
- Show `GET /health`: `{"handbook": 40, "website": 523}`.

## 3. Retrieval and generation (≈ 2 min), Part 2

- Open `app/rag.py` (`retrieve`): each source is searched separately, best matches from both are kept, at most 2 chunks per page, and a similarity floor applies.
- Open `app/llm.py` (`SYSTEM_PROMPT`): answer only from the excerpts, refuse otherwise, cite excerpt numbers.

## 4. API (≈ 1.5 min), Part 3

- Start the API: `uvicorn app.main:app --port 8000`, then open http://localhost:8000/docs.
- `POST /ask` with:
  - "What courses does ZAIO offer?" → website URL source
  - "When are the tutor support hours?" → `Student Handbook - Page 20`
  - "How much does the Fullstack AI Engineer bootcamp cost?" → both sources, and the answer explains how the handbook and website prices differ
  - "Who won the 2022 FIFA World Cup?" → fallback message
  - `{}` → 422 JSON error

## 5. Testing (≈ 1 min), Part 4

- Run `pytest` (59 passed).
- Open `docs/TEST_RESULTS.md`: 21/21, covering handbook, website, both, and unanswerable questions, with the question, retrieved source and answer for each.

## 6. n8n (≈ 1.5 min), Part 5

- Import `n8n/zaio_assistant_workflow.json` and walk through the nodes.
- Publish it, open the chat, and ask two questions (one website, one handbook), then one off-topic question.
- Optionally call the webhook with curl and show the JSON response.
- Mention the optional Discord forwarding node.

## 7. Wrap up (≈ 30 s)

- Summary, and possible next steps (scheduled re-crawls, a source filter in the request, conversation memory).
