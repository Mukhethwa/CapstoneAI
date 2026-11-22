# ZAIO Knowledge Assistant (AI Capstone)

A Retrieval-Augmented Generation (RAG) assistant that answers questions using two knowledge sources:

- 📄 **ZAIO Student Handbook** (Jan 2026 Full-Stack AI Engineer Bootcamp, PDF)
- 🌐 **ZAIO website**: https://www.zaio.io (crawled and cleaned)

Both sources are chunked, embedded and stored in **one** vector database. Every answer is generated only from retrieved context and comes with its source: a ZAIO URL or a handbook page. If neither source has the answer, the assistant replies:

> I could not find that information in the available knowledge base.

The API is exposed as `POST /ask` and is connected to an **n8n** workflow (chat and webhook).

This project builds on the previous practical (the handbook-only RAG assistant). The git history shows each step.

## Architecture

```
                        ┌────────────── Ingestion ───────────────┐
 handbook.pdf ─► pypdf ─► clean ─► chunk per page ──────────────┐
                                                                ├─► embed (MiniLM) ─► ChromaDB
 zaio.io ─► crawler ─► strip nav/header/footer ─► chunk per URL ┘     (one collection,
           (robots.txt, sitemap,  + boilerplate removal                metadata: source,
            internal links)                                            page | url, title)

                        ┌────────────── Question ────────────────┐
 POST /ask ─► embed question ─► search handbook + website ─► best chunks (per-source & per-page caps)
                                     │
                    nothing above similarity floor? ─► fallback message (LLM not called)
                                     │
                     Gemini (numbered excerpts only, JSON output) ─► answer + cited excerpts
                                     │
                     {"answer", "source", "sources", "found"}
```

### Part 1: Extending the knowledge base

| Step | Handbook | Website |
|------|----------|---------|
| Load | `pypdf` reads every page | `httpx` crawler seeded from `sitemap.xml` and the homepage, following internal links breadth first |
| Rules | | respects `robots.txt`, same host only, skips files, blog posts (opt-in) and print-edition duplicates, polite delay |
| Clean | page-number lines removed, spaced headings (`E X C I T I N G`) joined | removes `<nav>`, site `<header>`, `<footer>`, scripts, styles, SVG, forms, `aria-hidden` mock-ups, cookie/announcement bars; keeps `<main>`; puts each card (course tile, list item, table row) on one line; removes lines repeated on more than half of all pages |
| Chunk | ~800 characters, 150 overlap, per page | same sizes, per URL; each chunk starts with the page title |
| Metadata | `source=handbook`, `page` | `source=website`, `url`, `title` |
| Store | same ChromaDB collection (`knowledge_base`, cosine distance) | same |

zaio.io is a server-rendered Next.js site, so the full content is in the HTML the server returns. A plain HTTP crawler with BeautifulSoup is enough, and a headless browser such as Puppeteer isn't needed.

The current knowledge base holds **40 handbook chunks** (26 pages) and **523 website chunks** (59 pages).

### Part 2: Retrieval and generation

1. The question is embedded with the same model (`sentence-transformers/all-MiniLM-L6-v2`).
2. **Each source is searched separately.** The website has more than ten times as many chunks as the handbook, so a single global search would often push relevant handbook passages out. The best 2 matches from each source are always kept (if relevant), and the remaining slots (6 in total) go to the best matches overall.
3. At most 2 chunks come from any one page or URL, so one page with a matching title cannot fill the context.
4. Chunks below a cosine similarity of 0.3 are discarded. If nothing is left, the API returns the fallback message without calling the LLM.
5. Gemini receives the numbered excerpts, each labelled with its source, and a strict system prompt: answer only from the excerpts, refuse if they don't contain the answer, and point out where the handbook and website disagree. It returns JSON (`found`, `answer`, `sources`: the numbers of the excerpts it used).
6. Cited excerpt numbers are mapped back to the retrieved chunks; numbers that don't exist are ignored. `source` is the most important cited source, and `sources` lists all of them.
7. Gemini errors such as rate limits or overloads are retried with backoff. If the main model stays unavailable, fallback models are tried.

## Project structure

```
app/
  config.py              settings from environment / .env
  web_crawler.py         ZAIO website crawler and HTML cleaning
  document_processor.py  PDF loading, text cleaning, chunking (handbook + website)
  embeddings.py          Sentence Transformers embedder
  vector_store.py        ChromaDB wrapper (one collection, source metadata)
  llm.py                 Gemini client, prompt, JSON parsing, retries/fallbacks
  rag.py                 knowledge-base build + retrieval + answer pipeline
  main.py                FastAPI app (POST /ask, GET /health)
scripts/
  crawl.py               crawl zaio.io into data/website_pages.json
  ingest.py              build the vector database from both sources
  evaluate.py            run the test questions, write docs/TEST_RESULTS.md
n8n/
  zaio_assistant_workflow.json   importable n8n workflow
tests/                   unit tests (offline: fake embedder, fake LLM, mocked HTTP)
data/handbook.pdf        the Student Handbook
docs/TEST_RESULTS.md     evaluation results (question, retrieved source, answer)
docs/LOOM_SCRIPT.md      outline for the demonstration video
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY`. You can get a free key at https://aistudio.google.com/app/apikey.

## Run

```bash
# 1. Crawl the website and build the vector database (about 5 minutes the first time)
python -m scripts.ingest            # add --recrawl to fetch the website again

# 2. Start the API
uvicorn app.main:app --port 8000
```

If the vector database is empty, the API builds it on startup. Interactive docs are at http://localhost:8000/docs.

## API

### `POST /ask`

```json
{ "question": "What courses does ZAIO offer?" }
```

Website answer:

```json
{
  "answer": "ZAIO offers Occupational Certificates (such as Software Developer, AI & ML Engineer, ...) and job-skills bootcamps ...",
  "source": "https://www.zaio.io/",
  "sources": ["https://www.zaio.io/", "https://www.zaio.io/2027"],
  "found": true
}
```

Handbook answer:

```json
{
  "answer": "Tutors host support hours every Tuesday from 2 pm to 4 pm and 6 pm to 8 pm, and on Thursdays from 10 am to 12 pm and 6 pm to 8 pm.",
  "source": "Student Handbook - Page 20",
  "sources": ["Student Handbook - Page 20"],
  "found": true
}
```

Not in either source:

```json
{
  "answer": "I could not find that information in the available knowledge base.",
  "source": null,
  "sources": [],
  "found": false
}
```

| Field | Meaning |
|-------|---------|
| `answer` | generated answer, or the fallback message |
| `source` | the most appropriate source: a ZAIO URL or `Student Handbook - Page N` |
| `sources` | every source the answer used, most important first |
| `found` | `false` when the assistant refused; useful for branching in n8n |

### Errors (always JSON)

| Status | When | Body |
|--------|------|------|
| 422 | missing, empty or non-string `question`, more than 1000 characters, or invalid JSON | `{"error": "Invalid request", "details": ["question: Field required"]}` |
| 404 / 405 | unknown route or wrong method | `{"error": "Not Found", "details": null}` |
| 502 | Gemini unavailable (bad key, quota, every model overloaded) | `{"error": "The language model is unavailable", "details": ["..."]}` |
| 500 | unexpected error | `{"error": "Internal server error", "details": null}` |

### `GET /health`

```json
{ "status": "ok", "chunks": 563, "by_source": { "handbook": 40, "website": 523 } }
```

## n8n integration (Part 5)

[`n8n/zaio_assistant_workflow.json`](n8n/zaio_assistant_workflow.json) contains two flows:

```
Chat message received ─► Ask RAG API (chat) ─► Reply in chat ─► Forward to Discord (optional, disabled)
Webhook question received ─► Ask RAG API (webhook) ─► Respond to webhook
```

- **Chat flow**: n8n's built-in chat page. The user types a question, the workflow calls `POST /ask`, and the reply is shown in the chat as the answer followed by "Source: …". The optional Discord node also posts every Q&A to a Discord channel once you enable it and paste a channel webhook URL.
- **Webhook flow**: `POST /webhook/zaio-ask` with `{"question": "..."}` returns `{question, answer, source, sources, found, error}`. Any service (Slack, Telegram, a website) can call it.
- If the API is down, the HTTP nodes continue with an error item, so the user gets a "Sorry, the assistant is unavailable" message instead of a silent failure.

To use it:

1. Start the API (`uvicorn app.main:app --port 8000`).
2. In n8n, go to **Workflows → Import from File** and choose `n8n/zaio_assistant_workflow.json`.
3. If n8n runs in Docker, change the URL in both **Ask RAG API** nodes to `http://host.docker.internal:8000/ask`. The default is `http://127.0.0.1:8000/ask`. Use `127.0.0.1` rather than `localhost`, because n8n may resolve `localhost` to IPv6 while uvicorn listens on IPv4.
4. Publish (activate) the workflow, then open the chat from the **Chat message received** node, or test the webhook:

```bash
curl -X POST http://localhost:5678/webhook/zaio-ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is ZAIO'\''s refund policy?"}'
```

The workflow was tested end to end on n8n 2.31.7 through both the chat endpoint and the webhook.

## Testing (Part 4)

### Unit tests

```bash
pytest
```

The 59 tests run offline. They cover crawling (robots.txt, link following, page limits, blog/print rules, using a mocked HTTP transport), HTML cleaning, card flattening, boilerplate removal, chunking and metadata for both sources, the vector store (search, source filter, per-source counts), retrieval (per-source and per-page caps, similarity floor), source citation, the fallback behaviour, Gemini retries and fallback models (with a fake client), and every API success and error response.

### Evaluation

```bash
python -m scripts.evaluate                  # full RAG, needs GEMINI_API_KEY
python -m scripts.evaluate --retrieval-only # retrieval only, no LLM calls
```

21 questions: 7 from the handbook, 8 from the website, 2 that need both, and 4 that neither source can answer. For each question it records the question, the retrieved source and the generated answer, and checks the result. **The current result is 21/21**; see [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).

## Configuration

All settings can be overridden in `.env` (see `.env.example`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | (none) | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | main model |
| `GEMINI_FALLBACK_MODELS` | `gemini-3.1-flash-lite,gemini-3.5-flash` | tried in order when the main model is overloaded |
| `WEBSITE_URL` | `https://www.zaio.io/` | site to crawl |
| `WEBSITE_MAX_PAGES` | `80` | crawl limit |
| `TOP_K` | `6` | chunks passed to the model |
| `MIN_SIMILARITY` | `0.3` | cosine similarity floor |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | chunking (characters) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model |

## Demonstration video

The Loom walkthrough outline is in [docs/LOOM_SCRIPT.md](docs/LOOM_SCRIPT.md).

Loom video: *add your link here*
