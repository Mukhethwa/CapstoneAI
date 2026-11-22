# Handbook RAG Assistant

An AI assistant that answers student questions about the **Jan 2026 Full-Stack AI Engineer Bootcamp** using only the bootcamp handbook. It is a Retrieval-Augmented Generation (RAG) system exposed as a JSON REST API, ready to be called from n8n.

- 📖 Searches the handbook for relevant information (Sentence Transformers + ChromaDB)
- 🤖 Generates answers with an LLM (Google Gemini) using only the retrieved context
- 🚫 Responds with *"I don't know. That information is not available in the handbook."* when the answer is not in the handbook

## How it works

```
                 ┌──────────── Ingestion (Part 1) ─────────────┐
 handbook.pdf ─► extract text ─► clean ─► chunk (per page) ─► embed ─► ChromaDB
                                                                         │
                 ┌──────────── Question (Part 2) ──────────────┐         │
 POST /ask ─► embed question ─► similarity search (top-k) ◄──────────────┘
                                   │
                    no chunk above threshold? ──► "I don't know..."
                                   │
                         Gemini (context only) ─► {"answer", "source"}
```

1. **Load and extract**: `pypdf` reads every page. The text is cleaned (page-number lines are removed, letter-spaced headings such as `E X C I T I N G` are joined, and whitespace is normalised).
2. **Chunk**: each page is split separately into chunks of up to 800 characters with a 150-character overlap, so every chunk maps to exactly one page number.
3. **Embed and store**: chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` (normalised vectors) and stored in a persistent ChromaDB collection that uses cosine distance.
4. **Retrieve**: the question is embedded with the same model, and the top 4 chunks are retrieved. Chunks below a minimum cosine similarity (0.25) are discarded.
5. **Generate**: Gemini receives only the retrieved excerpts and a strict system prompt. It returns structured JSON (`found`, `answer`, `pages`). If nothing relevant was retrieved, or the model reports the context is insufficient, the API returns the "I don't know" message. Page numbers the model cites are checked against the pages actually retrieved.

## Project structure

```
app/
  config.py              settings from environment / .env
  document_processor.py  PDF loading, text cleaning, chunking
  embeddings.py          Sentence Transformers embedder
  vector_store.py        ChromaDB wrapper
  llm.py                 Gemini client, prompt, JSON parsing
  rag.py                 ingestion + retrieval + answer pipeline
  main.py                FastAPI app (POST /ask, GET /health)
scripts/
  ingest.py              (re)build the vector database
  evaluate.py            run the test questions and write docs/TEST_RESULTS.md
tests/                   unit tests (offline, with fake embedder and LLM)
data/handbook.pdf        the bootcamp handbook
docs/TEST_RESULTS.md     evaluation results (question, source, answer)
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

The first run downloads the embedding model (about 90 MB).

## Run

```bash
# Build the vector database (optional: the API also does this on first start)
python -m scripts.ingest

# Start the API
uvicorn app.main:app --reload --port 8000
```

Interactive docs are at http://localhost:8000/docs.

## API

### `POST /ask`

Request:

```json
{ "question": "What are the total fees for the bootcamp?" }
```

Response `200`:

```json
{
  "answer": "The total fees for the bootcamp are R 38,950, payable in full before orientation day.",
  "source": "Page 16",
  "found": true
}
```

When the handbook does not contain the answer:

```json
{
  "answer": "I don't know. That information is not available in the handbook.",
  "source": null,
  "found": false
}
```

`source` holds the PDF page number(s) the answer came from, for example `"Page 16"` or `"Pages 16, 17"`. `found` makes it easy to branch on in n8n.

### Errors

Every error is returned as JSON:

| Status | When | Body |
|--------|------|------|
| 422 | Missing/empty/non-string `question`, question over 1000 chars, body is not valid JSON | `{"error": "Invalid request", "details": ["question: Field required"]}` |
| 405 / 404 | Wrong method or route | `{"error": "Method Not Allowed", "details": null}` |
| 502 | Gemini unavailable (bad key, quota, network) | `{"error": "The language model is unavailable", "details": ["..."]}` |
| 500 | Unexpected error | `{"error": "Internal server error", "details": null}` |

### `GET /health`

```json
{ "status": "ok", "chunks": 30 }
```

### Examples

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When are the tutor support hours?"}'
```

PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType "application/json" -Body '{"question": "When are the tutor support hours?"}'
```

## Using it from n8n

Add an **HTTP Request** node:

- Method: `POST`
- URL: `http://localhost:8000/ask` (use `http://host.docker.internal:8000/ask` if n8n runs in Docker)
- Send Body: JSON, `{ "question": "{{ $json.question }}" }`

The node outputs `answer`, `source` and `found`, which you can pass to a chat reply or an IF node.

## Testing

Unit tests run offline. They use a fake embedder and a fake LLM, so no API key or model download is needed:

```bash
pytest
```

They cover text cleaning and chunking, real handbook extraction, vector search, the not-found behaviour, citation validation, LLM response parsing, and the API's success and error responses.

### Evaluation (Part 4)

```bash
python -m scripts.evaluate                  # full RAG, needs GEMINI_API_KEY
python -m scripts.evaluate --retrieval-only # retrieval only, no LLM calls
```

This runs 16 questions covering fees, dates, live classes, hardware, grading, tutor support, communication, the calendar, placement, the team, Discord levels and the sprint curriculum, plus 3 questions that are not in the handbook. It writes the question, expected page, returned source and answer to [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | (none) | Google AI Studio key |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model |
| `HANDBOOK_PATH` | `data/handbook.pdf` | PDF to index |
| `CHROMA_DIR` | `chroma_db` | where the vector DB is stored |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | chunking (characters) |
| `TOP_K` | `4` | chunks retrieved per question |
| `MIN_SIMILARITY` | `0.25` | cosine similarity floor for "relevant" |

After changing the handbook, chunking or the embedding model, run `python -m scripts.ingest` to rebuild the index.
