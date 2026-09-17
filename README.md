# RAGBench

A Retrieval-Augmented Generation chatbot built with FastAPI, LangChain, and OpenAI. Supports three ingestion/retrieval strategies for A/B/C comparison: standard chunking (v1), layout-aware hybrid retrieval (v2), and format-aware structured ingestion for tabular/multi-format data (v3) — plus an [evaluation harness](#evaluation) that scores all three against each other with real numbers, not just a feature table.

> **Automation:** Creating a Jira ticket automatically opens a PR implemented by
> Claude Code. See [`docs/jira-automation-setup.md`](docs/jira-automation-setup.md).

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   FastAPI    │────▶│  RAG Service  │────▶│   OpenAI     │
│   Routers   │     │  (LangChain)  │     │  gpt-4o-mini │
└─────────────┘     └──────┬───────┘     └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  FAISS   │ │   BM25   │ │ Ensemble │
        │(vectors) │ │(keywords)│ │ Retriever│
        └──────────┘ └──────────┘ └──────────┘
```

## Retrieval Strategies

| | Strategy v1 | Strategy v2 | Strategy v3 |
|---|---|---|---|
| **Formats** | `.pdf` only | `.pdf` only | `.pdf`, `.md`, `.html`, `.csv`/`.xlsx`, `.json` |
| **PDF Parsing** | LlamaIndex `SimpleDirectoryReader` | `pymupdf4llm` (layout-aware markdown) | `pymupdf4llm` for prose **+** PyMuPDF `find_tables()` for real embedded tables |
| **Tabular data** | flattened into prose text (if picked up at all) | flattened into prose text (if picked up at all) | one node per row/record, column alignment preserved |
| **Chunking** | `SentenceSplitter` (1000 tokens, 200 overlap) | `MarkdownNodeParser` (heading boundaries) | `MarkdownNodeParser` for prose, 1 row/record = 1 chunk for tables/CSV/JSON |
| **Retrieval** | FAISS similarity search | Hybrid: FAISS (0.6) + BM25 (0.4) via `EnsembleRetriever` | Same hybrid as v2, but **no `score_threshold`** (see [Known Issues](#known-issues)) |
| **Metadata** | Basic file info | Content type tagging (prose/code/table) | Content type tagging incl. `table_row`/`json_record`; logs a warning on embedding drops instead of silently dropping them |
| **FAISS Index** | `faiss_index_v1/` | `faiss_index_v2/` | `faiss_index_v3/` |

Pass `"strategy": "v1"`, `"v2"`, or `"v3"` in request bodies to select.

v3 exists specifically to fix the tabular/format gap v1 and v2 share: a table embedded in a PDF, or a CSV/JSON/HTML file, is either dropped entirely (v1/v2 only ever look for `.pdf` files) or flattened into a plain-text blob that loses row/column alignment. See [Evaluation](#evaluation) for the measured before/after.

## Project Structure

```
├── main.py                          # FastAPI app + lifespan startup
├── config.py                        # Settings (pydantic-settings, .env)
│
├── models/
│   ├── chat.py                      # ChatRequest, ChatResponse
│   └── document.py                  # IngestRequest, IngestResponse
│
├── routers/
│   ├── chat.py                      # POST /chat/message
│   └── documents.py                 # POST /documents/ingest
│
├── services/
│   ├── rag_service.py               # Retrieval + generation pipeline
│   ├── ingestion_service.py         # Dual-strategy document ingestion
│   ├── web_search.py                # OpenAI web search tool
│   ├── llm/                         # LLM providers (OpenAI)
│   ├── embeddings/                  # Embedding providers (OpenAI, Qwen)
│   └── vector_store/                # Vector store (FAISS + factory)
│
├── data/
│   ├── python/                      # Python documentation PDFs
│   ├── kubernetes/                  # Kubernetes documentation PDFs
│   └── finance/                     # Synthetic multi-format domain: PDF (with an embedded table), CSV, JSON, HTML
│
├── eval/                            # Strategy evaluation harness (see "Evaluation" below)
│   ├── testsets/                    # Per-domain Q&A test cases
│   ├── metrics.py                   # Retrieval-quality proxies (keyword recall, source-hit rate)
│   ├── judge.py                     # LLM-as-judge (faithfulness/relevance/completeness)
│   ├── known_issues.py              # Runnable proofs of the v1/v2 problems below
│   ├── report.py                    # Markdown report generator
│   └── run_eval.py                  # Entry point: `python -m eval.run_eval`
│
├── faiss_index_v1/                  # Persisted FAISS index (strategy v1)
├── faiss_index_v2/                  # Persisted FAISS index (strategy v2)
├── faiss_index_v3/                  # Persisted FAISS index (strategy v3)
├── ingestion_cache/                 # LlamaIndex pipeline cache
```

## Setup

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- OpenAI API key

### Install

```bash
git clone <repo-url>
cd RAGBench
poetry install
```

### Configure

Create a `.env` file:

```env
APP_ENV=dev
APP_HOST=0.0.0.0
APP_PORT=8000
OPENAI_API_KEY=sk-your-key-here
```

### Run

```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or via VS Code: use the included launch configuration (F5).

## API Endpoints

### Chat

```
POST /chat/message
```

```json
{
  "message": "What are Python decorators?",
  "strategy": "v1"
}
```

Response:

```json
{
  "reply": "Decorators are functions that wrap another function..."
}
```

The response includes a **Context Quality** self-assessment:
- **Summary** — what the retrieved context covers
- **Relevance** — High / Medium / Low with reasoning
- **What was missing** — if relevance is Medium or Low

### Document Ingestion

```
POST /documents/ingest
```

```json
{
  "directory": "data/finance",
  "strategy": "v3"
}
```

Response:

```json
{
  "chunks_ingested": 142
}
```

### Interactive Docs

FastAPI auto-generates interactive API docs:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| LLM | OpenAI gpt-4o-mini (via LangChain) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | FAISS (faiss-cpu) |
| Keyword Search | BM25 (rank-bm25) |
| Hybrid Retrieval | LangChain EnsembleRetriever |
| PDF Parsing (v1) | LlamaIndex SimpleDirectoryReader + pypdf |
| PDF Parsing (v2) | pymupdf4llm (layout-aware markdown) |
| PDF Table Extraction (v3) | PyMuPDF (fitz) `Page.find_tables()` |
| Tabular/JSON Ingestion (v3) | pandas (`.csv`/`.xlsx`), stdlib `json`, BeautifulSoup (`.html`) |
| Chunking (v1) | LlamaIndex SentenceSplitter |
| Chunking (v2/v3 prose) | LlamaIndex MarkdownNodeParser |
| Web Search | OpenAI Responses API (web_search tool) |
| Config | pydantic-settings (.env) |
| Evaluation | Custom harness (`eval/`): retrieval-proxy metrics + LLM-as-judge |

## Evaluation

`eval/` scores v1, v2, and v3 against each other on the same Q&A test sets (`eval/testsets/*.json`), across three domains: `python`, `kubernetes`, and `finance` (the multi-format domain that specifically exercises tabular/CSV/JSON/HTML data). For each `(domain, strategy, question)` it records:

- **Keyword recall** and **source-hit rate** — did the retrieved context actually contain what was needed, and come from the right file?
- **Faithfulness / relevance / completeness** (1-5) — an LLM-as-judge score (`eval/judge.py`), grading the final answer strictly against the retrieved context.
- **Retrieval/generation latency**.

Run it:

```bash
poetry run python -m eval.run_eval                       # all domains, all strategies
poetry run python -m eval.run_eval --strategies v1,v3     # just a subset
```

It ingests any domain/strategy combination that isn't indexed yet, then writes `eval/results/report.md` (human-readable) and `eval/results/raw_<timestamp>.json` (every question's full trace). The report includes a data-driven **Strategy Progression** section — the measured v1→v2 and v2→v3 deltas, not a narrative claim.

A real run of this repo's data (2026-09-15) found: `table`/`json`-category keyword recall went from **0.44 (v2) → 1.0 (v3)**, and source-hit rate across all categories went from **81% (v1/v2) → 100% (v3)** — because v1/v2 only ever ingest `.pdf` files, so the CSV/JSON/HTML content in `data/finance/` is invisible to them. Concretely: asked for the Q3 EMEA revenue figure (in `quarterly_revenue.csv`) or a SKU's price (in `product_catalog.json`), v1 and v2 both correctly say "I don't have enough information" — v3 answers correctly with the exact figure. See `eval/results/report.md` for the full breakdown, including where v2's hybrid retrieval actually scored *lower* than v1 on relevance/completeness (BM25 pulling in a less-relevant chunk alongside the correct one).

## Known Issues

Proven, not just claimed — see `eval/known_issues.py` for runnable checks and `eval/results/report.md` for the latest run's evidence:

- **No dedup on ingestion** — re-running `/documents/ingest` on an unchanged directory duplicates every chunk (confirmed: 1 chunk → 2 after re-ingesting the same tiny directory). No content hashing, no upsert, no delete/replace path.
- **`score_threshold` is miscalibrated, not just a no-op** — FAISS's default relevance-score function produces scores like 0.21-0.28 for the single most relevant chunk, well outside the assumed `[0,1]` range. `score_threshold=0.0` (v1/v2's setting) lets everything through; any real threshold (e.g. 0.5) can filter out the correct answer entirely. v3 avoids the parameter altogether (plain top-k).
- **BM25 is rebuilt from scratch on every v2/v3 chat request** (`BM25Retriever.from_documents(...)` inside `rag_service.ask()`), and the cost scales with corpus size (measured: ~0.9ms at 50 chunks, ~430ms at 20,000).
- **No OCR** — a scanned/image-only PDF page yields 0 characters of extracted text in both v1 and v2; content on such a page is silently unretrievable.
- **`_tag_content_type` heuristic misclassifies** — e.g. an indented block-quote gets tagged `code`, prose using pipe characters gets tagged `table`, and unindented real code gets tagged `prose`.
- Other lower-severity gaps found by code inspection: no `top_k` request parameter, no re-ranking step, minimal chunk metadata (no page numbers/timestamps), no per-content-type chunk sizing, and `_ingest_v1`'s cache directory is a hardcoded module constant shared across every caller.

## Design Patterns

- **Factory + Singleton** — `@lru_cache` on `get_llm()`, `get_vector_store(strategy)`, `get_pipeline_v1/v2/v3()` for efficient reuse
- **Strategy Pattern** — v1/v2/v3 routing through request body, separate FAISS indexes, separate ingestion pipelines
- **Abstract Base Classes** — pluggable providers for LLM, embeddings, and vector store
- **Lifespan Startup** — all dependencies eagerly initialized at server start (no cold-start latency on first request)

## License

Private project.
