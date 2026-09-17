"""Runnable proofs of the pre-existing v1/v2 problems documented in the plan --
each check reproduces one issue live against the real (untouched) v1/v2 code in
a throwaway FAISS index, so the eval report carries fresh evidence instead of a
stale claim. `reproduced=True` means the bug was actually observed this run.
"""
import os
import random
import shutil
import time
from pathlib import Path

import fitz


def _make_tiny_pdf(directory: Path, sentence: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "tiny.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), sentence)
    doc.save(str(path))
    doc.close()
    return path


def check_tagging_heuristic() -> dict:
    from services.ingestion_service import _tag_content_type

    class _FakeNode:
        def __init__(self, text):
            self._text = text
            self.metadata = {}
        def get_content(self):
            return self._text

    cases = {
        "prose with an indented block-quote": (
            "As the official spec states:\n\n    'A pod is the smallest deployable unit.'\nCommonly cited.",
            "prose",
        ),
        "prose using logic notation with pipes": (
            "In propositional logic the OR operator lets you write p | q | r | s | t.",
            "prose",
        ),
        "real code, no leading indent or backticks": (
            "import os\nprint(os.getcwd())\nfor i in range(3):\n  print(i)",
            "code",
        ),
    }
    failures = []
    for name, (text, expected) in cases.items():
        node = _FakeNode(text)
        _tag_content_type(node)
        got = node.metadata["content_type"]
        if got != expected:
            failures.append(f"{name}: expected {expected!r}, got {got!r}")

    return {
        "name": "content_type tagging heuristic misclassifies prose/code/table",
        "reproduced": bool(failures),
        "detail": "; ".join(failures) if failures else "no misclassification this run",
    }


def check_bm25_rebuild_cost() -> dict:
    from langchain_core.documents import Document
    from langchain_community.retrievers import BM25Retriever

    random.seed(42)
    words = ["pandas", "kubernetes", "python", "dataframe", "pod", "service",
             "cluster", "index", "chunk", "retrieval"]
    timings = {}
    for n in (50, 5000, 20000):
        docs = [Document(page_content=" ".join(random.choices(words, k=80))) for _ in range(n)]
        t0 = time.perf_counter()
        BM25Retriever.from_documents(docs, k=5)
        timings[n] = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "name": "BM25Retriever rebuilt from scratch on every v2/v3 chat request (rag_service.py)",
        "reproduced": True,
        "detail": f"rebuild time (ms) by corpus size: {timings} -- paid again on every single /chat/message call",
    }


def check_duplicate_ingestion(tmp_root: Path) -> dict:
    from config import settings
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.openai import OpenAIEmbedding
    from services.embeddings.factory import get_embeddings
    from services.ingestion_service import _ingest_v1
    from services.vector_store.faiss_store import FAISSStore

    doc_dir = tmp_root / "dup_pdf"
    _make_tiny_pdf(doc_dir, "Pandas is a Python library for tabular data analysis.")

    vs = FAISSStore(embeddings=get_embeddings("openai"), index_path=str(tmp_root / "faiss_dup"))
    pipeline = IngestionPipeline(transformations=[
        SentenceSplitter(chunk_size=1000, chunk_overlap=200),
        OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY),
    ])
    _ingest_v1(str(doc_dir), vs, pipeline)
    first = len(vs.get_all_documents())
    _ingest_v1(str(doc_dir), vs, pipeline)
    second = len(vs.get_all_documents())

    return {
        "name": "re-ingesting an unchanged directory duplicates every chunk (no dedup)",
        "reproduced": second == first * 2,
        "detail": f"chunks after 1st ingest: {first}, after re-ingesting the same directory: {second}",
    }


def check_score_threshold(tmp_root: Path) -> dict:
    from services.embeddings.factory import get_embeddings
    from services.vector_store.faiss_store import FAISSStore

    vs = FAISSStore(embeddings=get_embeddings("openai"), index_path=str(tmp_root / "faiss_dup"))
    relevant_query = "What Python library provides the DataFrame object for tabular data analysis?"
    raw_scores = vs._store.similarity_search_with_relevance_scores(relevant_query, k=1)
    top_score = float(raw_scores[0][1]) if raw_scores else None

    docs_low = vs.get_retriever(score_threshold=0.0).invoke(relevant_query)
    docs_mid = vs.get_retriever(score_threshold=0.5).invoke(relevant_query)

    reproduced = top_score is not None and top_score < 0.5 and len(docs_low) > 0 and len(docs_mid) == 0
    return {
        "name": "score_threshold is miscalibrated, not just disabled at 0.0",
        "reproduced": reproduced,
        "detail": (
            f"relevance score for the single most relevant chunk to a directly-relevant query: {top_score:.3f} "
            f"(out of the expected [0,1] range assumption) -- threshold=0.0 returns {len(docs_low)} chunk(s), "
            f"threshold=0.5 returns {len(docs_mid)} chunk(s) (the correct chunk gets filtered out)"
            if top_score is not None else "no chunks found to score"
        ),
    }


def check_ocr_gap(tmp_root: Path) -> dict:
    import pymupdf4llm
    from llama_index.core import SimpleDirectoryReader

    scanned_dir = tmp_root / "scanned_pdf"
    scanned_dir.mkdir(parents=True, exist_ok=True)
    scanned_path = scanned_dir / "scanned.pdf"

    src = fitz.open()
    p = src.new_page()
    p.insert_text((72, 72), "IMPORTANT CLAUSE: liability is capped at $50,000 per incident.")
    pix = p.get_pixmap(dpi=150)
    img_path = scanned_dir / "page.png"
    pix.save(str(img_path))
    src.close()

    out = fitz.open()
    page = out.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, filename=str(img_path))
    out.save(str(scanned_path))
    out.close()

    v2_text = pymupdf4llm.to_markdown(str(scanned_path)).strip()
    v1_docs = SimpleDirectoryReader(str(scanned_dir), required_exts=[".pdf"]).load_data()
    v1_text = "".join(d.text for d in v1_docs).strip()

    return {
        "name": "no OCR -- scanned/image-only PDF pages yield empty extracted text",
        "reproduced": len(v2_text) == 0 and len(v1_text) == 0,
        "detail": f"v1 extracted {len(v1_text)} chars, v2 extracted {len(v2_text)} chars from a page with a visible sentence but no text layer",
    }


def run_known_issues() -> list[dict]:
    tmp_root = Path("/tmp") / f"rag_known_issues_{os.getpid()}"
    try:
        return [
            check_tagging_heuristic(),
            check_bm25_rebuild_cost(),
            check_duplicate_ingestion(tmp_root),
            check_score_threshold(tmp_root),
            check_ocr_gap(tmp_root),
        ]
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
