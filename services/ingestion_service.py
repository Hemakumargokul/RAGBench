import os
import json
import logging
from pathlib import Path
from functools import lru_cache

import fitz
import pandas as pd
import pymupdf4llm
from bs4 import BeautifulSoup
from google.cloud import documentai
from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, MarkdownNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

from services.vector_store.base import BaseVectorStore
from config import settings

logger = logging.getLogger(__name__)
_CACHE_DIR = "ingestion_cache"

def _tag_content_type(node):
    text = node.get_content()
    if "```" in text or "\n    " in text[:200]:
        node.metadata["content_type"] = "code"
    elif text.count("|") > 3:
        node.metadata["content_type"] = "table"
    else:
        node.metadata["content_type"] = "prose"
    return node

@lru_cache(maxsize=None)
def get_pipeline_v1() -> IngestionPipeline:
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1000, chunk_overlap=200),
            OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY),
        ]
    )
    if os.path.exists(os.path.join(_CACHE_DIR, "llama_cache")):
        pipeline.load(persist_dir=_CACHE_DIR)
    return pipeline

@lru_cache(maxsize=None)
def get_pipeline_v2() -> IngestionPipeline:
    return IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY),
        ]
    )

@lru_cache(maxsize=None)
def get_pipeline_v3() -> IngestionPipeline:
    # Chunking is format-aware and done before this pipeline runs (see _ingest_v3):
    # rows/records are already one-node-per-chunk, and prose already went through
    # MarkdownNodeParser. This pipeline only computes embeddings.
    return IngestionPipeline(
        transformations=[
            OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY),
        ]
    )

@lru_cache(maxsize=None)
def get_pipeline_v4() -> IngestionPipeline:
    # Same as v3: chunking already happened (this time via GCP Document AI's Layout
    # Parser instead of pymupdf4llm+find_tables()), this pipeline only embeds.
    return IngestionPipeline(
        transformations=[
            OpenAIEmbedding(model="text-embedding-3-small", api_key=settings.OPENAI_API_KEY),
        ]
    )

def _ingest_v1(directory: str, vector_store: BaseVectorStore, pipeline: IngestionPipeline) -> int:
    from llama_index.core import SimpleDirectoryReader
    docs = SimpleDirectoryReader(directory, required_exts=[".pdf"], recursive=True).load_data()
    logger.info("v1: Loaded %d documents", len(docs))
    nodes = pipeline.run(documents=docs)
    pipeline.cache.persist(_CACHE_DIR)
    logger.info("v1: Pipeline produced %d nodes", len(nodes))
    texts_and_embeddings = [(n.get_content(), n.embedding) for n in nodes if n.embedding]
    metadatas = [n.metadata for n in nodes if n.embedding]
    vector_store.add_embeddings(texts_and_embeddings, metadatas)
    return len(texts_and_embeddings)

def _ingest_v2(directory: str, vector_store: BaseVectorStore, pipeline: IngestionPipeline) -> int:
    docs = []
    for path in Path(directory).rglob("*.pdf"):
        md = pymupdf4llm.to_markdown(str(path))
        docs.append(Document(text=md, metadata={"file_path": str(path)}))
    if not docs:
        raise ValueError(f"No PDF files found in {directory}.")
    logger.info("v2: Loaded %d PDFs as markdown", len(docs))
    nodes = pipeline.run(documents=docs)
    logger.info("v2: Pipeline produced %d nodes", len(nodes))
    nodes = [_tag_content_type(n) for n in nodes if n.embedding]
    texts_and_embeddings = [(n.get_content(), n.embedding) for n in nodes]
    metadatas = [n.metadata for n in nodes]
    vector_store.add_embeddings(texts_and_embeddings, metadatas)
    return len(nodes)

def _pdf_tables_v3(path: Path) -> list[Document]:
    """Extract real tables from a PDF via PyMuPDF's layout analysis (independent of
    the markdown-flattening pymupdf4llm does), one node per row so numeric lookups
    keep their column alignment instead of being buried in a text blob."""
    docs = []
    pdf = fitz.open(str(path))
    for page_num, page in enumerate(pdf):
        for table in page.find_tables().tables:
            rows = table.extract()
            if len(rows) < 2:
                continue
            header = [str(h).strip() if h else f"col{i}" for i, h in enumerate(rows[0])]
            for row in rows[1:]:
                cells = [f"{h}: {c}" for h, c in zip(header, row) if c is not None]
                if not cells:
                    continue
                docs.append(Document(
                    text=" | ".join(cells),
                    metadata={"file_path": str(path), "page": page_num, "content_type": "table_row"},
                ))
    pdf.close()
    return docs

def _tabular_rows_v3(path: Path) -> list[Document]:
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    docs = []
    for i, row in df.iterrows():
        text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        docs.append(Document(
            text=text,
            metadata={"file_path": str(path), "row_index": int(i), "content_type": "table_row"},
        ))
    return docs

def _json_records_v3(path: Path) -> list[Document]:
    data = json.loads(path.read_text())
    records = data if isinstance(data, list) else [data]
    docs = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            record = {"value": record}
        text = "\n".join(f"{k}: {v}" for k, v in record.items())
        docs.append(Document(
            text=text,
            metadata={"file_path": str(path), "record_index": i, "content_type": "json_record"},
        ))
    return docs

def _ingest_v3(directory: str, vector_store: BaseVectorStore, pipeline: IngestionPipeline) -> int:
    prose_docs: list[Document] = []
    struct_docs: list[Document] = []
    for path in Path(directory).rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext == ".pdf":
            md = pymupdf4llm.to_markdown(str(path))
            prose_docs.append(Document(text=md, metadata={"file_path": str(path), "content_type": "prose_source"}))
            struct_docs.extend(_pdf_tables_v3(path))
        elif ext == ".md":
            prose_docs.append(Document(text=path.read_text(), metadata={"file_path": str(path), "content_type": "prose_source"}))
        elif ext == ".html":
            text = BeautifulSoup(path.read_text(), "html.parser").get_text("\n")
            prose_docs.append(Document(text=text, metadata={"file_path": str(path), "content_type": "prose_source"}))
        elif ext in (".csv", ".xlsx"):
            struct_docs.extend(_tabular_rows_v3(path))
        elif ext == ".json":
            struct_docs.extend(_json_records_v3(path))

    if not prose_docs and not struct_docs:
        raise ValueError(f"No supported files found in {directory}.")

    prose_nodes = MarkdownNodeParser().get_nodes_from_documents(prose_docs) if prose_docs else []
    prose_nodes = [_tag_content_type(n) for n in prose_nodes]
    logger.info("v3: %d prose nodes + %d structured (table/json) nodes before embedding",
                len(prose_nodes), len(struct_docs))

    all_nodes = prose_nodes + struct_docs
    nodes = pipeline.run(documents=all_nodes)
    dropped = len(nodes) - sum(1 for n in nodes if n.embedding)
    if dropped:
        logger.warning("v3: %d/%d nodes had no embedding and were dropped", dropped, len(nodes))

    texts_and_embeddings = [(n.get_content(), n.embedding) for n in nodes if n.embedding]
    metadatas = [n.metadata for n in nodes if n.embedding]
    vector_store.add_embeddings(texts_and_embeddings, metadatas)
    return len(texts_and_embeddings)

@lru_cache(maxsize=None)
def _document_ai_client() -> tuple[documentai.DocumentProcessorServiceClient, str]:
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(
        settings.DOCUMENT_AI_PROJECT_ID, settings.DOCUMENT_AI_LOCATION, settings.DOCUMENT_AI_PROCESSOR_ID
    )
    return client, name

def _cell_text(cell) -> str:
    return " ".join(
        b.text_block.text for b in cell.blocks if b._pb.WhichOneof("block") == "text_block"
    ).strip()

def _walk_layout_blocks(blocks, heading_stack, headers_by_heading, prose_by_heading, table_docs, path):
    for block in blocks:
        kind = block._pb.WhichOneof("block")
        if kind == "text_block":
            tb = block.text_block
            if tb.type_.startswith("heading"):
                _walk_layout_blocks(tb.blocks, heading_stack + [tb.text], headers_by_heading, prose_by_heading, table_docs, path)
            else:
                heading_label = " / ".join(heading_stack) if heading_stack else "(no heading)"
                if tb.text.strip():
                    prose_by_heading.setdefault(heading_label, []).append(tb.text)
                _walk_layout_blocks(tb.blocks, heading_stack, headers_by_heading, prose_by_heading, table_docs, path)
        elif kind == "table_block":
            # Document AI detects one table_block per physical page here (our source PDF
            # repeats the same heading text on every page of a multi-page table), so rows
            # for the "same" table are spread across several table_blocks sharing a heading
            # -- headers_by_heading lets a later page's rows reuse the header row detected
            # on an earlier page of the same section.
            t = block.table_block
            heading_label = " / ".join(heading_stack) if heading_stack else "(no heading)"
            body_rows = list(t.body_rows)
            if t.header_rows:
                headers_by_heading[heading_label] = [_cell_text(c) for c in t.header_rows[0].cells]
            elif heading_label not in headers_by_heading and body_rows:
                # Document AI didn't detect a header row for this table at all -- heuristically
                # treat the first body row as the header the first time we see this heading,
                # rather than emitting it as a garbage data row with no column labels.
                headers_by_heading[heading_label] = [_cell_text(c) for c in body_rows[0].cells]
                body_rows = body_rows[1:]
            header = headers_by_heading.get(heading_label)
            for row in body_rows:
                cells = [_cell_text(c) for c in row.cells]
                if header and len(header) == len(cells):
                    text = " | ".join(f"{h}: {v}" for h, v in zip(header, cells) if v)
                else:
                    text = " | ".join(c for c in cells if c)
                if text:
                    table_docs.append(Document(
                        text=text,
                        metadata={"file_path": path, "heading": heading_label, "content_type": "table_row"},
                    ))
        elif kind == "list_block":
            heading_label = " / ".join(heading_stack) if heading_stack else "(no heading)"
            for item in block.list_block.list_entries:
                text = " ".join(_cell_text_from_blocks(item.blocks))
                if text.strip():
                    prose_by_heading.setdefault(heading_label, []).append(text)

def _cell_text_from_blocks(blocks) -> list[str]:
    return [b.text_block.text for b in blocks if b._pb.WhichOneof("block") == "text_block"]

def _pdf_to_nodes_v4(path: Path) -> tuple[list[Document], list[Document]]:
    """Parse a PDF via GCP Document AI's Layout Parser instead of pymupdf4llm+find_tables() --
    a managed cloud layout model (does real OCR, and stitches structure across a page) rather
    than our own local heuristics."""
    client, processor_name = _document_ai_client()
    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=documentai.RawDocument(content=path.read_bytes(), mime_type="application/pdf"),
    )
    doc = client.process_document(request=request).document

    headers_by_heading: dict[str, list[str]] = {}
    prose_by_heading: dict[str, list[str]] = {}
    table_docs: list[Document] = []
    _walk_layout_blocks(doc.document_layout.blocks, [], headers_by_heading, prose_by_heading, table_docs, str(path))

    prose_docs = [
        Document(text=f"{heading}\n\n{' '.join(parts)}", metadata={"file_path": str(path), "content_type": "prose_source", "heading": heading})
        for heading, parts in prose_by_heading.items()
    ]
    return prose_docs, table_docs

def _ingest_v4(directory: str, vector_store: BaseVectorStore, pipeline: IngestionPipeline) -> int:
    prose_docs: list[Document] = []
    struct_docs: list[Document] = []
    for path in Path(directory).rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext == ".pdf":
            pdf_prose, pdf_tables = _pdf_to_nodes_v4(path)
            prose_docs.extend(pdf_prose)
            struct_docs.extend(pdf_tables)
        elif ext == ".md":
            prose_docs.append(Document(text=path.read_text(), metadata={"file_path": str(path), "content_type": "prose_source"}))
        elif ext == ".html":
            text = BeautifulSoup(path.read_text(), "html.parser").get_text("\n")
            prose_docs.append(Document(text=text, metadata={"file_path": str(path), "content_type": "prose_source"}))
        elif ext in (".csv", ".xlsx"):
            struct_docs.extend(_tabular_rows_v3(path))
        elif ext == ".json":
            struct_docs.extend(_json_records_v3(path))

    if not prose_docs and not struct_docs:
        raise ValueError(f"No supported files found in {directory}.")

    # prose_docs are already one-per-heading (Document AI's own layout sectioning) --
    # no further splitting needed, unlike v3's MarkdownNodeParser pass.
    prose_nodes = [_tag_content_type(d) for d in prose_docs]
    logger.info("v4: %d prose nodes + %d structured (table) nodes before embedding",
                len(prose_nodes), len(struct_docs))

    all_nodes = prose_nodes + struct_docs
    nodes = pipeline.run(documents=all_nodes)
    dropped = len(nodes) - sum(1 for n in nodes if n.embedding)
    if dropped:
        logger.warning("v4: %d/%d nodes had no embedding and were dropped", dropped, len(nodes))

    texts_and_embeddings = [(n.get_content(), n.embedding) for n in nodes if n.embedding]
    metadatas = [n.metadata for n in nodes if n.embedding]
    vector_store.add_embeddings(texts_and_embeddings, metadatas)
    return len(texts_and_embeddings)

def ingest(directory: str, vector_store: BaseVectorStore, pipeline: IngestionPipeline, strategy: str) -> int:
    if strategy == "v4":
        return _ingest_v4(directory, vector_store, pipeline)
    if strategy == "v3":
        return _ingest_v3(directory, vector_store, pipeline)
    if strategy == "v2":
        return _ingest_v2(directory, vector_store, pipeline)
    return _ingest_v1(directory, vector_store, pipeline)
