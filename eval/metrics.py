"""Retrieval-quality proxies that don't depend on a shared 'gold chunk id' --
each strategy chunks differently, so we score against the retrieved text/metadata
instead of trying to match chunk boundaries across strategies.

Two different questions, two different metrics -- keep them separate:
- context_recall: was the right fact anywhere in what got retrieved? (retrieval quality)
- answer_recall: did the final reply actually say the right fact? (end-to-end correctness)
A case can score 1.0 on the first and 0.0 on the second -- e.g. the right row was retrieved
alongside several confusable near-duplicates (other years/quarters), and the LLM picked a
wrong one anyway. That gap IS the finding in those cases, not a bug in the metric."""

def _keyword_hits(text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    text = text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text)
    return hits / len(expected_keywords)

def context_recall(context_docs: list, expected_keywords: list[str]) -> float:
    return _keyword_hits(" ".join(d.page_content for d in context_docs), expected_keywords)

def answer_recall(reply: str, expected_keywords: list[str]) -> float:
    return _keyword_hits(reply, expected_keywords)

def source_hit(context_docs: list, expected_source_contains: str | None) -> bool:
    if not expected_source_contains:
        return True
    needle = expected_source_contains.lower()
    return any(needle in str(d.metadata.get("file_path", "")).lower() for d in context_docs)
