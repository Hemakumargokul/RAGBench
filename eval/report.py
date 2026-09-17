from collections import defaultdict

_REFUSAL_PHRASES = ("don't have enough information", "do not have enough information",
                     "not enough information", "cannot answer", "no information")


def _avg(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 2) if values else None


def _pct(values):
    values = [v for v in values if v is not None]
    return round(100 * sum(1 for v in values if v) / len(values), 1) if values else None


def _is_refusal(reply: str) -> bool:
    reply = reply.lower()
    return any(p in reply for p in _REFUSAL_PHRASES)


def _cell_mark(r: dict) -> str:
    """One glanceable symbol per (question, strategy): did it find the fact, and did
    it actually say the fact? These can differ -- that gap is a real finding, not noise."""
    if r["category"] == "adversarial":
        return "correctly refused" if _is_refusal(r["reply"]) else "did NOT refuse (should have)"
    if r["context_recall"] >= 1.0 and r["answer_recall"] >= 1.0:
        return "right"
    if r["context_recall"] >= 1.0 and r["answer_recall"] < 1.0:
        return f"retrieved right info, ANSWERED WRONG (ctx={r['context_recall']:.2g})"
    if r["context_recall"] > 0:
        return f"partial (ctx={r['context_recall']:.2g}, ans={r['answer_recall']:.2g})"
    return "never retrieved the fact"


def _score_row(rows):
    return {
        "n": len(rows),
        "context_recall": _avg([r["context_recall"] for r in rows]),
        "answer_recall": _avg([r["answer_recall"] for r in rows]),
        "source_hit_pct": _pct([r["source_hit"] for r in rows]),
        "faithfulness": _avg([r["faithfulness"] for r in rows]),
        "relevance": _avg([r["relevance"] for r in rows]),
        "completeness": _avg([r["completeness"] for r in rows]),
        "retrieval_ms": _avg([r["retrieval_ms"] for r in rows]),
        "generation_ms": _avg([r["generation_ms"] for r in rows]),
    }


def build_report(results: list[dict], known_issues: list[dict]) -> str:
    lines = ["# RAG Strategy Evaluation Report", ""]

    has_judge = any(r.get("faithfulness") is not None for r in results)
    if not has_judge:
        lines.append("_LLM-as-judge scoring disabled for this run (`--with-judge` to enable) -- "
                     "scores below are deterministic: **context_recall** (did retrieval find the fact?) "
                     "and **answer_recall** (did the final reply actually say the fact?). These can differ "
                     "-- see the per-question table below._")
        lines.append("")

    strategies = sorted({r["strategy"] for r in results})
    domains = sorted({r["domain"] for r in results})
    categories = sorted({r["category"] for r in results})
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    # ---- Known issues (proven, not claimed) ----
    lines += ["## Known Issues (v1/v2 baseline) -- proven this run", ""]
    for issue in known_issues:
        mark = "CONFIRMED" if issue["reproduced"] else "not reproduced this run"
        lines.append(f"- **{issue['name']}** -- {mark}")
        lines.append(f"  - {issue['detail']}")
    lines.append("")

    # ---- THE MAIN TABLE: one row per question, one column per strategy ----
    lines += ["## Per-question comparison -- which strategy got which case right", "",
              "Read this table first. Each cell is `context_recall/answer_recall` plus what that means in "
              "plain terms. `context_recall`=1.0 means the right fact was somewhere in what got retrieved; "
              "`answer_recall`=1.0 means the final reply actually said it. A cell can have the fact retrieved "
              "but still answer wrong (confusable near-duplicates, e.g. same row from a different year) -- that's "
              "flagged explicitly, not hidden in an average. Full retrieved chunks for any row are in the "
              "\"Full trace\" section below, searchable by the case id in brackets.", ""]
    lines.append("| Case | Category | Question | " + " | ".join(strategies) + " |")
    lines.append("|---|---|---|" + "---|" * len(strategies))
    by_domain_case = defaultdict(list)
    for r in results:
        by_domain_case[(r["domain"], r["case_id"])].append(r)
    for domain in domains:
        case_ids = sorted({cid for d, cid in by_domain_case if d == domain})
        for case_id in case_ids:
            rows = {r["strategy"]: r for r in by_domain_case[(domain, case_id)]}
            first = rows[strategies[0]]
            question = first["question"]
            q_short = (question[:70] + "...") if len(question) > 70 else question
            cat_label = first["category"] + (f"/{first['subtype']}" if first.get("subtype") else "")
            cells = [_cell_mark(rows[s]) if s in rows else "-" for s in strategies]
            lines.append(f"| `{domain}/{case_id}` | {cat_label} | {q_short} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Per-strategy summary (aggregate -- context for the table above, not a substitute) ----
    judge_header = " Faithfulness | Relevance | Completeness |" if has_judge else ""
    judge_sep = "---|---|---|" if has_judge else ""
    lines += ["## Per-strategy summary (aggregate)", "",
              f"| Strategy | n | Context recall | Answer recall | Source hit % |{judge_header} Avg retrieval ms | Avg generation ms |",
              f"|---|---|---|---|---|{judge_sep}---|---|"]
    for s in strategies:
        row = _score_row(by_strategy[s])
        judge_cells = f" {row['faithfulness']} | {row['relevance']} | {row['completeness']} |" if has_judge else ""
        lines.append(f"| {s} | {row['n']} | {row['context_recall']} | {row['answer_recall']} | {row['source_hit_pct']} |{judge_cells} "
                     f"{row['retrieval_ms']} | {row['generation_ms']} |")
    lines.append("")

    # ---- Per-category x strategy breakdown ----
    lines += ["## Per-category breakdown (context_recall / answer_recall)", "",
              "| Category | " + " | ".join(strategies) + " |",
              "|---|" + "---|" * len(strategies)]
    by_cat_strategy = defaultdict(list)
    for r in results:
        by_cat_strategy[(r["category"], r["strategy"])].append(r)
    for cat in categories:
        cells = []
        for s in strategies:
            rows = by_cat_strategy[(cat, s)]
            if not rows:
                cells.append("-")
            else:
                cells.append(f"{_avg([r['context_recall'] for r in rows])} / {_avg([r['answer_recall'] for r in rows])}")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Strategy progression (data-driven, not narrative) ----
    lines += ["## Strategy progression", ""]

    def cat_recall(strategy, category, field="context_recall", subtype=None):
        rows = [r for r in results if r["strategy"] == strategy and r["category"] == category
                and (subtype is None or r.get("subtype") == subtype)]
        return _avg([r[field] for r in rows]) if rows else None

    v1_coding = cat_recall("v1", "coding")
    v2_coding = cat_recall("v2", "coding")
    lines.append(f"- **v1 -> v2** (`coding` context_recall, same PDFs both strategies ingest): "
                 f"{v1_coding} -> {v2_coding} -- BM25/FAISS hybrid retrieval vs. pure vector search, on rare exact-token "
                 f"lookups (e.g. a class name) that don't carry much semantic meaning on their own.")

    v2_single = cat_recall("v2", "table", subtype="single_row")
    v3_single = cat_recall("v3", "table", subtype="single_row")
    v3_single_ans = cat_recall("v3", "table", field="answer_recall", subtype="single_row")
    lines.append(f"- **v2 -> v3** (`table`/`single_row` context_recall, same `company_overview.pdf` table data "
                 f"both strategies ingest): {v2_single} -> {v3_single} -- isolated per-row nodes (via `find_tables()`) "
                 f"vs. asking the LLM to pick the right row out of a ~14-row flattened text block. "
                 f"**But** v3's answer_recall on the same cases is only {v3_single_ans} -- retrieval finding the "
                 f"right row doesn't guarantee the reply uses it, when confusable near-duplicate rows (other years) "
                 f"are retrieved alongside it. See the per-question table for exactly which case.")

    v2_agg = cat_recall("v2", "table", subtype="aggregate")
    v3_agg = cat_recall("v3", "table", subtype="aggregate")
    lines.append(f"- **v2 vs v3** (`table`/`aggregate` context_recall -- an open question, not a settled win): "
                 f"v2={v2_agg}, v3={v3_agg}.")

    v2_source = _pct([r["source_hit"] for r in results if r["strategy"] == "v2"])
    v3_source = _pct([r["source_hit"] for r in results if r["strategy"] == "v3"])
    lines.append(f"- **v2 -> v3** (source-hit rate, all categories): {v2_source}% -> {v3_source}% "
                 f"-- v1/v2 only ever ingest `.pdf` files (see `required_exts`/`rglob('*.pdf')`); "
                 f"v3 additionally ingests the `.csv`/`.json`/`.html` files in `data/finance/`.")
    lines.append("")

    # ---- Full trace: every question, every strategy, every retrieved chunk ----
    lines += ["## Full trace (every question x strategy x retrieved chunk)", "",
              "Live detail backing the table above -- the actual retrieved chunks and reply for every case.", ""]
    for domain in domains:
        case_ids = sorted({cid for d, cid in by_domain_case if d == domain})
        for case_id in case_ids:
            rows = sorted(by_domain_case[(domain, case_id)], key=lambda r: strategies.index(r["strategy"]))
            lines.append(f"### {domain}/{case_id}: {rows[0]['question']!r}")
            lines.append("")
            for r in rows:
                lines.append(f"**{r['strategy']}** -- {_cell_mark(r)} "
                             f"(context_recall={r['context_recall']}, answer_recall={r['answer_recall']}, "
                             f"source_hit={r['source_hit']}, retrieval_ms={r['retrieval_ms']}, generation_ms={r['generation_ms']})")
                lines.append("")
                lines.append(f"> Reply: {r['reply'].strip().splitlines()[0] if r['reply'].strip() else '(empty)'}")
                lines.append("")
                chunks = r.get("retrieved_chunks", [])
                if not chunks:
                    lines.append("_No chunks retrieved._")
                else:
                    lines.append(f"Retrieved {len(chunks)} chunk(s):")
                    for i, c in enumerate(chunks):
                        content = c["content"].replace("\n", " ").strip()
                        truncated = content if len(content) <= 600 else content[:600] + " [...truncated, see raw_*.json for full text]"
                        lines.append(f"{i+1}. `{c.get('content_type')}` from `{c.get('file_path')}`:")
                        lines.append(f"   > {truncated}")
                lines.append("")

    return "\n".join(lines)
