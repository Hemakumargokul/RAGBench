import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from config import settings
from services.llm.factory import get_llm
from services.vector_store.factory import get_vector_store
from services.ingestion_service import get_pipeline_v1, get_pipeline_v2, get_pipeline_v3, get_pipeline_v4, ingest
from services.rag_service import ask_with_trace
from eval.metrics import context_recall, answer_recall, source_hit
from eval.judge import judge_answer
from eval.known_issues import run_known_issues
from eval.report import build_report

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).parent
DOMAIN_DIRS = {"python": "data/python", "kubernetes": "data/kubernetes", "finance": "data/finance"}
PIPELINE_GETTERS = {"v1": get_pipeline_v1, "v2": get_pipeline_v2, "v3": get_pipeline_v3, "v4": get_pipeline_v4}


def load_testset(domain: str) -> list[dict]:
    return json.loads((_EVAL_DIR / "testsets" / f"{domain}.json").read_text())


def _is_domain_file(file_path, domain_dir_resolved: str) -> bool:
    if not file_path:
        return False
    try:
        return str(Path(file_path).resolve()).startswith(domain_dir_resolved)
    except OSError:
        return False

def ensure_ingested(domain: str, strategy: str):
    vector_store = get_vector_store(strategy)
    # Resolve both sides before comparing: strategies store file_path inconsistently
    # (SimpleDirectoryReader/v1 stores absolute paths; v2/v3/v4 store whatever path
    # string was passed to ingest(), which is relative here) -- comparing raw strings
    # silently fails for the relative ones and re-ingests (duplicating) every run.
    domain_dir_resolved = str(Path(DOMAIN_DIRS[domain]).resolve())
    already = any(_is_domain_file(d.metadata.get("file_path"), domain_dir_resolved) for d in vector_store.get_all_documents())
    if not already:
        pipeline = PIPELINE_GETTERS[strategy]()
        n = ingest(DOMAIN_DIRS[domain], vector_store, pipeline, strategy)
        print(f"  ingested {domain}/{strategy}: {n} chunks")
    return vector_store


async def run(domains: list[str], strategies: list[str], with_judge: bool) -> list[dict]:
    llm = get_llm()
    results = []
    for domain in domains:
        testset = load_testset(domain)
        for strategy in strategies:
            print(f"== {domain} / {strategy} ==")
            vector_store = ensure_ingested(domain, strategy)
            for case in testset:
                result = await ask_with_trace(case["question"], llm, vector_store, strategy)
                # LLM-as-judge (faithfulness/relevance/completeness) is off by default: with only
                # 5-6 questions per domain, judge noise on refusal-type answers swings the aggregate
                # more than real strategy differences do. Re-enable with --with-judge once the
                # retrieval-quality signal (keyword_recall/source_hit) is trusted on its own.
                if with_judge:
                    judged = await judge_answer(llm, case["question"], result["context_docs"], result["reply"])
                else:
                    judged = {"faithfulness": None, "relevance": None, "completeness": None, "rationale": None}
                results.append({
                    "domain": domain,
                    "strategy": strategy,
                    "case_id": case["id"],
                    "category": case["category"],
                    "subtype": case.get("subtype"),
                    "question": case["question"],
                    "reply": result["reply"],
                    "retrieval_ms": round(result["retrieval_ms"], 1),
                    "generation_ms": round(result["generation_ms"], 1),
                    "context_recall": context_recall(result["context_docs"], case.get("expected_keywords", [])),
                    "answer_recall": answer_recall(result["reply"], case.get("expected_keywords", [])),
                    "source_hit": source_hit(result["context_docs"], case.get("expected_source_contains")),
                    "retrieved_chunks": [
                        {
                            "content": d.page_content,
                            "content_type": d.metadata.get("content_type"),
                            "file_path": d.metadata.get("file_path"),
                        }
                        for d in result["context_docs"]
                    ],
                    **judged,
                })
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG strategies (v1/v2/v3) across domains.")
    parser.add_argument("--domains", default="python,kubernetes,finance")
    parser.add_argument("--strategies", default="v1,v2,v3")
    parser.add_argument("--skip-known-issues", action="store_true")
    parser.add_argument("--with-judge", action="store_true", help="Also score answers with the LLM-as-judge (faithfulness/relevance/completeness). Off by default -- see the comment in run().")
    args = parser.parse_args()

    domains = args.domains.split(",")
    strategies = args.strategies.split(",")

    if not settings.OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set -- required for embeddings and retrieval (and the LLM judge, if --with-judge is passed).")

    results = asyncio.run(run(domains, strategies, args.with_judge))
    known_issues = [] if args.skip_known_issues else run_known_issues()

    out_dir = _EVAL_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    (out_dir / f"raw_{ts}.json").write_text(json.dumps({"results": results, "known_issues": known_issues}, indent=2))

    report_md = build_report(results, known_issues)
    (out_dir / "report.md").write_text(report_md)
    print(f"\nWrote {out_dir / f'raw_{ts}.json'}")
    print(f"Wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
