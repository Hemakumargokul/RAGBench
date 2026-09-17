import time
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from services.vector_store.base import BaseVectorStore

logger = logging.getLogger(__name__)

_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the provided context.

After your answer, always include a "Context Quality" section using this format:

Context Quality:
- Summary: <one sentence describing what the retrieved context covers>
- Relevance: <High | Medium | Low> — <brief reason>
- If relevance is Medium or Low: What was missing: <what information would have been needed to answer better>

If the context is completely empty or irrelevant, respond with "I don't have enough information to answer that." and still fill in the Context Quality section.

Context:
{context}

Question: {question}

Answer:""")

def _build_retriever(vector_store: BaseVectorStore, strategy: str):
    if strategy in ("v2", "v3", "v4"):
        all_docs = vector_store.get_all_documents()
        # v1/v2 keep score_threshold=0.0 unchanged as baselines; v3/v4 use plain
        # top-k (score_threshold=None) since the threshold itself is proven broken
        # (see eval/known_issues.py) rather than genuinely disabled at 0.0.
        faiss_retriever = vector_store.get_retriever(score_threshold=None if strategy in ("v3", "v4") else 0.0)
        if all_docs:
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=5)
            retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.6, 0.4])
        else:
            retriever = faiss_retriever
        logger.info("%s: using hybrid retriever", strategy)
    else:
        retriever = vector_store.get_retriever(score_threshold=0.0)
        logger.info("v1: using FAISS retriever")
    return retriever

async def ask_with_trace(question: str, llm: BaseChatModel, vector_store: BaseVectorStore, strategy: str = "v1") -> dict:
    retriever = _build_retriever(vector_store, strategy)

    t0 = time.perf_counter()
    docs = await retriever.ainvoke(question)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    context = "\n\n".join(doc.page_content for doc in docs) if docs else ""
    logger.info("Retrieved %d chunks (strategy=%s)\nContext:\n%s", len(docs), strategy, context)

    prompt_value = _PROMPT.invoke({"context": context, "question": question})
    t0 = time.perf_counter()
    reply = await (llm | StrOutputParser()).ainvoke(prompt_value)
    generation_ms = (time.perf_counter() - t0) * 1000

    return {
        "reply": reply,
        "context_docs": docs,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
    }

async def ask(question: str, llm: BaseChatModel, vector_store: BaseVectorStore, strategy: str = "v1") -> str:
    result = await ask_with_trace(question, llm, vector_store, strategy)
    return result["reply"]
