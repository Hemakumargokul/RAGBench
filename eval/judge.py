import json
import logging
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = ChatPromptTemplate.from_template("""
You are grading a RAG system's answer strictly against the context it was given -- not your own knowledge.

Question: {question}

Context given to the model:
{context}

Model's answer:
{answer}

Return ONLY a JSON object (no markdown fence, no other text) with this exact shape:
{{"faithfulness": <1-5 int>, "relevance": <1-5 int>, "completeness": <1-5 int>, "rationale": "<one sentence>"}}

faithfulness: does the answer avoid claiming things not supported by the given context?
relevance: does the answer actually address the question asked?
completeness: does the answer use the relevant information that IS present in the context (not information it couldn't have had)?
If the context is empty or irrelevant and the model correctly said it doesn't have enough information, score all three 5.
""")

async def judge_answer(llm, question: str, context_docs: list, answer: str) -> dict:
    context = "\n\n".join(d.page_content for d in context_docs) if context_docs else "(empty -- no chunks retrieved)"
    prompt_value = _JUDGE_PROMPT.invoke({"question": question, "context": context, "answer": answer})
    raw = await (llm | StrOutputParser()).ainvoke(prompt_value)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("judge returned unparseable JSON: %s", raw[:200])

    return {
        "faithfulness": data.get("faithfulness"),
        "relevance": data.get("relevance"),
        "completeness": data.get("completeness"),
        "rationale": data.get("rationale", ""),
    }
