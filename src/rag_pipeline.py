from typing import Dict, List
from langchain_core.documents import Document
from src.llm import get_llm


def format_docs(docs: List[Document]) -> str:
    """
    Converts retrieved documents into readable context.
    """
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown source")
        content = doc.page_content
        formatted.append(f"[Local Source {i}: {source}]\n{content}")

    return "\n\n".join(formatted)


def format_web_results(results: list) -> str:
    """
    Converts Tavily web results into context.
    """
    formatted = []

    for i, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        formatted.append(f"[Web Source {i}: {title}]\nURL: {url}\n{content}")

    return "\n\n".join(formatted)


def generate_answer(
    query: str,
    local_docs: List[Document],
    web_results: list,
    answer_style: str = "Detailed report"
) -> Dict:
    """
    Generates a final answer using local document context and web context.
    """
    llm = get_llm()

    local_context = format_docs(local_docs)
    web_context = format_web_results(web_results)

    prompt = f"""
You are a Pharma & Data Intelligence Assistant.

Your job is to answer the user's question using the provided local document context and live web search context.

User question:
{query}

Answer style:
{answer_style}

Local document context:
{local_context if local_context else "No local document context was used."}

Live web context:
{web_context if web_context else "No live web context was used."}

Instructions:
1. Give a clear answer.
2. Use a professional, analytical tone.
3. Prioritise pharma, data quality, governance, compliance, analytics, and business impact.
4. Separate facts from interpretation.
5. Include a section called "Evidence Used".
6. Include a section called "Limitations".
7. Do not invent sources.
8. If evidence is weak, say so clearly.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "local_sources": local_docs,
        "web_sources": web_results
    }