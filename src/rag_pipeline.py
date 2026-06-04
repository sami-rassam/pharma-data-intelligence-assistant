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
    Converts Tavily web results into readable context.
    """
    formatted = []

    for i, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        formatted.append(f"[Web Source {i}: {title}]\nURL: {url}\n{content}")

    return "\n\n".join(formatted)


def get_style_instruction(answer_style: str) -> str:
    """
    Returns formatting and tone instructions based on the selected answer style.
    """
    if answer_style == "Simplified non-technical explanation":
        return """
Write the answer in simple, plain English for a non-technical audience.
Avoid jargon where possible.
If technical terms are needed, explain them briefly.
Use short paragraphs.
Focus on what the findings mean, why they matter, and what action could be taken.
Use practical examples where helpful.
Avoid overly technical implementation detail unless it is necessary.
"""

    if answer_style == "Executive summary":
        return """
Write a concise executive summary.
Focus on key findings, risks, implications, and recommendations.
Use clear business language.
Keep the answer suitable for senior stakeholders.
"""

    if answer_style == "Bullet point summary":
        return """
Write the answer mainly as bullet points.
Keep each bullet clear, specific, and easy to scan.
Group related points under short headings.
"""

    if answer_style == "Interview-style explanation":
        return """
Explain the answer as if preparing someone for a job interview.
Use confident, professional language.
Include examples of how the user could explain the concept to an interviewer.
"""

    if answer_style == "Technical analysis":
        return """
Write a detailed technical analysis.
Use appropriate terminology for data analytics, AI, governance, compliance, and pharmaceutical quality.
Include technical reasoning, risks, assumptions, and possible implementation considerations.
"""

    return """
Write a detailed structured report.
Use a professional, analytical tone.
Include clear headings and practical recommendations.
"""


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
    style_instruction = get_style_instruction(answer_style)

    prompt = f"""
You are a Pharma & Data Intelligence Assistant.

Your job is to answer the user's question using the provided local document context and live web search context.

User question:
{query}

Selected answer style:
{answer_style}

Style instruction:
{style_instruction}

Local document context:
{local_context if local_context else "No local document context was used."}

Live web context:
{web_context if web_context else "No live web context was used."}

Instructions:
1. Give a clear answer.
2. Follow the selected answer style closely.
3. Prioritise pharma, data quality, governance, compliance, analytics, and business impact.
4. Separate facts from interpretation.
5. Include a section called "Evidence Used".
6. Include a section called "Limitations".
7. Do not invent sources.
8. If evidence is weak, say so clearly.
9. If writing for a non-technical audience, explain what the findings mean in practical terms.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "local_sources": local_docs,
        "web_sources": web_results
    }