from typing import Dict, List
from langchain_core.documents import Document
from src.llm import get_llm


def researcher_agent(query: str, local_docs: List[Document], web_results: list) -> Dict:
    """
    Researcher agent summarises the evidence gathered from local and web sources.
    """
    return {
        "query": query,
        "local_evidence_count": len(local_docs),
        "web_evidence_count": len(web_results),
        "local_docs": local_docs,
        "web_results": web_results
    }


def evidence_checker_agent(local_docs: List[Document], web_results: list) -> Dict:
    """
    Creates a simple confidence score based on available evidence.
    """
    local_count = len(local_docs)
    web_count = len(web_results)

    if local_count >= 3 and web_count >= 3:
        confidence = "High"
        explanation = "The answer is supported by several local document chunks and several live web sources."
    elif local_count >= 2 or web_count >= 2:
        confidence = "Medium"
        explanation = "The answer has some supporting evidence, but coverage may be incomplete."
    else:
        confidence = "Low"
        explanation = "Limited evidence was retrieved. The answer should be treated cautiously."

    return {
        "confidence": confidence,
        "confidence_explanation": explanation,
        "local_sources_used": local_count,
        "web_sources_used": web_count
    }


def critic_agent(answer: str) -> str:
    """
    Reviews the generated answer and suggests improvements.
    """
    llm = get_llm(temperature=0.1)

    prompt = f"""
Review the following AI-generated answer.

Check for:
1. unsupported claims
2. missing limitations
3. unclear wording
4. weak evidence
5. opportunities to make the answer more useful for pharma/data intelligence

Answer:
{answer}

Return a concise critique with suggested improvements.
"""

    response = llm.invoke(prompt)
    return response.content


def final_writer_agent(answer: str, confidence_report: Dict, critique: str) -> str:
    """
    Produces the final polished answer.
    """
    llm = get_llm(temperature=0.2)

    prompt = f"""
Improve the answer below using the confidence report and critique.

Original answer:
{answer}

Confidence report:
{confidence_report}

Critique:
{critique}

Return the final version with these sections:
1. Final Answer
2. Evidence Used
3. Confidence Assessment
4. Limitations
5. Recommended Next Steps
"""

    response = llm.invoke(prompt)
    return response.content