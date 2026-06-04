import time
import pandas as pd


def calculate_basic_metrics(
    query: str,
    local_docs: list,
    web_results: list,
    start_time: float,
    end_time: float
) -> dict:
    """
    Creates basic evaluation metrics for each answer.
    """
    return {
        "query": query,
        "local_chunks_retrieved": len(local_docs),
        "web_results_retrieved": len(web_results),
        "response_time_seconds": round(end_time - start_time, 2),
        "used_local_rag": len(local_docs) > 0,
        "used_web_search": len(web_results) > 0
    }


def create_evaluation_dataframe(metrics_list: list) -> pd.DataFrame:
    return pd.DataFrame(metrics_list)


def retrieval_quality_label(local_docs: list, web_results: list) -> str:
    """
    Simple retrieval quality label.
    """
    total_sources = len(local_docs) + len(web_results)

    if total_sources >= 8:
        return "Strong retrieval coverage"
    elif total_sources >= 4:
        return "Moderate retrieval coverage"
    elif total_sources >= 1:
        return "Limited retrieval coverage"
    else:
        return "No retrieval coverage"
    
def calculate_automated_scores(
    local_docs: list,
    web_results: list,
    confidence: str,
    answer: str
) -> dict:
    """
    Produces simple automated evaluation scores for portfolio demonstration.
    Scores are heuristic, not formal validation.
    """

    source_count = len(local_docs) + len(web_results)

    source_coverage_score = min(source_count * 10, 100)

    if confidence == "High":
        confidence_score = 100
    elif confidence == "Medium":
        confidence_score = 70
    else:
        confidence_score = 40

    answer_length = len(answer.split())

    if answer_length >= 250:
        completeness_score = 100
    elif answer_length >= 120:
        completeness_score = 70
    else:
        completeness_score = 40

    reliability_scores = [
        result.get("reliability_score", 3)
        for result in web_results
    ]

    if reliability_scores:
        average_reliability_score = round(
            sum(reliability_scores) / len(reliability_scores),
            2
        )
        source_reliability_score = round((average_reliability_score / 5) * 100, 2)
    else:
        source_reliability_score = 50

    overall_score = round(
        (
            source_coverage_score
            + confidence_score
            + completeness_score
            + source_reliability_score
        ) / 4,
        2
    )

    return {
        "source_coverage_score": source_coverage_score,
        "confidence_score": confidence_score,
        "completeness_score": completeness_score,
        "source_reliability_score": source_reliability_score,
        "overall_answer_quality_score": overall_score
    }