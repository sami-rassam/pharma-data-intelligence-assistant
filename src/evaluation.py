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