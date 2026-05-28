import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def tavily_search(query: str, max_results: int = 5) -> list:
    """
    Runs live web search using Tavily.
    """
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return []

    client = TavilyClient(api_key=api_key)

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=True,
        include_raw_content=False
    )

    return response.get("results", [])