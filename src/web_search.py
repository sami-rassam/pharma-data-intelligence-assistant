import os
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def get_secret(key: str):
    """
    Gets secrets from Streamlit Cloud first, then falls back to local .env.
    """
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


def tavily_search(query: str, max_results: int = 5) -> list:
    api_key = get_secret("TAVILY_API_KEY")

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