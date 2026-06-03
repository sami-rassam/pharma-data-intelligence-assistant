import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_secret(key: str):
    """
    Gets secrets from Streamlit Cloud first, then falls back to local .env.
    """
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)


def get_llm(temperature: float = 0.2):
    groq_api_key = get_secret("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud secrets or your local .env file."
        )

    return ChatGroq(
        groq_api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=temperature
    )