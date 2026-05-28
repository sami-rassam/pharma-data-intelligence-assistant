import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


def get_llm(temperature: float = 0.2):
    """
    Creates the Groq chat model.
    """
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=temperature
    )