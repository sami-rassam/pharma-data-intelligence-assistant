# Pharma & Data Intelligence Assistant

## Project Overview

The Pharma & Data Intelligence Assistant is an agentic RAG application designed to support pharmaceutical, regulatory, data quality, and analytics research.

It combines local document retrieval with live web search and LLM-based answer generation.

## Key Features

- Upload PDF, TXT, MD, and DOCX files
- Search local documents using FAISS vector search
- Use HuggingFace embeddings
- Search the live web using Tavily
- Generate answers using Groq LLMs
- Agent-style workflow:
  - Researcher Agent
  - Evidence Checker Agent
  - Writer Agent
  - Critic Agent
- Confidence scoring
- Source display
- Markdown report export
- Evaluation dashboard
- Retrieval quality metrics

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq
- Tavily
- FAISS
- HuggingFace sentence-transformers
- CrewAI-inspired agent workflow
- Pandas

## Problem Statement

Pharmaceutical and data teams often need to answer complex questions using a mixture of internal documents and current external information. Standard LLMs may hallucinate or provide unsupported answers. This project uses Retrieval-Augmented Generation to ground answers in retrieved evidence.

## How It Works

1. The user uploads documents.
2. The app chunks the documents.
3. Chunks are embedded using a HuggingFace model.
4. FAISS retrieves the most relevant chunks.
5. Tavily retrieves live web results.
6. Groq generates a structured answer.
7. An evidence checker assigns a confidence score.
8. A critic agent reviews the answer.
9. The final report can be downloaded.

## Example Questions

- What are the main data integrity risks in pharmaceutical reporting?
- How can dashboards improve audit readiness?
- What are the limitations of RAG in regulated environments?
- How could Power BI support deviation monitoring?

## What Makes This Different

This project is not a generic Perplexity clone. It is focused on pharmaceutical and data intelligence use cases, with emphasis on:

- data integrity
- regulated reporting
- audit readiness
- analytics governance
- dashboard validation
- evidence-based answers

## Evaluation

The app tracks:

- number of local chunks retrieved
- number of web results retrieved
- response time
- confidence level
- retrieval quality label

The notebook also includes manual evaluation and chunk-size comparison.

## Limitations

- The quality of answers depends on uploaded documents.
- Web search results may vary over time.
- The confidence score is a simple heuristic.
- The system does not replace expert regulatory review.
- LLM-generated answers should be checked before use in regulated decisions.

## Future Improvements

- Add ChromaDB comparison
- Add reranking
- Add user authentication
- Add persistent vector database
- Add PDF export
- Add stronger source validation
- Add automated evaluation scoring
- Deploy on Streamlit Community Cloud

## Final Reflection

This project demonstrates how RAG can reduce hallucinations by grounding LLM answers in local and external evidence. It also shows how agentic workflows can improve answer quality by separating research, evidence checking, writing, and critique.

The project is relevant to real-world pharmaceutical and analytics environments where teams need reliable, traceable, and current information to support decision-making.