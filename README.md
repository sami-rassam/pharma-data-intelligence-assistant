# Pharma & Data Intelligence Assistant

## Live Demo

🔗 https://sami-rassam-pharma-data-intelligence-assistant-app-r0rsch.streamlit.app/

## Portfolio Summary

The Pharma & Data Intelligence Assistant is an end-to-end Agentic Retrieval-Augmented Generation (RAG) application designed to support pharmaceutical, regulatory, data quality, and analytics research.

The application combines local document retrieval, live web search, LLM-powered reasoning, source reliability assessment, confidence scoring, report generation, and evaluation metrics to provide evidence-based answers for complex pharmaceutical and data-driven questions.

This project demonstrates practical skills in Python development, AI engineering, data analytics, vector databases, information retrieval, prompt engineering, and application deployment.

---

## Setup Instructions

Clone the repository:

```bash
git clone https://github.com/sami-rassam/pharma-data-intelligence-assistant.git
cd pharma-data-intelligence-assistant
```

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your API keys:

```text
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Run the app locally:

```bash
streamlit run app.py
```

---

## Docker Deployment

Build the image:

```bash
docker build -t pharma-data-intelligence-assistant .
```

Run the container:

```bash
docker run -p 8501:8501 --env-file .env pharma-data-intelligence-assistant
```

Open:

```text
http://localhost:8501
```

---

## Key Features

### Knowledge Retrieval

- Upload PDF, TXT, MD and DOCX documents
- Local document retrieval using FAISS vector search
- HuggingFace sentence-transformer embeddings
- Retrieval-Augmented Generation (RAG) architecture
- Persistent vector storage for improved performance

### AI Research Workflow

- Groq LLM integration
- Tavily live web search
- Research Agent
- Evidence Checker Agent
- Critic Agent
- Final Writer Agent
- Multi-step research workflow visibility

### Transparency & Reliability

- Confidence scoring
- Source transparency
- Source reliability scoring
- Retrieval quality metrics
- Automated answer quality scoring
- Evidence-based answer generation

### Reporting & User Experience

- Markdown report export
- PDF report export
- Persistent chat history
- Example prompt templates
- Interactive evaluation dashboard
- Streamlit web interface

---

## Tech Stack

### AI & Retrieval

- LangChain
- Groq
- Tavily
- FAISS
- HuggingFace Sentence Transformers

### Application Development

- Python
- Streamlit
- Pandas

### Deployment

- Streamlit Community Cloud
- Docker

---

## Problem Statement

Pharmaceutical, quality, regulatory, and analytics teams frequently need to answer complex questions using a combination of internal documentation and external information sources.

Traditional LLMs can generate hallucinated or unsupported answers. This project addresses that challenge by grounding responses in retrieved evidence from uploaded documents and trusted web sources, improving transparency and reducing misinformation risk.

---

## System Architecture

1. User submits a research question.
2. Relevant document chunks are retrieved from FAISS.
3. Tavily retrieves live external information.
4. Source reliability scores are assigned.
5. Research Agent gathers evidence.
6. Evidence Checker Agent assesses confidence.
7. Groq generates an initial answer.
8. Critic Agent reviews the response.
9. Final Writer Agent produces the final answer.
10. Reports, metrics, and evaluation scores are generated.

---

## Example Use Cases

### Pharmaceutical Quality

- Data integrity assessments
- SOP reviews
- GMP compliance support
- Deviation investigations
- CAPA effectiveness reviews

### Data Analytics

- Dashboard KPI recommendations
- Data governance reviews
- Reporting quality assessments
- Root cause investigations
- Audit readiness evaluations

### Research & Intelligence

- Regulatory intelligence
- Industry trend analysis
- Technology landscape reviews
- Best practice comparisons

---

## What Makes This Different

This project is not a generic Perplexity clone.

It is specifically designed around pharmaceutical and analytics use cases, with emphasis on:

- Data integrity
- Audit readiness
- Regulatory compliance
- Evidence-based reporting
- Source transparency
- AI-assisted decision support
- Quality management principles

The project reflects many of the challenges encountered in regulated environments where traceability, documentation quality, and evidence-based decision making are essential.

---

## Evaluation Framework

The application automatically tracks:

- Local chunks retrieved
- Web results retrieved
- Response time
- Confidence score
- Retrieval quality score
- Source reliability score
- Automated answer quality score
- Answer length metrics

The accompanying notebook includes:

- Manual evaluation framework
- Retrieval testing
- Chunk-size comparison
- RAG performance analysis

---

## Screenshots

### Application Homepage

![Homepage](screenshots/homepage.png)

### Example Research Workflow

![Research Workflow](screenshots/research-workflow.png)

### Source Reliability Scoring

![Source Reliability](screenshots/source-reliability.png)

### Evaluation Dashboard

![Evaluation Dashboard](screenshots/evaluation-dashboard.png)

---

## Skills Demonstrated

### Data Analytics

- Data quality assessment
- Root cause analysis
- KPI development
- Dashboard thinking
- Data governance principles
- Reporting automation

### Artificial Intelligence

- Retrieval-Augmented Generation (RAG)
- Prompt engineering
- LLM orchestration
- Vector databases
- Semantic search
- Agent-based workflows

### Software Engineering

- Python application development
- Streamlit deployment
- Docker containerisation
- API integration
- Modular architecture
- Git version control

### Pharmaceutical Domain Knowledge

- GMP principles
- Data integrity
- Audit readiness
- Regulatory intelligence
- SOP review
- Compliance-focused analysis

---

## Limitations

- Retrieval quality depends on uploaded document quality.
- Source reliability scoring uses rule-based heuristics.
- Confidence scores are indicative rather than formal validation.
- Responses should not replace expert regulatory review.
- LLM-generated outputs should always be reviewed before use in regulated decision-making environments.

---

## Completed Enhancements

- Persistent chat history
- Source reliability scoring
- PDF report export
- Persistent FAISS vector storage
- Automated evaluation scoring
- Docker support
- Streamlit Cloud deployment

---

## Future Improvements

- Retrieval reranking
- User authentication
- ChromaDB comparison
- Database-backed analytics
- Persistent chat history across sessions
- Enhanced source verification
- Multi-user support
- Advanced evaluation framework

---

## Final Reflection

This project demonstrates how Agentic RAG systems can improve answer quality by combining document retrieval, live web search, confidence assessment, source transparency, and iterative AI review.

The project mirrors real-world pharmaceutical and analytics workflows where decisions must be evidence-based, traceable, and supported by reliable information sources.

Through this project I strengthened skills in Python, AI engineering, information retrieval, data analytics, application deployment, prompt engineering, and user-focused product development while building a practical solution for regulated, data-driven environments.
