import time
import streamlit as st
import pandas as pd

@st.cache_resource
def build_cached_vectorstore(folders_to_use):
    all_chunks = []

    for folder in folders_to_use:
        chunks = load_and_split_documents(folder)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None

    return create_vectorstore(all_chunks)

from src.utils import save_uploaded_files
from src.document_loader import load_and_split_documents
from src.vectorstore import create_vectorstore, similarity_search
from src.web_search import tavily_search
from src.rag_pipeline import generate_answer
from src.agents import (
    researcher_agent,
    evidence_checker_agent,
    critic_agent,
    final_writer_agent
)
from src.evaluation import calculate_basic_metrics, retrieval_quality_label
from src.reporting import create_markdown_report, save_report


st.set_page_config(
    page_title="Pharma & Data Intelligence Assistant",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Pharma & Data Intelligence Assistant")
st.write(
    "An agentic RAG research assistant for pharmaceutical, regulatory, "
    "data quality, and analytics intelligence."
)

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "last_report" not in st.session_state:
    st.session_state.last_report = None


with st.sidebar:
    st.header("Settings")

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, MD or DOCX files",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True
    )

    use_sample_docs = st.checkbox("Use sample documents", value=True)
    use_uploaded_docs = st.checkbox("Use uploaded documents", value=True)
    use_web = st.checkbox("Use live Tavily web search", value=True)
    use_critic = st.checkbox("Use critic agent", value=True)

    answer_style = st.selectbox(
        "Answer style",
        [
            "Detailed report",
            "Executive summary",
            "Bullet point summary",
            "Interview-style explanation",
            "Technical analysis"
        ]
    )

    k_value = st.slider("Number of local chunks to retrieve", 1, 10, 5)
    max_web_results = st.slider("Number of web results", 1, 10, 5)


st.subheader("Ask a research question")

query = st.text_area(
    "Example: What are the main data integrity risks in pharmaceutical reporting dashboards?",
    height=100
)

run_button = st.button("Run Intelligence Workflow")


if run_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        start_time = time.time()

        with st.spinner("Running research workflow..."):
            folders_to_use = []

            if use_sample_docs:
                folders_to_use.append("data/sample_docs")

            if uploaded_files and use_uploaded_docs:
                uploaded_folder = save_uploaded_files(uploaded_files)
                folders_to_use.append(uploaded_folder)

            local_docs = []

            try:
                vectorstore = build_cached_vectorstore(tuple(folders_to_use))

                if vectorstore:
                    local_docs = similarity_search(vectorstore, query, k=k_value)

            except Exception as error:
                st.error(f"Error building vectorstore: {error}")

            web_results = []

            if use_web:
                web_results = tavily_search(query, max_results=max_web_results)

            research_summary = researcher_agent(query, local_docs, web_results)
            confidence_report = evidence_checker_agent(local_docs, web_results)

            initial_answer = generate_answer(
                query=query,
                local_docs=local_docs,
                web_results=web_results,
                answer_style=answer_style
            )

            answer = initial_answer["answer"]

            if use_critic:
                critique = critic_agent(answer)
                final_answer = final_writer_agent(answer, confidence_report, critique)
            else:
                critique = "Critic agent was not used."
                final_answer = answer

            end_time = time.time()

            metrics = calculate_basic_metrics(
                query=query,
                local_docs=local_docs,
                web_results=web_results,
                start_time=start_time,
                end_time=end_time
            )

            metrics["retrieval_quality"] = retrieval_quality_label(local_docs, web_results)
            metrics["confidence"] = confidence_report["confidence"]

            st.session_state.metrics.append(metrics)

            report = create_markdown_report(
                query=query,
                answer=final_answer,
                confidence=confidence_report,
                web_sources=web_results,
                local_sources=local_docs
            )

            report_path = save_report(report)
            st.session_state.last_report = report

        st.success("Research workflow complete.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Local chunks", len(local_docs))
        col2.metric("Web sources", len(web_results))
        col3.metric("Confidence", confidence_report["confidence"])
        col4.metric("Time", f"{metrics['response_time_seconds']}s")

        st.subheader("Final Answer")
        st.markdown(final_answer)

        st.subheader("Confidence Assessment")
        st.info(confidence_report["confidence_explanation"])

        with st.expander("Research Agent Summary"):
            st.json({
                "query": research_summary["query"],
                "local_evidence_count": research_summary["local_evidence_count"],
                "web_evidence_count": research_summary["web_evidence_count"]
            })

        with st.expander("Critic Agent Review"):
            st.write(critique)

        with st.expander("Local Sources Retrieved"):
            for i, doc in enumerate(local_docs, start=1):
                st.markdown(f"### Local Source {i}")
                st.write(doc.metadata)
                st.write(doc.page_content[:1000])

        with st.expander("Web Sources Retrieved"):
            for i, result in enumerate(web_results, start=1):
                st.markdown(f"### Web Source {i}: {result.get('title', 'Untitled')}")
                st.write(result.get("url", "No URL"))
                st.write(result.get("content", ""))

        st.download_button(
            label="Download Markdown Report",
            data=report,
            file_name="pharma_data_intelligence_report.md",
            mime="text/markdown"
        )


st.divider()

st.subheader("Evaluation Dashboard")

if st.session_state.metrics:
    metrics_df = pd.DataFrame(st.session_state.metrics)
    st.dataframe(metrics_df)

    st.bar_chart(metrics_df[["local_chunks_retrieved", "web_results_retrieved"]])
else:
    st.write("No evaluation data yet. Run a query to generate metrics.")


st.divider()

st.subheader("Suggested Example Questions")

st.markdown("""
- What are the main data integrity risks in pharmaceutical reporting?
- How can Power BI dashboards support GMP deviation monitoring?
- What are the benefits of RAG for regulated knowledge management?
- How should a pharmaceutical company validate analytics dashboards?
- Compare local document evidence with live web evidence on AI governance.
""")