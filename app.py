import time
import streamlit as st
import pandas as pd

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


@st.cache_resource
def build_cached_vectorstore(folders_to_use):
    all_chunks = []

    for folder in folders_to_use:
        chunks = load_and_split_documents(folder)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None

    return create_vectorstore(all_chunks)


if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "query" not in st.session_state:
    st.session_state.query = ""


# -----------------------------
# Sidebar
# -----------------------------

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
        "Simplified non-technical explanation",
        "Interview-style explanation",
        "Technical analysis"
    ]
)

    k_value = st.slider("Number of local chunks to retrieve", 1, 10, 5)
    max_web_results = st.slider("Number of web results", 1, 10, 5)

    st.markdown("---")
    st.info(
        "Developed by Sami Rassam\n\n"
        "Aspiring Data Analyst | AI & Data Science Portfolio Project"
    )


# -----------------------------
# Main App Header
# -----------------------------

st.title("💊 Pharma & Data Intelligence Assistant")

st.write(
    "An agentic RAG research assistant for pharmaceutical, regulatory, "
    "data quality, and analytics intelligence."
)


# -----------------------------
# Example Prompt Buttons
# -----------------------------

st.subheader("Suggested Example Questions")

col1, col2 = st.columns(2)

if col1.button("🧪 Data Integrity Assessment"):
    st.session_state.query = (
        "What are the main data integrity risks in pharmaceutical reporting dashboards?"
    )

if col2.button("📋 SOP Gap Analysis"):
    st.session_state.query = (
        "Review this SOP and identify potential GMP, compliance, or data governance gaps."
    )

col3, col4 = st.columns(2)

if col3.button("⚠️ Root Cause Investigation"):
    st.session_state.query = (
        "What possible root causes could explain recurring data quality issues in a GMP environment?"
    )

if col4.button("📊 Dashboard KPI Review"):
    st.session_state.query = (
        "What KPIs should be monitored in a pharmaceutical quality dashboard?"
    )


# -----------------------------
# Query Input
# -----------------------------

st.subheader("Ask a research question")

query = st.text_area(
    "Enter your question below:",
    value=st.session_state.query,
    height=120,
    placeholder="Example: What are the main data integrity risks in pharmaceutical reporting dashboards?"
)

run_button = st.button("Run Intelligence Workflow")


# -----------------------------
# Workflow
# -----------------------------

if run_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        start_time = time.time()
        workflow_steps = []

        with st.spinner("Running research workflow..."):
            folders_to_use = []

            if use_sample_docs:
                folders_to_use.append("data/sample_docs")
                workflow_steps.append("✓ Sample documents selected")

            if uploaded_files and use_uploaded_docs:
                uploaded_folder = save_uploaded_files(uploaded_files)
                folders_to_use.append(uploaded_folder)
                workflow_steps.append("✓ Uploaded documents saved")

            local_docs = []

            try:
                vectorstore = build_cached_vectorstore(tuple(folders_to_use))

                if vectorstore:
                    local_docs = similarity_search(vectorstore, query, k=k_value)
                    workflow_steps.append("✓ Local document retrieval completed")
                else:
                    workflow_steps.append("⚠ No local documents available")

            except Exception as error:
                st.error(f"Error building vectorstore: {error}")
                workflow_steps.append("✗ Vectorstore build failed")

            web_results = []

            if use_web:
                web_results = tavily_search(query, max_results=max_web_results)
                workflow_steps.append("✓ Tavily web search completed")
            else:
                workflow_steps.append("⚠ Tavily web search skipped")

            research_summary = researcher_agent(query, local_docs, web_results)
            workflow_steps.append("✓ Research agent completed")

            confidence_report = evidence_checker_agent(local_docs, web_results)
            workflow_steps.append("✓ Evidence checker completed")

            initial_answer = generate_answer(
                query=query,
                local_docs=local_docs,
                web_results=web_results,
                answer_style=answer_style
            )

            answer = initial_answer["answer"]
            workflow_steps.append("✓ LLM answer generated")

            if use_critic:
                critique = critic_agent(answer)
                final_answer = final_writer_agent(answer, confidence_report, critique)
                workflow_steps.append("✓ Critic agent reviewed answer")
                workflow_steps.append("✓ Final writer agent produced final response")
            else:
                critique = "Critic agent was not used."
                final_answer = answer
                workflow_steps.append("⚠ Critic agent skipped")

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
            metrics["answer_words"] = len(final_answer.split())

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

        # -----------------------------
        # KPI Metrics
        # -----------------------------

        st.subheader("Evaluation Metrics")

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Local Chunks", len(local_docs))
        col2.metric("Web Sources", len(web_results))
        col3.metric("Confidence", confidence_report["confidence"])
        col4.metric("Answer Words", len(final_answer.split()))
        col5.metric("Time", f"{metrics['response_time_seconds']}s")

        metrics_chart_df = pd.DataFrame({
            "Source Type": ["Local Documents", "Web Results"],
            "Count": [len(local_docs), len(web_results)]
        })

        st.bar_chart(metrics_chart_df.set_index("Source Type"))

        # -----------------------------
        # Final Answer
        # -----------------------------

        st.subheader("Final Answer")
        st.markdown(final_answer)

        # -----------------------------
        # Workflow Visibility
        # -----------------------------

        st.subheader("Workflow Executed")

        for step in workflow_steps:
            st.write(step)

        # -----------------------------
        # Confidence Assessment
        # -----------------------------

        st.subheader("Confidence Assessment")
        st.info(confidence_report["confidence_explanation"])

        # -----------------------------
        # Source Transparency
        # -----------------------------

        st.subheader("Sources Used")

        if local_docs:
            st.markdown("### Local Document Sources")
            for i, doc in enumerate(local_docs, start=1):
                source = doc.metadata.get("source", "Unknown source")
                st.write(f"📄 Local Source {i}: {source}")
        else:
            st.write("No local document sources used.")

        if web_results:
            st.markdown("### Web Sources")
            for i, result in enumerate(web_results, start=1):
                title = result.get("title", "Untitled")
                url = result.get("url", "")
                if url:
                    st.markdown(f"🌐 Web Source {i}: [{title}]({url})")
                else:
                    st.write(f"🌐 Web Source {i}: {title}")
        else:
            st.write("No web sources used.")

        # -----------------------------
        # Expanders
        # -----------------------------

        with st.expander("Research Agent Summary"):
            st.json({
                "query": research_summary["query"],
                "local_evidence_count": research_summary["local_evidence_count"],
                "web_evidence_count": research_summary["web_evidence_count"]
            })

        with st.expander("Critic Agent Review"):
            st.write(critique)

        with st.expander("Local Sources Retrieved"):
            if local_docs:
                for i, doc in enumerate(local_docs, start=1):
                    st.markdown(f"### Local Source {i}")
                    st.write(doc.metadata)
                    st.write(doc.page_content[:1000])
            else:
                st.write("No local sources retrieved.")

        with st.expander("Web Sources Retrieved"):
            if web_results:
                for i, result in enumerate(web_results, start=1):
                    st.markdown(f"### Web Source {i}: {result.get('title', 'Untitled')}")
                    st.write(result.get("url", "No URL"))
                    st.write(result.get("content", ""))
            else:
                st.write("No web sources retrieved.")

        # -----------------------------
        # Download Report
        # -----------------------------

        st.download_button(
            label="Download Markdown Report",
            data=report,
            file_name="pharma_data_intelligence_report.md",
            mime="text/markdown"
        )


# -----------------------------
# Evaluation Dashboard
# -----------------------------

st.divider()

st.subheader("Evaluation Dashboard")

if st.session_state.metrics:
    metrics_df = pd.DataFrame(st.session_state.metrics)

    st.dataframe(metrics_df)

    if {
        "local_chunks_retrieved",
        "web_results_retrieved"
    }.issubset(metrics_df.columns):
        st.bar_chart(metrics_df[["local_chunks_retrieved", "web_results_retrieved"]])
else:
    st.write("No evaluation data yet. Run a query to generate metrics.")


# -----------------------------
# Project Footer
# -----------------------------

st.divider()

st.caption(
    "Developed by Sami Rassam | Python | Streamlit | LangChain | RAG | Tavily | Groq"
)