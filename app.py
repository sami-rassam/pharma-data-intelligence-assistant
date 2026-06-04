import time
import streamlit as st
import pandas as pd

from src.utils import save_uploaded_files
from src.document_loader import load_and_split_documents
from src.vectorstore import (
    create_vectorstore,
    get_or_create_vectorstore,
    similarity_search
)
from src.web_search import tavily_search
from src.rag_pipeline import generate_answer
from src.agents import (
    researcher_agent,
    evidence_checker_agent,
    critic_agent,
    final_writer_agent
)
from src.evaluation import (
    calculate_basic_metrics,
    retrieval_quality_label,
    calculate_automated_scores
)
from src.reporting import (
    create_markdown_report,
    save_report,
    save_pdf_report
)
from src.source_reliability import add_reliability_scores


st.set_page_config(
    page_title="Pharma & Data Intelligence Assistant",
    page_icon="💊",
    layout="wide"
)


# -----------------------------
# Cached Vectorstore Builders
# -----------------------------

@st.cache_resource
def build_sample_vectorstore():
    chunks = load_and_split_documents("data/sample_docs")

    if not chunks:
        return None

    return get_or_create_vectorstore(chunks)


def build_uploaded_vectorstore(uploaded_files):
    uploaded_folder = save_uploaded_files(uploaded_files)
    chunks = load_and_split_documents(uploaded_folder)

    if not chunks:
        return None

    return create_vectorstore(chunks)


# -----------------------------
# Session State
# -----------------------------

if "metrics" not in st.session_state:
    st.session_state.metrics = []

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    use_web = st.checkbox(
        "Use live Tavily web search",
        value=False,
        help="Turn this on when you need current external information."
    )

    use_critic = st.checkbox(
        "Use critic agent",
        value=False,
        help="Adds an extra LLM review step, but may slow the app down."
    )

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

    k_value = st.slider("Number of local chunks to retrieve", 1, 10, 3)
    max_web_results = st.slider("Number of web results", 1, 10, 3)

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
# Chat History
# -----------------------------

st.divider()

st.subheader("Chat History")

if st.session_state.chat_history:
    for i, item in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"Question {i}: {item['question']}"):
            st.write(f"Confidence: {item['confidence']}")
            st.write(f"Local sources used: {item['local_sources']}")
            st.write(f"Web sources used: {item['web_sources']}")
            st.write(f"Quality score: {item.get('quality_score', 'N/A')}%")
            st.markdown(item["answer"])
else:
    st.write("No chat history yet.")


# -----------------------------
# Example Prompt Buttons
# -----------------------------

st.divider()

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

st.divider()

st.subheader("Ask a research question")

query = st.text_area(
    "Enter your question below:",
    value=st.session_state.query,
    height=120,
    placeholder="Example: What are the main data integrity risks in pharmaceutical reporting dashboards?"
)

run_button = st.button("Run Intelligence Workflow")


# -----------------------------
# Main Workflow
# -----------------------------

if run_button:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        start_time = time.time()
        workflow_steps = []

        local_docs = []
        web_results = []
        critique = "Critic agent was not used."
        final_answer = ""
        confidence_report = {}

        with st.spinner("Running research workflow..."):

            # -----------------------------
            # Local RAG Retrieval
            # -----------------------------

            vectorstores = []

            if use_sample_docs:
                try:
                    sample_vectorstore = build_sample_vectorstore()

                    if sample_vectorstore:
                        vectorstores.append(sample_vectorstore)
                        workflow_steps.append("✓ Sample document vectorstore loaded")
                    else:
                        workflow_steps.append("⚠ No sample documents available")

                except Exception as error:
                    st.error(f"Error loading sample documents: {error}")
                    workflow_steps.append("✗ Sample document vectorstore failed")

            if uploaded_files and use_uploaded_docs:
                try:
                    uploaded_vectorstore = build_uploaded_vectorstore(uploaded_files)

                    if uploaded_vectorstore:
                        vectorstores.append(uploaded_vectorstore)
                        workflow_steps.append("✓ Uploaded document vectorstore created")
                    else:
                        workflow_steps.append("⚠ No uploaded document chunks available")

                except Exception as error:
                    st.error(f"Error loading uploaded documents: {error}")
                    workflow_steps.append("✗ Uploaded document vectorstore failed")

            for vectorstore in vectorstores:
                try:
                    docs = similarity_search(vectorstore, query, k=k_value)
                    local_docs.extend(docs)
                except Exception as error:
                    st.error(f"Error searching vectorstore: {error}")

            if local_docs:
                workflow_steps.append("✓ Local document retrieval completed")
            else:
                workflow_steps.append("⚠ No local document sources retrieved")

            # -----------------------------
            # Tavily Web Search
            # -----------------------------

            if use_web:
                try:
                    web_results = tavily_search(query, max_results=max_web_results)
                    web_results = add_reliability_scores(web_results)
                    workflow_steps.append("✓ Tavily web search completed")
                    workflow_steps.append("✓ Source reliability scoring completed")

                except Exception as error:
                    st.error(f"Error running Tavily web search: {error}")
                    web_results = []
                    workflow_steps.append("✗ Tavily web search failed")
            else:
                workflow_steps.append("⚠ Tavily web search skipped")

            # -----------------------------
            # Agent Workflow
            # -----------------------------

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
                final_answer = final_writer_agent(
                    answer,
                    confidence_report,
                    critique
                )
                workflow_steps.append("✓ Critic agent reviewed answer")
                workflow_steps.append("✓ Final writer agent produced final response")
            else:
                final_answer = answer
                workflow_steps.append("⚠ Critic agent skipped")

            end_time = time.time()

            # -----------------------------
            # Evaluation Metrics
            # -----------------------------

            metrics = calculate_basic_metrics(
                query=query,
                local_docs=local_docs,
                web_results=web_results,
                start_time=start_time,
                end_time=end_time
            )

            metrics["retrieval_quality"] = retrieval_quality_label(
                local_docs,
                web_results
            )

            metrics["confidence"] = confidence_report["confidence"]
            metrics["answer_words"] = len(final_answer.split())

            automated_scores = calculate_automated_scores(
                local_docs=local_docs,
                web_results=web_results,
                confidence=confidence_report["confidence"],
                answer=final_answer
            )

            metrics.update(automated_scores)
            st.session_state.metrics.append(metrics)

            # -----------------------------
            # Reports
            # -----------------------------

            report = create_markdown_report(
                query=query,
                answer=final_answer,
                confidence=confidence_report,
                web_sources=web_results,
                local_sources=local_docs
            )

            report_path = save_report(report)
            pdf_report_path = save_pdf_report(report)

            st.session_state.last_report = report

            # -----------------------------
            # Chat History Update
            # -----------------------------

            st.session_state.chat_history.append({
                "question": query,
                "answer": final_answer,
                "confidence": confidence_report["confidence"],
                "local_sources": len(local_docs),
                "web_sources": len(web_results),
                "quality_score": automated_scores["overall_answer_quality_score"]
            })

        st.success("Research workflow complete.")

        # -----------------------------
        # KPI Metrics
        # -----------------------------

        st.subheader("Evaluation Metrics")

        col1, col2, col3, col4, col5, col6 = st.columns(6)

        col1.metric("Local Chunks", len(local_docs))
        col2.metric("Web Sources", len(web_results))
        col3.metric("Confidence", confidence_report["confidence"])
        col4.metric("Answer Words", len(final_answer.split()))
        col5.metric("Time", f"{metrics['response_time_seconds']}s")
        col6.metric(
            "Quality Score",
            f"{automated_scores['overall_answer_quality_score']}%"
        )

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

                reliability_label = result.get("reliability_label", "Not scored")
                reliability_score = result.get("reliability_score", "N/A")

                if url:
                    st.markdown(
                        f"🌐 Web Source {i}: [{title}]({url}) "
                        f"— Reliability: {reliability_label} ({reliability_score}/5)"
                    )
                else:
                    st.write(
                        f"🌐 Web Source {i}: {title} "
                        f"— Reliability: {reliability_label} ({reliability_score}/5)"
                    )
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

        with st.expander("Automated Evaluation Scores"):
            st.json(automated_scores)

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
                    st.markdown(
                        f"### Web Source {i}: {result.get('title', 'Untitled')}"
                    )

                    st.write(result.get("url", "No URL"))

                    st.write(
                        f"Reliability: "
                        f"{result.get('reliability_label', 'Not scored')} "
                        f"({result.get('reliability_score', 'N/A')}/5)"
                    )

                    st.caption(
                        result.get(
                            "reliability_reason",
                            "No reliability reason available."
                        )
                    )

                    st.write(result.get("content", ""))
            else:
                st.write("No web sources retrieved.")

        # -----------------------------
        # Download Reports
        # -----------------------------

        st.subheader("Download Reports")

        st.download_button(
            label="Download Markdown Report",
            data=report,
            file_name="pharma_data_intelligence_report.md",
            mime="text/markdown"
        )

        with open(pdf_report_path, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name="pharma_data_intelligence_report.pdf",
                mime="application/pdf"
            )


# -----------------------------
# Evaluation Dashboard
# -----------------------------

st.divider()

st.subheader("Evaluation Dashboard")

if st.session_state.metrics:
    metrics_df = pd.DataFrame(st.session_state.metrics)

    st.dataframe(metrics_df)

    chart_columns = [
        "local_chunks_retrieved",
        "web_results_retrieved",
        "overall_answer_quality_score"
    ]

    available_chart_columns = [
        column for column in chart_columns if column in metrics_df.columns
    ]

    if available_chart_columns:
        st.bar_chart(metrics_df[available_chart_columns])
else:
    st.write("No evaluation data yet. Run a query to generate metrics.")


# -----------------------------
# Project Footer
# -----------------------------

st.divider()

st.caption(
    "Developed by Sami Rassam | Python | Streamlit | LangChain | RAG | Tavily | Groq"
)