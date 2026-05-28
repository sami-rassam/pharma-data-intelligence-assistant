from datetime import datetime
from pathlib import Path


def create_markdown_report(
    query: str,
    answer: str,
    confidence: dict,
    web_sources: list,
    local_sources: list
) -> str:
    """
    Creates a Markdown report.
    """
    date_created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Pharma & Data Intelligence Report

**Date created:** {date_created}

## User Question

{query}

## Final Answer

{answer}

## Confidence Assessment

- **Confidence:** {confidence.get("confidence")}
- **Explanation:** {confidence.get("confidence_explanation")}
- **Local sources used:** {confidence.get("local_sources_used")}
- **Web sources used:** {confidence.get("web_sources_used")}

## Local Sources

"""

    for i, doc in enumerate(local_sources, start=1):
        source = doc.metadata.get("source", "Unknown source")
        preview = doc.page_content[:300].replace("\n", " ")
        report += f"### Local Source {i}: {source}\n\n{preview}...\n\n"

    report += "## Web Sources\n\n"

    for i, source in enumerate(web_sources, start=1):
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        report += f"{i}. [{title}]({url})\n"

    report += """
## Limitations

This report depends on the quality of uploaded documents, retrieval settings, and live web search results. It should be reviewed by a subject matter expert before being used for regulated, legal, clinical, or compliance decisions.
"""

    return report


def save_report(report: str, folder: str = "reports") -> str:
    """
    Saves report to reports folder.
    """
    Path(folder).mkdir(exist_ok=True)

    filename = f"pharma_data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path = Path(folder) / filename

    with open(path, "w", encoding="utf-8") as file:
        file.write(report)

    return str(path)