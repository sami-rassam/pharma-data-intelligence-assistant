from urllib.parse import urlparse


def score_source_reliability(url: str) -> dict:
    """
    Scores web source reliability using simple domain-based rules.
    """
    if not url:
        return {
            "score": 1,
            "label": "Unknown",
            "reason": "No URL available"
        }

    domain = urlparse(url).netloc.lower()

    high_trust_domains = [
        ".gov",
        ".edu",
        "who.int",
        "fda.gov",
        "ema.europa.eu",
        "mhra.gov.uk",
        "nice.org.uk",
        "nhs.uk",
        "iso.org"
    ]

    medium_trust_domains = [
        "nature.com",
        "sciencedirect.com",
        "springer.com",
        "pubmed.ncbi.nlm.nih.gov",
        "bmj.com",
        "pharmaceutical-journal.com"
    ]

    if any(trusted in domain for trusted in high_trust_domains):
        return {
            "score": 5,
            "label": "High",
            "reason": "Recognised government, academic, regulatory, healthcare, or standards source"
        }

    if any(trusted in domain for trusted in medium_trust_domains):
        return {
            "score": 4,
            "label": "Medium-High",
            "reason": "Recognised scientific, healthcare, or professional publication source"
        }

    if domain:
        return {
            "score": 3,
            "label": "Medium",
            "reason": "General web source; check carefully before relying on it"
        }

    return {
        "score": 1,
        "label": "Unknown",
        "reason": "Unable to assess source reliability"
    }


def add_reliability_scores(web_results: list) -> list:
    """
    Adds reliability scoring to each Tavily web result.
    """
    scored_results = []

    for result in web_results:
        url = result.get("url", "")
        reliability = score_source_reliability(url)
        result["reliability_score"] = reliability["score"]
        result["reliability_label"] = reliability["label"]
        result["reliability_reason"] = reliability["reason"]
        scored_results.append(result)

    return scored_results