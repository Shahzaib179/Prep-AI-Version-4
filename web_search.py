from __future__ import annotations

from ddgs import DDGS

TRUSTED_HINTS = (".gov", ".edu", "who.int", "nih.gov", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "ac.uk")


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    rows = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for r in results:
            url = r.get("href", "")
            rows.append({"title": r.get("title", ""), "url": url, "snippet": r.get("body", ""), "trusted": str(any(h in url.lower() for h in TRUSTED_HINTS))})
    rows.sort(key=lambda x: x["trusted"] == "True", reverse=True)
    return rows
