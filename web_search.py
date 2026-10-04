from __future__ import annotations

from ddgs import DDGS

TRUSTED_HINTS = (
    ".gov", ".edu", "who.int", "nih.gov", "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", "ac.uk",
    "pmdc.pk", "pec.org.pk", "uhs.edu.pk", "numspak.edu.pk",
)


def search_web(query: str, max_results: int = 5, trusted_only: bool = False) -> list[dict[str, str]]:
    """DuckDuckGo search. trusted_only keeps only government/education/regulator domains."""
    rows = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results * (3 if trusted_only else 1))
        for r in results:
            url = r.get("href", "")
            rows.append({"title": r.get("title", ""), "url": url, "snippet": r.get("body", ""), "trusted": str(any(h in url.lower() for h in TRUSTED_HINTS))})
    if trusted_only:
        rows = [r for r in rows if r["trusted"] == "True"]
    rows.sort(key=lambda x: x["trusted"] == "True", reverse=True)
    return rows[:max_results]
