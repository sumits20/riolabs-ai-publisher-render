import os
from tavily import TavilyClient
from langchain_core.tools import tool

def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    response = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results
    )

    results = []
    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", "")
        })

    return results


@tool
def research_topic(query: str) -> str:
    """Search the web for recent information and related article ideas."""
    results = tavily_search(query=query, max_results=5)

    if not results:
        return "No research results found."

    return "\n\n".join(
        f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['content']}"
        for r in results
    )
