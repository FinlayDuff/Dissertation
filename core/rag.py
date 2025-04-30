from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from utils.langchain.llm_model_selector import retry_on_api_exceptions

EMB = SentenceTransformer("all-MiniLM-L6-v2")

REPORTING_LEMMA = {
    "say",
    "tell",
    "report",
    "claim",
    "announce",
    "according",
    "according_to",
}
BANNED_DOMAINS = [
    "factcheck.org",
    "snopes.com",
    "politifact.com",
    "kaggle.com",
]


@retry_on_api_exceptions()
def retrieve_from_news(queries, k_per_query=10):
    from duckduckgo_search import DDGS

    hits = []
    with DDGS() as ddgs:
        for q in queries:
            hits.extend(ddgs.news(q, max_results=k_per_query))
    return hits


@retry_on_api_exceptions()
def retrieve_from_search(queries: str | List[str], recall_n: int = 5) -> List[dict]:
    if isinstance(queries, str):
        queries = [queries]
    raw_hits = []
    with DDGS() as ddgs:
        for q in queries:
            hits = list(ddgs.text(q, max_results=recall_n))
            if hits:
                raw_hits.extend(hits)
            # stop as soon as we have a handful to work with
            if len(raw_hits) >= recall_n:
                break
    return raw_hits


def remove_exact_matching_results(title, start, results, similarity_threshold=0.95):
    title_emb = EMB.encode(
        title + start, normalize_embeddings=True, show_progress_bar=False
    )
    candidates = []
    for result in results:
        emb = EMB.encode(
            result.get("title", "") + result.get("body", "")[:200],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sim = float(np.dot(title_emb, emb))
        if sim > similarity_threshold:
            continue
        candidates.append(
            {
                "href": result["href"] if "href" in result else result["url"],
                "title": result["title"],
                "body": result["body"],
                "source": result.get("source"),
            }
        )
    return candidates


def remove_credibility_sources(
    results: List[dict], banned_domains: List[str] = BANNED_DOMAINS
) -> List[dict]:
    """
    Remove sources whose URLs contain any of the banned domains.
    """
    filtered_results = []
    for result in results:
        key = "href" if "href" in result else "url"
        if not any(domain in result[key] for domain in banned_domains):
            filtered_results.append(result)
    return filtered_results
