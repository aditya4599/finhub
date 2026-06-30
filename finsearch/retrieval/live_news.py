"""Semantic search over recent news fetched from the Finnhub API.

Requires a Finnhub API key (free tier: https://finnhub.io/register), read from
the ``FINNHUB_API_KEY`` environment variable by the CLI script.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import requests

from finsearch.indexing.faiss_index import FaissVectorIndex
from finsearch.models.encoders import BaseEncoder

FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_MARKET_NEWS_URL = "https://finnhub.io/api/v1/news"
REQUEST_TIMEOUT_SECONDS = 15

Article = Dict[str, Any]


def _get_json(url: str, params: Dict[str, Any]) -> List[Article]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        print(f"Error fetching news from Finnhub: {error}")
        return []


def fetch_company_news(ticker: str, api_key: str, days_back: int = 7) -> List[Article]:
    """Fetch the last ``days_back`` days of news for one ticker."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return _get_json(
        FINNHUB_COMPANY_NEWS_URL,
        {
            "symbol": ticker.upper(),
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "token": api_key,
        },
    )


def fetch_general_news(api_key: str, category: str = "general") -> List[Article]:
    """Fetch the latest market-wide news for a Finnhub news category."""
    return _get_json(FINNHUB_MARKET_NEWS_URL, {"category": category, "token": api_key})


def _format_timestamp(article: Article) -> str:
    timestamp = article.get("datetime")
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M") if timestamp else "Unknown"


def search_news(
    query_text: str, articles: List[Article], encoder: BaseEncoder, k: int = 5
) -> List[Dict[str, Any]]:
    """Rank Finnhub ``articles`` by cosine similarity between ``query_text`` and each headline.

    Articles with an empty headline are skipped. Returns dicts with
    ``headline``, ``source``, ``url``, ``date`` and ``score``.
    """
    articles_with_headline = [article for article in articles if article.get("headline")]
    if not articles_with_headline:
        return []

    headline_texts = [article["headline"] for article in articles_with_headline]
    document_embeddings = encoder.encode(headline_texts, normalize=True)
    query_embeddings = encoder.encode([query_text], normalize=True)

    index = FaissVectorIndex(dimension=document_embeddings.shape[1], index_type="flat", metric="cosine")
    index.add(document_embeddings)
    scores, indices = index.search(query_embeddings, min(k, len(headline_texts)))

    retrieval_results = []
    for score, article_index in zip(scores[0], indices[0]):
        if article_index < 0:
            continue
        article = articles_with_headline[article_index]
        retrieval_results.append(
            {
                "headline": article.get("headline", ""),
                "source": article.get("source", "Unknown"),
                "url": article.get("url", ""),
                "date": _format_timestamp(article),
                "score": float(score),
            }
        )
    return retrieval_results
