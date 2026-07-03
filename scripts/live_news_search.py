#!/usr/bin/env python3
"""Semantic search over live Finnhub news.

Set FINNHUB_API_KEY in your environment or in .env (see .env.example).

Usage:
    python scripts/live_news_search.py                      # general market news, interactive
    python scripts/live_news_search.py --ticker TSLA
    python scripts/live_news_search.py --ticker AAPL --query "earnings guidance"
"""

from __future__ import annotations

import argparse
import os
import sys

from finsearch.config import get_config, load_env_file, resolve_device
from finsearch.models.encoders import BaseEncoder, FineTunedSBERTEncoder, SBERTEncoder
from finsearch.retrieval.live_news import fetch_company_news, fetch_general_news, search_news


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live news semantic search (Finnhub)")
    parser.add_argument(
        "--api_key", type=str, default=None, help="Finnhub API key (default: $FINNHUB_API_KEY)"
    )
    parser.add_argument("--ticker", type=str, default=None, help="Ticker for company news, e.g. AAPL")
    parser.add_argument("--query", type=str, default=None, help="Run one query and exit")
    parser.add_argument("--days", type=int, default=7, help="Days of company news to fetch")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(get_config().paths.default_checkpoint("sbert")),
        help="Fine-tuned SBERT checkpoint (falls back to base SBERT)",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Results per query")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    return parser.parse_args()


def load_encoder(model_path: str, device: str) -> BaseEncoder:
    try:
        encoder = FineTunedSBERTEncoder(checkpoint_path=model_path, device=device).load()
        print("✓ Loaded fine-tuned SBERT")
        return encoder
    except Exception as error:  # noqa: BLE001 - any load failure falls back to the base model
        print(f"Could not load fine-tuned model ({error}); falling back to base SBERT.")
        return SBERTEncoder(device=device).load()


def fetch_articles(args: argparse.Namespace, api_key: str) -> list:
    if args.ticker:
        articles = fetch_company_news(args.ticker, api_key, args.days)
        print(f"✓ Fetched {len(articles)} articles for {args.ticker.upper()}")
    else:
        articles = fetch_general_news(api_key)
        print(f"✓ Fetched {len(articles)} general market articles")
    return articles


def print_results(retrieval_results: list, show_url: bool) -> None:
    print(f"\nTop {len(retrieval_results)} results:")
    for result in retrieval_results:
        print(f"\n  [{result['score']:.4f}] {result['headline']}")
        print(f"     Source: {result['source']} | Date: {result['date']}")
        if show_url and result["url"]:
            print(f"     URL: {result['url']}")


def main() -> None:
    load_env_file()
    args = parse_args()
    api_key = args.api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        sys.exit(
            "Missing Finnhub API key. Get a free key at https://finnhub.io/register, then set "
            "FINNHUB_API_KEY in your environment or .env, or pass --api_key."
        )

    print("\n" + "=" * 60 + "\nLive News Semantic Search\n" + "=" * 60)
    encoder = load_encoder(args.model_path, resolve_device(args.no_cuda))

    print("\nFetching live news from Finnhub...")
    articles = fetch_articles(args, api_key)
    if not articles:
        sys.exit("No articles returned. Check the API key or try a different ticker.")

    if args.query:
        print(f"\nQuery: '{args.query}'")
        print_results(search_news(args.query, articles, encoder, args.top_k), show_url=True)
        return

    print("\n" + "-" * 60 + "\nType a query; 'refresh' re-fetches news; 'quit' exits.\n" + "-" * 60)
    while True:
        try:
            query_text = input("\nQuery> ").strip()
        except EOFError:
            break
        if query_text.lower() in ("quit", "exit"):
            break
        if query_text.lower() == "refresh":
            print("Refreshing news...")
            articles = fetch_articles(args, api_key)
            continue
        if query_text:
            print_results(search_news(query_text, articles, encoder, args.top_k), show_url=False)


if __name__ == "__main__":
    main()
