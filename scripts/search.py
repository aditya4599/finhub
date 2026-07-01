#!/usr/bin/env python3
"""Interactive search over a saved index.

Usage:
    python scripts/search.py --index_dir indices/sbert_ft_10k
    python scripts/search.py --query "Fed rate hike" --top_k 5

If the index directory does not exist, a 10-headline demo index is built with
base SBERT so the pipeline can be tried without any training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from finsearch.config import get_config, load_config, resolve_device
from finsearch.models.encoders import BaseEncoder, FineTunedSBERTEncoder, SBERTEncoder
from finsearch.retrieval.search import SAMPLE_HEADLINES, FinancialSearchEngine, SearchResult


def parse_args() -> argparse.Namespace:
    paths = get_config().paths
    parser = argparse.ArgumentParser(description="Search a saved FinSearch index")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument(
        "--index_dir", type=str, default=str(paths.index_dir / "default"), help="Saved index directory"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(paths.default_checkpoint("sbert")),
        help="Fine-tuned SBERT checkpoint",
    )
    parser.add_argument("--query", type=str, default=None, help="Run one query and exit")
    parser.add_argument("--top_k", type=int, default=5, help="Results per query")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    return parser.parse_args()


def load_query_encoder(model_path: str, device: str) -> BaseEncoder:
    """Load the fine-tuned SBERT checkpoint, falling back to base SBERT."""
    if Path(model_path).exists():
        print(f"Loading fine-tuned SBERT from {model_path}...")
        return FineTunedSBERTEncoder(checkpoint_path=model_path, device=device).load()
    print(f"No checkpoint at {model_path}; using base SBERT.")
    return SBERTEncoder(device=device).load()


def print_results(retrieval_results: list[SearchResult]) -> None:
    print(f"\nTop {len(retrieval_results)} results:")
    for result in retrieval_results:
        print(f"  {result.rank}. [{result.score:.4f}] {result.headline}")
        if result.date:
            print(f"     Date: {result.date} | Ticker: {result.ticker}")


def main() -> None:
    args = parse_args()
    if args.config:
        load_config(args.config)

    print("\n" + "=" * 60 + "\nFinSearch\n" + "=" * 60)
    device = resolve_device(args.no_cuda)
    index_dir = Path(args.index_dir)

    # The query encoder must match the encoder used to build the index.
    encoder = load_query_encoder(args.model_path, device)
    engine = FinancialSearchEngine(encoder)

    if index_dir.exists():
        print(f"Loading index from {index_dir}...")
        engine.load(str(index_dir))
        print(f"Loaded {len(engine.documents)} documents.")
    else:
        print(f"No index at {index_dir}; building a {len(SAMPLE_HEADLINES)}-headline demo index.")
        engine.build_index(SAMPLE_HEADLINES, index_type="flat", show_progress=False)

    if args.query:
        print_results(engine.search(args.query, k=args.top_k))
        return

    print("\n" + "-" * 60 + "\nType a query (e.g. 'Tesla stock price'); 'quit' to exit.\n" + "-" * 60)
    while True:
        try:
            query_text = input("\nQuery> ").strip()
        except EOFError:
            break
        if query_text.lower() in ("quit", "exit"):
            break
        if query_text:
            print_results(engine.search(query_text, k=args.top_k))


if __name__ == "__main__":
    main()
