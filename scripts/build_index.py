#!/usr/bin/env python3
"""Encode a headline collection and save a FAISS search index.

Usage:
    # Hugging Face news corpus (first 10k streamed records), fine-tuned SBERT, exact index
    python scripts/build_index.py --use_hf --model sbert-ft --sample_size 10000 \
        --index_type flat --output indices/sbert_ft_10k --no_cuda

    # Local file
    python scripts/build_index.py --headlines_path my_headlines.csv --model sbert

    # Synthetic headlines (smoke test only; not real news)
    python scripts/build_index.py --create_sample --model sbert --index_type flat
"""

from __future__ import annotations

import argparse
from pathlib import Path

from finsearch.config import get_config, load_config, load_env_file, resolve_device
from finsearch.data.ingestion import NewsHeadlinesLoader, create_sample_headlines_file, load_hf_financial_news
from finsearch.models.encoders import (
    BaseEncoder,
    FinBERTEncoder,
    FineTunedFinBERTEncoder,
    FineTunedSBERTEncoder,
    SBERTEncoder,
)
from finsearch.retrieval.search import FinancialSearchEngine

MODEL_CHOICES = ("sbert", "sbert-ft", "finbert", "finbert-ft")
SAMPLE_SEED = 42
TEST_QUERIES = [
    "Apple quarterly earnings",
    "Federal Reserve interest rate decision",
    "Tesla stock price movement",
    "Bitcoin cryptocurrency market",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS search index")
    parser.add_argument("--config", type=str, default=None, help="YAML config file")
    parser.add_argument(
        "--headlines_path", type=str, default=None, help="CSV / JSON / JSONL / Parquet headlines file"
    )
    parser.add_argument(
        "--use_hf", action="store_true", help="Stream the Hugging Face news corpus instead of a file"
    )
    parser.add_argument("--create_sample", action="store_true", help="Generate and index synthetic headlines")
    parser.add_argument("--model", type=str, default="sbert", choices=MODEL_CHOICES, help="Encoder")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Fine-tuned checkpoint (for *-ft models)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: <index_dir>/default)"
    )
    parser.add_argument(
        "--index_type", type=str, default="ivf", choices=["flat", "ivf"], help="FAISS index type"
    )
    parser.add_argument("--sample_size", type=int, default=None, help="Number of headlines to index")
    parser.add_argument("--batch_size", type=int, default=64, help="Encoding batch size")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU")
    return parser.parse_args()


def load_encoder(model_choice: str, model_path: str | None, device: str) -> BaseEncoder:
    """Instantiate and load the requested encoder."""
    paths = get_config().paths
    if model_choice == "sbert":
        print("Loading base Sentence-BERT...")
        encoder: BaseEncoder = SBERTEncoder(device=device)
    elif model_choice == "sbert-ft":
        checkpoint = model_path or str(paths.default_checkpoint("sbert"))
        print(f"Loading fine-tuned Sentence-BERT from {checkpoint}...")
        encoder = FineTunedSBERTEncoder(checkpoint_path=checkpoint, device=device)
    elif model_choice == "finbert":
        print("Loading base FinBERT...")
        encoder = FinBERTEncoder(device=device)
    else:
        checkpoint = model_path or str(paths.default_checkpoint("finbert"))
        print(f"Loading fine-tuned FinBERT from {checkpoint}...")
        encoder = FineTunedFinBERTEncoder(checkpoint_path=checkpoint, device=device)
    return encoder.load()


def load_headlines(args: argparse.Namespace):
    """Return a DataFrame with at least a ``headline`` column."""
    if args.use_hf:
        print("\nStreaming headlines from the Hugging Face news corpus...")
        return load_hf_financial_news(n_samples=args.sample_size)

    headlines_path = args.headlines_path
    if args.create_sample or headlines_path is None:
        if headlines_path is None and not args.create_sample:
            print("\nNo --headlines_path given: generating SYNTHETIC sample headlines (not real news).")
        headlines_path = create_sample_headlines_file(
            get_config().paths.data_dir / "sample_headlines.csv", n_samples=5000
        )

    print(f"\nLoading headlines from: {headlines_path}")
    loader = NewsHeadlinesLoader(headlines_path)
    headlines_df = loader.load()
    if loader.headline_col != "headline":
        headlines_df = headlines_df.rename(columns={loader.headline_col: "headline"})
    if args.sample_size and args.sample_size < len(headlines_df):
        headlines_df = headlines_df.sample(n=args.sample_size, random_state=SAMPLE_SEED)
    return headlines_df


def main() -> None:
    args = parse_args()
    if args.config:
        load_config(args.config)
    load_env_file()

    print("=" * 60 + "\nBuild Search Index\n" + "=" * 60)
    device = resolve_device(args.no_cuda)
    print(f"\nDevice: {device}")

    encoder = load_encoder(args.model, args.model_path, device)
    headlines_df = load_headlines(args)
    print(f"Loaded {len(headlines_df)} headlines")

    def optional_column(name: str):
        return headlines_df[name].tolist() if name in headlines_df.columns else None

    sources = optional_column("source")
    engine = FinancialSearchEngine(encoder)
    print(f"Building {args.index_type.upper()} index...")
    engine.build_index(
        headlines=headlines_df["headline"].tolist(),
        dates=optional_column("date"),
        tickers=optional_column("ticker"),
        metadata=[{"source": source} for source in sources] if sources is not None else None,
        batch_size=args.batch_size,
        index_type=args.index_type,
        show_progress=True,
    )

    output_dir = Path(args.output) if args.output else get_config().paths.index_dir / "default"
    print(f"\nSaving index to: {output_dir}")
    engine.save(str(output_dir))

    print("\n" + "=" * 60 + "\nTest queries\n" + "=" * 60)
    for query_text in TEST_QUERIES:
        print(f"\nQuery: '{query_text}'")
        for result in engine.search(query_text, k=3):
            print(f"  {result.rank}. [{result.score:.4f}] {result.headline[:70]}...")

    print(f"\nIndex saved to: {output_dir}")


if __name__ == "__main__":
    main()
