"""Loading unlabelled news headlines for indexing.

Two sources are supported:
* a local CSV / JSON / JSONL / Parquet file (:class:`NewsHeadlinesLoader`);
* the Hugging Face dataset configured as ``news_corpus_dataset``
  (default ``Brianferrell787/financial-news-multisource``) via
  :func:`load_hf_financial_news`. Set ``HF_TOKEN`` if the dataset requires it.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd

from finsearch.config import get_config

# Characters kept from each article body when the HF corpus has no separate headline.
MAX_HF_TEXT_CHARS = 500


@dataclass
class NewsHeadline:
    """A single news headline record."""

    headline: str
    date: Optional[datetime] = None
    ticker: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def load_hf_financial_news(n_samples: Optional[int] = None, streaming: bool = True) -> pd.DataFrame:
    """Load records from the configured Hugging Face news corpus.

    Args:
        n_samples: Number of records to read from the start of the stream
            (``None`` reads everything, which is tens of millions of rows).
        streaming: Stream rows instead of downloading the full dataset.

    Returns:
        With ``streaming=True``: a DataFrame with ``headline`` (first
        ``MAX_HF_TEXT_CHARS`` characters of ``text``), ``date``, ``source``
        and ``url`` columns. With ``streaming=False``: the raw dataset columns.
    """
    from datasets import load_dataset

    dataset_name = get_config().news_corpus_dataset
    hf_token = os.environ.get("HF_TOKEN")

    if not streaming:
        hub_dataset = load_dataset(dataset_name, split="train", token=hf_token)
        if n_samples:
            hub_dataset = hub_dataset.select(range(min(n_samples, len(hub_dataset))))
        return hub_dataset.to_pandas()

    stream = load_dataset(dataset_name, split="train", streaming=True, token=hf_token)
    records = []
    for row_number, item in enumerate(stream):
        if n_samples and row_number >= n_samples:
            break

        extra_fields: Dict[str, Any] = {}
        if item.get("extra_fields"):
            try:
                extra_fields = json.loads(item["extra_fields"])
            except (TypeError, json.JSONDecodeError):
                pass

        records.append(
            {
                "headline": item["text"][:MAX_HF_TEXT_CHARS] if item.get("text") else "",
                "date": item.get("date"),
                "source": extra_fields.get("publication", extra_fields.get("source", "")),
                "url": extra_fields.get("url", ""),
            }
        )
    return pd.DataFrame(records)


class NewsHeadlinesLoader:
    """Load a local headlines file and auto-detect its column names.

    Args:
        file_path: CSV, JSON, JSONL or Parquet file (default: ``paths.headlines_path``).
        headline_col / date_col / ticker_col / source_col: Explicit column names;
            detected from common names when ``None``.
        date_format: ``strftime`` format passed to ``pd.to_datetime``.
    """

    HEADLINE_COLUMNS = ["headline", "title", "text", "content", "news"]
    DATE_COLUMNS = ["date", "datetime", "timestamp", "published", "pub_date"]
    TICKER_COLUMNS = ["ticker", "symbol", "stock", "company"]
    SOURCE_COLUMNS = ["source", "publisher", "outlet", "provider"]

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        headline_col: Optional[str] = None,
        date_col: Optional[str] = None,
        ticker_col: Optional[str] = None,
        source_col: Optional[str] = None,
        date_format: Optional[str] = None,
    ):
        self.file_path = Path(file_path) if file_path else get_config().paths.headlines_path
        self.headline_col = headline_col
        self.date_col = date_col
        self.ticker_col = ticker_col
        self.source_col = source_col
        self.date_format = date_format
        self._df: Optional[pd.DataFrame] = None

    def _require_path(self) -> Path:
        if self.file_path is None:
            raise ValueError("No headlines file specified. Pass file_path or call set_headlines_path().")
        return self.file_path

    def _detect_file_format(self) -> str:
        suffix = self._require_path().suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix in (".json", ".jsonl"):
            return "json"
        if suffix in (".parquet", ".pq"):
            return "parquet"
        raise ValueError(
            f"Unsupported headlines file format '{suffix}'. Use .csv, .json, .jsonl or .parquet."
        )

    @staticmethod
    def _first_match(candidates: List[str], lowercase_to_original: Dict[str, str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in lowercase_to_original:
                return lowercase_to_original[candidate]
        return None

    def _auto_detect_columns(self, frame: pd.DataFrame) -> None:
        lowercase_to_original = {column.lower(): column for column in frame.columns}

        if self.headline_col is None:
            self.headline_col = self._first_match(self.HEADLINE_COLUMNS, lowercase_to_original)
            if self.headline_col is None:
                raise ValueError(
                    "Could not detect a headline column. Pass headline_col explicitly. "
                    f"Available columns: {frame.columns.tolist()}"
                )
        if self.date_col is None:
            self.date_col = self._first_match(self.DATE_COLUMNS, lowercase_to_original)
        if self.ticker_col is None:
            self.ticker_col = self._first_match(self.TICKER_COLUMNS, lowercase_to_original)
        if self.source_col is None:
            self.source_col = self._first_match(self.SOURCE_COLUMNS, lowercase_to_original)

    def load(
        self,
        n_rows: Optional[int] = None,
        sample_frac: Optional[float] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Read the file into memory, optionally truncating to ``n_rows`` and then
        sampling ``sample_frac`` of the rows. Dates are parsed when a date column exists."""
        file_path = self._require_path()
        file_format = self._detect_file_format()

        if file_format == "csv":
            frame = pd.read_csv(file_path, nrows=n_rows)
        elif file_format == "json" and file_path.suffix == ".jsonl":
            frame = pd.read_json(file_path, lines=True, nrows=n_rows)
        elif file_format == "json":
            frame = pd.read_json(file_path)
            frame = frame.head(n_rows) if n_rows else frame
        else:
            frame = pd.read_parquet(file_path)
            frame = frame.head(n_rows) if n_rows else frame

        if sample_frac is not None and sample_frac < 1.0:
            frame = frame.sample(frac=sample_frac, random_state=seed)

        self._auto_detect_columns(frame)

        if self.date_col and self.date_col in frame.columns:
            try:
                frame[self.date_col] = pd.to_datetime(
                    frame[self.date_col], format=self.date_format, errors="coerce"
                )
            except (ValueError, TypeError):
                pass  # keep the raw values if they cannot be parsed

        self._df = frame
        return frame

    def stream(self, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Yield the file in ``chunk_size``-row DataFrames. CSV and JSONL are read
        incrementally; JSON and Parquet are loaded once and then chunked."""
        file_path = self._require_path()
        file_format = self._detect_file_format()

        if file_format == "csv":
            chunks: Iterator[pd.DataFrame] = pd.read_csv(file_path, chunksize=chunk_size)
        elif file_format == "json" and file_path.suffix == ".jsonl":
            chunks = pd.read_json(file_path, lines=True, chunksize=chunk_size)
        else:
            frame = pd.read_json(file_path) if file_format == "json" else pd.read_parquet(file_path)
            chunks = (frame.iloc[i : i + chunk_size] for i in range(0, len(frame), chunk_size))

        for chunk in chunks:
            if self.headline_col is None:
                self._auto_detect_columns(chunk)
            yield chunk

    def get_headlines(
        self,
        start_date: Optional[Union[str, date, datetime]] = None,
        end_date: Optional[Union[str, date, datetime]] = None,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[NewsHeadline]:
        """Return headlines filtered by date range, ticker and/or source."""
        if self._df is None:
            self.load()
        frame = self._df.copy()

        if start_date and self.date_col:
            frame = frame[frame[self.date_col] >= pd.to_datetime(start_date)]
        if end_date and self.date_col:
            frame = frame[frame[self.date_col] <= pd.to_datetime(end_date)]
        if ticker and self.ticker_col:
            frame = frame[frame[self.ticker_col].str.upper() == ticker.upper()]
        if source and self.source_col:
            frame = frame[frame[self.source_col].str.lower() == source.lower()]
        if max_results:
            frame = frame.head(max_results)

        return [
            NewsHeadline(
                headline=row[self.headline_col],
                date=row.get(self.date_col) if self.date_col else None,
                ticker=row.get(self.ticker_col) if self.ticker_col else None,
                source=row.get(self.source_col) if self.source_col else None,
            )
            for _, row in frame.iterrows()
        ]

    def get_headline_texts(self) -> List[str]:
        """Return all headline strings."""
        if self._df is None:
            self.load()
        return self._df[self.headline_col].tolist()

    @property
    def is_loaded(self) -> bool:
        return self._df is not None

    @property
    def num_headlines(self) -> int:
        return len(self._df) if self._df is not None else 0


def create_sample_headlines_file(output_path: Path, n_samples: int = 1000) -> Path:
    """Write a CSV of synthetic, template-generated headlines for local smoke testing.

    The rows are random and NOT real news; never use them for evaluation.
    """
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA", "JPM", "BAC", "WFC"]
    sources = ["Reuters", "Bloomberg", "CNBC", "WSJ", "MarketWatch", "Yahoo Finance"]
    templates = [
        "{ticker} shares rise {pct}% on strong earnings",
        "{ticker} announces quarterly dividend of ${amt}",
        "{ticker} CEO comments on market conditions",
        "Analyst upgrades {ticker} to Buy with ${target} target",
        "{ticker} reports {pct}% revenue growth in Q{q}",
        "Fed signals potential rate {direction} in coming months",
        "{ticker} acquires startup for ${amt}M in all-cash deal",
        "{ticker} faces regulatory scrutiny over {issue}",
        "Breaking: {ticker} {direction}s {pct}% in pre-market trading",
        "{ticker} launches new product line, stock reacts positively",
    ]

    base_date = datetime(2020, 1, 1)
    records = []
    for _ in range(n_samples):
        template = random.choice(templates)
        ticker = random.choice(tickers)
        headline = template.format(
            ticker=ticker,
            pct=random.randint(1, 20),
            amt=random.randint(1, 100),
            target=random.randint(100, 500),
            q=random.randint(1, 4),
            direction=random.choice(["rise", "fall", "jump", "drop"]),
            issue=random.choice(["privacy", "antitrust", "data handling"]),
        )
        records.append(
            {
                "headline": headline,
                "date": (base_date + pd.Timedelta(days=random.randint(0, 1500))).isoformat(),
                "ticker": ticker,
                "source": random.choice(sources),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)
    return output_path
