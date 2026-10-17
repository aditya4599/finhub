# FinSearch

FinSearch is a semantic search system for financial news. It represents headlines as dense vectors, stores those vectors in a FAISS index, and retrieves articles that are semantically related to a natural-language query—even when the query and headline do not use exactly the same words.

The project supports Sentence-BERT (SBERT) and FinBERT encoders, optional triplet-loss fine-tuning, local headline collections, and live news retrieval through the Finnhub API.

## Features

- **Semantic headline search** using SBERT or FinBERT embeddings.
- **Model fine-tuning** with triplet loss to bring related examples closer in embedding space.
- **FAISS retrieval** with flat and IVF index options.
- **Live news search** using Finnhub general-market or company news.
- **Retrieval evaluation** with Precision@k, Recall@k, Mean Reciprocal Rank (MRR), and nDCG.
- **Configurable runs** through YAML configuration and command-line arguments.
- **Automated tests** for data handling, encoders, training, indexing, retrieval, metrics, and configuration.

## How it works

1. **Prepare data** — load the financial-news topic dataset or provide a local headline file.
2. **Encode text** — convert headlines into vector embeddings using SBERT or FinBERT.
3. **Fine-tune (optional)** — train an encoder with triplet loss using labelled examples.
4. **Build an index** — save headline vectors and associated document information in FAISS.
5. **Search** — encode a query and retrieve the nearest indexed headlines.
6. **Evaluate** — measure retrieval quality on the held-out test split.

## Technology

- Python
- PyTorch
- Hugging Face Transformers and Datasets
- Sentence-Transformers
- FAISS
- NumPy and pandas
- scikit-learn and SciPy
- PyYAML
- pytest

## Requirements

- Python 3.9 or newer
- Internet access for first-time model and dataset downloads
- Additional disk space for downloaded datasets, model checkpoints, and indexes
- A CPU is sufficient for small demonstrations. Training and indexing larger collections can benefit from a compatible GPU.

The project pins Transformers below version 5 because the current FinBERT model-loading path depends on the Transformers 4.x stack.

## Installation

Run these commands from the repository root (the directory containing `pyproject.toml`):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[plots,env,dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Alternatively, install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Configuration

Default settings are in [`configs/default.yaml`](configs/default.yaml). They define dataset names, model identifiers, cache and output directories, training parameters, and retrieval settings.

To use a custom configuration, copy the default file and pass it to a script:

```bash
cp configs/default.yaml configs/local.yaml
python scripts/build_index.py --config configs/local.yaml --create_sample
```

Command-line arguments take precedence over YAML values. Paths are resolved relative to the project root by default; `FINSEARCH_HOME` can be set to override the root directory.

### Environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

For live news search, add a Finnhub API key to `.env`:

```dotenv
FINNHUB_API_KEY=your_finnhub_api_key
```

Obtain a key from the [Finnhub website](https://finnhub.io/register). Keep `.env` and real API keys out of version control. `HF_TOKEN` is only needed if access to a selected Hugging Face dataset requires authentication.

## Quick start: build an index and search it

### 1. Build a small demonstration index

This command creates synthetic sample headlines and indexes them. The synthetic records are only for checking that the pipeline runs; they are not real news and should not be used to report model quality.

```bash
python scripts/build_index.py --create_sample --model sbert --index_type flat --output indices/demo
```

### 2. Search the index

Use the same encoder family and checkpoint that were used to build the index.

```bash
python scripts/search.py --index_dir indices/demo --query "Federal Reserve interest rate decision" --top_k 5
```

For interactive search, omit `--query`:

```bash
python scripts/search.py --index_dir indices/demo
```

The search script can also create a tiny demonstration index if the requested index directory does not exist.

## Build an index from data

### Use a Hugging Face financial-news corpus

```bash
python scripts/build_index.py \
  --use_hf \
  --model sbert \
  --sample_size 10000 \
  --index_type flat \
  --output indices/sbert_10k \
  --no_cuda
```

### Use your own headline file

CSV, JSON, JSONL, and Parquet inputs are supported by the data loader. The file should contain a headline/text column; optional fields such as date, ticker, and source can be carried through as metadata.

```bash
python scripts/build_index.py \
  --headlines_path path/to/headlines.csv \
  --model sbert \
  --index_type flat \
  --output indices/my_news
```

Available encoder choices are:

- `sbert` — base Sentence-BERT
- `sbert-ft` — fine-tuned Sentence-BERT checkpoint
- `finbert` — base FinBERT
- `finbert-ft` — fine-tuned FinBERT checkpoint

For a fine-tuned encoder, supply `--model_path` if the checkpoint is not at the configured default location. Use `flat` for smaller collections; IVF indexes require a suitable number of vectors for the configured number of clusters.

## Fine-tune the encoders

Training uses the Twitter Financial News Topic dataset configured in `configs/default.yaml`. The scripts download the dataset and required pretrained model on first use.

### Sentence-BERT

```bash
python scripts/train_sbert.py --subset 4000 --epochs 3 --no_cuda
```

Run multiple random seeds:

```bash
python scripts/train_sbert.py --seeds 42 123 456 --subset 4000 --epochs 3
```

For a short pipeline check:

```bash
python scripts/train_sbert.py --quick --no_cuda
```

### FinBERT

```bash
python scripts/train_finbert.py --subset 4000 --epochs 3 --no_cuda
```

For a short pipeline check:

```bash
python scripts/train_finbert.py --quick --no_cuda
```

Training outputs are written under the configured model directory (by default `saved_models/`). Checkpoints and training logs can require substantial disk space. The `--quick` option is intended for smoke testing, not for producing final performance claims.

## Search live news

Live search retrieves news from Finnhub and ranks the returned articles against a semantic query. Set `FINNHUB_API_KEY` in `.env` first.

Search general market news interactively:

```bash
python scripts/live_news_search.py
```

Search company news:

```bash
python scripts/live_news_search.py --ticker TSLA
```

Run a single query against a company's recent news:

```bash
python scripts/live_news_search.py \
  --ticker AAPL \
  --query "earnings guidance" \
  --days 7 \
  --top_k 5
```

The API may return no articles for some tickers or time ranges. Availability and limits depend on the Finnhub account and API plan.

## Evaluate retrieval quality

Evaluate the available base and fine-tuned encoders on the held-out portion of the configured financial-news topic dataset:

```bash
python scripts/evaluate_all.py --num_queries 200 --seeds 42 --no_cuda
```

For a multi-seed run:

```bash
python scripts/evaluate_all.py \
  --num_queries 200 \
  --seeds 42 123 456 789 1024 \
  --no_cuda
```

The script evaluates whichever fine-tuned checkpoints are available alongside the base models and writes a JSON results file under the configured results directory unless `--output` is specified. Multi-seed evaluation can take a while, especially on CPU.

Metrics include:

- **Precision@k (P@k):** fraction of the top-*k* results considered relevant.
- **Recall@k (R@k):** fraction of relevant items retrieved in the top-*k* results.
- **Mean Reciprocal Rank (MRR):** reciprocal rank of the first relevant result, averaged over queries.
- **Normalized Discounted Cumulative Gain (nDCG@k):** ranking quality that gives more credit to relevant results near the top.

The evaluation labels are topic-based. A topic match is a useful proxy, but it is not necessarily the same as relevance to every real-world financial search query.

## Plot results

If the optional plotting dependencies are installed, generate visual summaries from the available evaluation results:

```bash
python scripts/plot_results.py
```

Check the script's help output if you need to specify custom input or output paths:

```bash
python scripts/plot_results.py --help
```

## Tests

Run the test suite from the repository root:

```bash
pytest
```

To skip tests marked as requiring model downloads:

```bash
pytest -m "not slow"
```

Some tests may need network access or cached Hugging Face assets, depending on which test is being run.

## Project structure

```text
FinSearch/
├── configs/
│   └── default.yaml          # Default model, data, training, and search settings
├── finsearch/
│   ├── config/               # Configuration and path handling
│   ├── data/                 # Dataset loading and triplet construction
│   ├── evaluation/           # Retrieval metrics and statistical analysis
│   ├── indexing/             # Vector-index support
│   ├── models/               # SBERT and FinBERT encoders
│   ├── retrieval/             # Search engine and live-news retrieval
│   └── training/              # Training objectives, callbacks, and trainer
├── scripts/
│   ├── build_index.py        # Build and save a FAISS index
│   ├── search.py             # Search a saved index
│   ├── live_news_search.py   # Search current Finnhub news
│   ├── train_sbert.py        # Fine-tune SBERT
│   ├── train_finbert.py      # Fine-tune FinBERT
│   ├── evaluate_all.py       # Compare retrieval models
│   └── plot_results.py       # Plot evaluation summaries
├── tests/                    # Automated tests
├── results/                  # Evaluation outputs and figures
├── requirements.txt
└── pyproject.toml
```

Generated datasets, model checkpoints, and FAISS indexes are stored in the configured output directories and may not be included in a fresh checkout.

## Limitations and responsible interpretation

- Semantic similarity is not a guarantee of financial correctness, factual accuracy, or investment usefulness.
- Retrieval quality depends on the encoder, corpus coverage, preprocessing, and index configuration.
- Topic labels used during evaluation do not perfectly represent user intent or relevance in production.
- The live-news workflow depends on the external Finnhub API, its coverage, and rate limits.
- Training, indexing, and evaluation results should be reproduced in the target environment before being treated as verified results.
- FinSearch is a research and information-retrieval project, not investment advice.

## License and attribution

Review `NOTICE.md` and the project's documentation before redistributing or publishing this repository. The project metadata does not declare a license; do not assume that the code is released under an open-source license unless one is explicitly added by the rights holder.
