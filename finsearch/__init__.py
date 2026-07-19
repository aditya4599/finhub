"""FinSearch: semantic search over financial news with triplet-loss fine-tuned encoders.

Subpackages:
    config      settings dataclasses and YAML loading
    data        labelled dataset, headline ingestion, triplet generation
    models      SBERT / FinBERT encoders
    training    triplet objectives, callbacks, training loop
    indexing    FAISS vector index
    retrieval   search engine and live (Finnhub) news search
    evaluation  IR metrics, evaluator, multi-seed statistics
"""

__version__ = "0.1.0"
