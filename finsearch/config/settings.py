"""Central configuration for FinSearch.

All defaults live in the dataclasses below. ``configs/default.yaml`` documents
the same values and can be copied and edited; pass the copy to any script with
``--config path/to/file.yaml`` (or call :func:`load_config` directly).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# finsearch/config/settings.py -> repository root is three levels up.
PROJECT_ROOT = Path(os.environ.get("FINSEARCH_HOME", Path(__file__).resolve().parents[2]))

# Twitter Financial News Topic dataset: label id -> topic name.
TOPIC_LABELS: Dict[int, str] = {
    0: "Analyst Update",
    1: "Fed | Central Banks",
    2: "Company | Product News",
    3: "Treasuries | Corporate Debt",
    4: "Dividend",
    5: "Earnings",
    6: "Energy | Oil",
    7: "Financials",
    8: "Currencies",
    9: "General News | Opinion",
    10: "Gold | Metals | Materials",
    11: "IPO",
    12: "Legal | Regulation",
    13: "M&A | Investments",
    14: "Macro",
    15: "Markets",
    16: "Politics",
    17: "Personnel Change",
    18: "Stock Commentary",
    19: "Stock Movement",
}


def default_device() -> str:
    """Return ``"cuda"`` when a GPU is visible to PyTorch, otherwise ``"cpu"``."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class PathConfig:
    """Filesystem locations. Relative paths are resolved against the project root."""

    project_root: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data_cache")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "saved_models")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "indices")
    results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "results")

    # Optional local headlines file (CSV / JSON / JSONL / Parquet) for indexing.
    headlines_path: Optional[Path] = None

    def __post_init__(self) -> None:
        for directory in (self.data_dir, self.models_dir, self.logs_dir, self.index_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    def default_checkpoint(self, model_family: str, seed: int = 42) -> Path:
        """Location where the training scripts save the best checkpoint for a seed."""
        return Path(self.models_dir) / model_family / f"seed_{seed}" / "best_model"


@dataclass
class ModelConfig:
    """Base encoder identifiers and tokenisation settings."""

    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    finbert_model_name: str = "yiyanghkust/finbert-tone"
    sbert_dim: int = 384
    finbert_dim: int = 768
    max_seq_length: int = 128


@dataclass
class TrainingConfig:
    """Triplet-loss fine-tuning hyperparameters."""

    margin: float = 0.5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 5
    batch_size: int = 32
    eval_batch_size: int = 64
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])
    patience: int = 3
    min_delta: float = 0.001
    device: str = field(default_factory=default_device)
    gradient_accumulation_steps: int = 1


@dataclass
class EvaluationConfig:
    """Retrieval evaluation settings."""

    k_values: List[int] = field(default_factory=lambda: [5, 10, 20])
    confidence_level: float = 0.95
    num_eval_queries: int = 500


@dataclass
class SearchConfig:
    """FAISS index settings."""

    index_type: str = "ivf"  # "ivf" (approximate) or "flat" (exact)
    nlist: int = 1000  # IVF clusters
    nprobe: int = 50  # IVF clusters probed per query
    top_k: int = 10


@dataclass
class Config:
    """Top-level configuration object."""

    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    # Labelled training/evaluation data (Hugging Face Hub id).
    dataset_name: str = "zeroshot/twitter-financial-news-topic"
    # Large unlabelled corpus used only for building a search index.
    news_corpus_dataset: str = "Brianferrell787/financial-news-multisource"
    topic_labels: Dict[int, str] = field(default_factory=lambda: dict(TOPIC_LABELS))


_config = Config()

_SECTIONS = ("paths", "model", "training", "evaluation", "search")
_TOP_LEVEL_KEYS = ("dataset_name", "news_corpus_dataset")


def get_config() -> Config:
    """Return the process-wide configuration instance."""
    return _config


def set_headlines_path(path: Union[str, Path]) -> None:
    """Set the local headlines file used by :class:`NewsHeadlinesLoader`."""
    _config.paths.headlines_path = Path(path)


def _apply_section(section: Any, values: Dict[str, Any], section_name: str) -> None:
    valid = {f.name: f for f in fields(section)}
    for key, value in values.items():
        if key not in valid:
            raise KeyError(f"Unknown setting '{section_name}.{key}'. Valid keys: {sorted(valid)}")
        if section_name == "paths" and value is not None:
            value = Path(value)
            if not value.is_absolute():
                value = _config.paths.project_root / value
        setattr(section, key, value)


def load_config(path: Union[str, Path]) -> Config:
    """Override the global configuration with values from a YAML file.

    Only keys present in the file are changed; everything else keeps its default.
    Unknown keys raise ``KeyError`` so typos are not silently ignored.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as handle:
        overrides = yaml.safe_load(handle) or {}

    for key, value in overrides.items():
        if key in _SECTIONS:
            _apply_section(getattr(_config, key), value or {}, key)
        elif key in _TOP_LEVEL_KEYS:
            setattr(_config, key, value)
        else:
            raise KeyError(
                f"Unknown top-level setting '{key}'. Valid keys: {sorted(_SECTIONS + _TOP_LEVEL_KEYS)}"
            )

    if "paths" in overrides:
        _config.paths.__post_init__()
    return _config


def load_env_file() -> None:
    """Load ``<project root>/.env`` into ``os.environ`` if python-dotenv is installed.

    Existing environment variables are never overwritten.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def resolve_device(no_cuda: bool = False) -> str:
    """Return the device scripts should use, honouring a ``--no_cuda`` flag."""
    return "cpu" if no_cuda else _config.training.device
