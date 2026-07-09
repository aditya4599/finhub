"""Tests for configuration loading and evaluation statistics."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from finsearch.config import settings
from finsearch.evaluation.analysis import StatisticalAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestConfig:
    def test_default_yaml_documents_config_defaults(self):
        """Values documented in default.yaml should match Config defaults."""
        yaml = pytest.importorskip("yaml")
        config_path = PROJECT_ROOT / "configs" / "default.yaml"
        documented = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        defaults = settings.Config()

        for section_name in ("model", "training", "evaluation", "search"):
            section_values = documented[section_name]
            default_section = getattr(defaults, section_name)
            for key, value in section_values.items():
                # The runtime may choose a device dynamically, so the YAML value
                # is not expected to match the machine-specific default.
                if section_name == "training" and key == "device":
                    continue
                assert getattr(default_section, key) == value, f"{section_name}.{key}"

        assert documented["dataset_name"] == defaults.dataset_name
        assert documented["news_corpus_dataset"] == defaults.news_corpus_dataset

    def test_load_config_applies_overrides_and_rejects_unknown_keys(self, tmp_path, monkeypatch):
        """Overrides should be applied while misspelled keys fail loudly."""
        monkeypatch.setattr(settings, "_config", settings.Config())
        valid_path = tmp_path / "valid.yaml"
        valid_path.write_text(
            "training:\n  margin: 0.3\nsearch:\n  nprobe: 8\n",
            encoding="utf-8",
        )

        config = settings.load_config(valid_path)
        assert config.training.margin == pytest.approx(0.3)
        assert config.search.nprobe == 8
        assert config.training.learning_rate == pytest.approx(2e-5)

        invalid_path = tmp_path / "invalid.yaml"
        invalid_path.write_text("training:\n  learnign_rate: 1.0\n", encoding="utf-8")
        with pytest.raises(KeyError, match="learnign_rate"):
            settings.load_config(invalid_path)


class TestStatisticalAnalyzer:
    def test_confidence_interval_matches_scipy_t_interval(self):
        """The reported interval should match a standard two-sided t interval."""
        values = np.array([0.40, 0.42, 0.41, 0.43, 0.39])
        result = StatisticalAnalyzer().compute_ci(values)
        expected_low, expected_high = stats.t.interval(
            0.95,
            df=len(values) - 1,
            loc=np.mean(values),
            scale=stats.sem(values),
        )

        assert result.mean == pytest.approx(np.mean(values))
        assert result.ci_lower == pytest.approx(expected_low)
        assert result.ci_upper == pytest.approx(expected_high)
        assert result.n_samples == len(values)

    def test_recorded_summary_matches_per_seed_results(self):
        """Summary means and lower confidence bounds should match the stored runs."""
        results_path = PROJECT_ROOT / "results" / "inherited" / "evaluation_results.json"
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        recomputed = StatisticalAnalyzer().analyze_multi_seed_results(payload["results"])

        for model_name, metric_summaries in payload["analysis"].items():
            assert model_name in recomputed
            for metric_name, stored_summary in metric_summaries.items():
                calculated = recomputed[model_name][metric_name]
                assert calculated.mean == pytest.approx(stored_summary["mean"])
                assert calculated.ci_lower == pytest.approx(stored_summary["ci_lower"])
