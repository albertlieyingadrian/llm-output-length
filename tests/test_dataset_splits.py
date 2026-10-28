"""
Unit tests for train/eval split selection in HuggingFaceDatasetBuilder.
"""

from pathlib import Path

import pytest
from datasets import Dataset
from omegaconf import OmegaConf

import miol.data.huggingface_dataset as hf_module
from miol.data.huggingface_dataset import HuggingFaceDatasetBuilder, _selected_rows

CONF_DIR = Path(hf_module.__file__).resolve().parents[1] / "conf" / "data"
MMLU_PRO_TEST_ROWS = 12032


def _overlap(a: range, b: range) -> int:
    return max(0, min(a.stop, b.stop) - max(a.start, b.start))


class TestSelectedRows:
    def test_all(self):
        assert _selected_rows(-1, 10) == range(0, 10)

    def test_first_n(self):
        assert _selected_rows(3, 10) == range(0, 3)

    def test_last_n(self):
        assert _selected_rows(-4, 10) == range(6, 10)


class TestSameSplitConfigs:
    def test_mmlu_pro_train_and_eval_are_disjoint(self):
        cfg = OmegaConf.load(CONF_DIR / "mmlu_pro.yaml")
        assert cfg.hf_train_split == cfg.hf_eval_split
        train_rows = _selected_rows(cfg.num_train_samples, MMLU_PRO_TEST_ROWS)
        eval_rows = _selected_rows(cfg.num_eval_samples, MMLU_PRO_TEST_ROWS)
        assert _overlap(train_rows, eval_rows) == 0
        assert len(eval_rows) == 2256

    def test_overlapping_slices_raise(self, monkeypatch):
        rows = Dataset.from_dict({"question": [str(i) for i in range(MMLU_PRO_TEST_ROWS)]})
        monkeypatch.setattr(hf_module, "load_dataset", lambda *args, **kwargs: rows)
        builder = HuggingFaceDatasetBuilder(
            num_train_samples=10000,
            num_eval_samples=-2256,
            hf_dataset="TIGER-Lab/MMLU-Pro",
            hf_prompt_col="question",
            hf_system_prompt="system",
            hf_eval_split="test",
            hf_train_split="test",
        )
        with pytest.raises(ValueError, match="share 224 rows"):
            builder.get_dataset("train", tokenizer=object())
