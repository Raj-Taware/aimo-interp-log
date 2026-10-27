"""CPU tests for scripts/dump_states.py (T-019) with a mock tokenizer/model; no downloads."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import dump_states as ds  # noqa: E402
from selfperturb.features import build_prompt  # noqa: E402


class FakeTokenizer:
    chat_template = None  # build_prompt falls back to plain text

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=2048):
        ids = torch.tensor([[(len(w) % 7) + 1 for w in text.split()] or [1]])
        ids = ids[:, :max_length]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, n_layers, d):
        self.n_layers, self.d = n_layers, d

    def __call__(self, input_ids, attention_mask, output_hidden_states, return_dict, use_cache):
        assert output_hidden_states and not use_cache and input_ids.shape[0] == 1
        T = input_ids.shape[1]
        base = input_ids.float().unsqueeze(-1)  # (1, T, 1)
        states = tuple((base * (layer + 1) + torch.arange(self.d).float()).half() for layer in range(self.n_layers + 1))
        return SimpleNamespace(hidden_states=states)


@pytest.mark.parametrize("n_layers,expected", [
    (36, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]),
    (30, [0, 4, 8, 12, 16, 20, 24, 28, 30]),  # last layer appended when not on the grid
    (4, [0, 4]),
])
def test_state_layers(n_layers, expected):
    assert ds.state_layers(n_layers) == expected


def test_extract_states_shapes_and_values():
    model, tok = FakeModel(n_layers=10, d=6), FakeTokenizer()
    mean, last, n_tokens, layers = ds.extract_states(tok, model, "a bb ccc dddd")
    assert layers == [0, 4, 8, 10]
    assert mean.shape == last.shape == (4, 6)
    assert mean.dtype == last.dtype == np.float16
    prompt = build_prompt(tok, "a bb ccc dddd")
    assert n_tokens == len(tok(prompt)["input_ids"][0])
    full = model(**tok(prompt), output_hidden_states=True, return_dict=True, use_cache=False).hidden_states
    for row, layer in enumerate(layers):
        state = full[layer][0].float()
        np.testing.assert_allclose(mean[row].astype(np.float32), state.mean(0).half().float().numpy(), atol=1e-3)
        np.testing.assert_array_equal(last[row], state[-1].half().numpy())


def test_run_writes_spec_keys_and_resumes(tmp_path):
    rows = [(f"p{i} x y", bool(i % 2)) for i in range(5)]
    tok, model = FakeTokenizer(), FakeModel(8, 4)
    path = ds.run(tok, model, rows[:3], tmp_path, "toy/model", flush_every=2)  # first 3 rows only
    z = np.load(path, allow_pickle=True)
    assert len(z["problems"]) == 3
    path = ds.run(tok, model, rows, tmp_path, "toy/model", flush_every=2)  # resumes 3 -> 5
    z = np.load(path, allow_pickle=True)
    assert set(z.files) == {"mean", "last", "layers", "n_tokens", "problems", "labels", "model_id"}
    assert z["mean"].shape == (5, 3, 4) and z["last"].shape == (5, 3, 4)
    assert z["layers"].tolist() == [0, 4, 8]
    assert z["n_tokens"].shape == (5,) and z["labels"].tolist() == [False, True, False, True, False]
    assert list(z["problems"]) == [r[0] for r in rows] and str(z["model_id"]) == "toy/model"
    assert z["mean"].dtype == np.float16


def test_split_by_layer(tmp_path):
    rows = [(f"p{i} q", False) for i in range(4)]
    path = ds.run(FakeTokenizer(), FakeModel(16, 64), rows, tmp_path, "toy/m", flush_every=2)
    full = np.load(path, allow_pickle=True)
    parts = ds.split_by_layer(path, max_bytes=2000)
    assert len(parts) > 1 and not path.exists()
    seen = []
    for part in parts:
        assert part.stat().st_size <= 4000  # one layer pair + small metadata
        z = np.load(part, allow_pickle=True)
        assert set(z.files) == set(full.files)
        seen += z["layers"].tolist()
    assert seen == full["layers"].tolist()
    merged = np.concatenate([np.load(p, allow_pickle=True)["mean"] for p in parts], axis=1)
    np.testing.assert_array_equal(merged, full["mean"])


def test_load_rows_schemas(tmp_path):
    tv2 = tmp_path / "tv2.parquet"
    pd.DataFrame({"model_id": ["a", "a", "b"], "problem": ["p1", "p2", "p3"], "is_robust": [True, False, True]}).to_parquet(tv2)
    assert ds.load_source_rows(str(tv2), "a") == [("p1", True), ("p2", False)]
    pool = tmp_path / "pool.parquet"
    pd.DataFrame({"model_id": ["a", "a", "b"], "original_problem": ["q1", "q2", "q1"],
                  "model_is_robust": [False, False, False]}).to_parquet(pool)
    assert sorted(ds.load_source_rows(str(pool), "a")) == [("q1", False), ("q2", False)]
