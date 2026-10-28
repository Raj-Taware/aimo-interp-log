"""CPU tests for scripts/label_gen/{make_kernel_t020,t020_phases}.py (T-020): no GPU, no downloads."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "label_gen"))
import make_kernel_t020 as mk  # noqa: E402
import t020_phases as ph  # noqa: E402


def test_generated_kernel_compiles_and_has_inputs(tmp_path):
    d = mk.build("gate", "aimo-t020-gate", tmp_path / "g")
    src = (d / "run.py").read_text()
    compile(src, "run.py", "exec")
    meta = json.loads((d / "kernel-metadata.json").read_text())
    assert meta["id"] == "rajtaware10/aimo-t020-gate" and meta["enable_gpu"] and meta["kernel_sources"] == []
    assert "results/grok/T-020/problems_gate.jsonl" in src and "t020_phases.py" in src
    for secret in ("kaggle.json", "KAGGLE_KEY", "CODABENCH_PASSWORD", "ghp_"):
        assert secret not in src


def test_halves_and_resume(tmp_path):
    d = mk.build("new", "aimo-t020-new1", tmp_path / "n", half=1, halves=2, resume_from="rajtaware10/old")
    assert json.loads((d / "kernel-metadata.json").read_text())["kernel_sources"] == ["rajtaware10/old"]
    assert "'--half', '1', '--halves', '2'" in (d / "run.py").read_text()
    with pytest.raises(SystemExit):
        mk.build("gate", "x", tmp_path / "x", halves=2)
    with pytest.raises(SystemExit):
        mk.build("new", "x", tmp_path / "x", half=2, halves=2)


def test_strided_halves_partition_problems():
    probs = [json.loads(l) for l in (ROOT / "results/grok/T-020/problems_new.jsonl").read_text().splitlines()]
    h0, h1 = probs[0::2], probs[1::2]
    assert len(h0) + len(h1) == len(probs) == 161
    assert {p["problem_id"] for p in h0}.isdisjoint(p["problem_id"] for p in h1)


def test_token_stats(tmp_path):
    recs = [{"tokens": t, "finish_reason": f, "batch_seconds": s, "finished_at": "2026-10-28T10:00:00+0000"}
            for t, f, s in [(100, "stop", 10.0), (300, "stop", 10.0), (6144, "length", 10.0), (500, "stop", 20.0)]]
    (tmp_path / "stage1_generations.jsonl").write_text("\n".join(map(json.dumps, recs)))
    s = ph.token_stats(tmp_path)["stage1"]
    assert s["median_tokens"] == 400 and s["n_hit_cap"] == 1 and s["n_chunks"] == 2
    assert s["tok_per_s"] == round((100 + 300 + 6144 + 500) / 30, 1)
    assert json.loads((tmp_path / "token_stats.json").read_text())["stage1"]["n"] == 4


def test_math_gate_on_inputs_runs(tmp_path):
    """math_gate.py on perfect fake labels for the committed gate split: agreement 1.0, pass true."""
    pytest.importorskip("sklearn")
    probs = [json.loads(l) for l in (ROOT / "results/grok/T-020/problems.jsonl").read_text().splitlines()]
    vars_ = [json.loads(l) for l in (ROOT / "results/grok/T-020/stage2_variants.jsonl").read_text().splitlines()]
    written = {}
    for v in vars_:
        written[v["problem_id"]] = written.get(v["problem_id"], 0) + 1
    rows = []
    for p in probs:
        if p["split"] != "gate" or p["problem_id"] not in written:
            continue
        off = bool(p["official_label"])
        rows.append({"model_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "problem": p["problem"],
                     "problem_id": p["problem_id"], "base_acc": 1.0, "n_types": written[p["problem_id"]],
                     "max_drop": 0.0 if off else 0.5, "is_robust": True if off else False})
    (tmp_path / "labels.jsonl").write_text("\n".join(map(json.dumps, rows)))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/label_gen/math_gate.py"), "--problems",
                        str(ROOT / "results/grok/T-020/problems.jsonl"), "--variants",
                        str(ROOT / "results/grok/T-020/stage2_variants.jsonl"), "--labels",
                        str(tmp_path / "labels.jsonl"), "--out", str(tmp_path)], capture_output=True, text=True,
                       cwd=ROOT / "scripts/label_gen")
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "gate_report.json").read_text())
    assert rep["rule_a"]["agreement"] == 1.0 and rep["pass"] is True
