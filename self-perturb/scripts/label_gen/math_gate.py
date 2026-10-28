"""T-020 gate: do our self-made DeepSeek labels agree with the organisers' on the gate split?

Rule A (organisers', build_extra_labels.py): robust iff max_drop <= 0.
Rule B (common.py): robust iff max_drop <= 0.10, spurious iff >= 0.25, else dropped.
Pre-registered pass (written before data): rule A on complete gate rows (see below), agreement >= 0.70
AND Cohen's kappa >= 0.30. Also writes <out>/labels_new.parquet (agg schema, rule A,
complete rows of the "new" split) for dump_features.py / dump_states.py.

    python scripts/label_gen/math_gate.py --problems <problems.jsonl> --variants <run>/stage2_variants.jsonl \
        --labels <run>/labels.jsonl --out <run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import read_jsonl

DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def agreement(official, mine) -> dict:
    if not len(official):
        return {"n": 0}
    same = sum(a == b for a, b in zip(official, mine))
    kappa = cohen_kappa_score(official, mine) if len(set(official) | set(mine)) > 1 else None
    return {"n": len(official), "agreement": round(same / len(official), 3),
            "kappa": None if kappa is None else round(float(kappa), 3),
            "official_robust": int(sum(official)), "mine_robust": int(sum(mine))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = {p["problem_id"]: p for p in read_jsonl(args.problems)}
    rows = pd.DataFrame([r for r in read_jsonl(args.labels) if r["model_id"] == DEEPSEEK])
    rows["split"] = rows["problem_id"].map(lambda i: meta[i]["split"])
    rows["official"] = rows["problem_id"].map(lambda i: meta[i]["official_label"])
    # complete = every variant written for the problem was solved (rename leaves some texts unchanged)
    written = pd.Series([v["problem_id"] for v in read_jsonl(args.variants)]).value_counts()
    ok = rows[(rows["n_types"] == rows["problem_id"].map(written)) & rows["max_drop"].notna()].copy()
    ok["rule_a"] = ok["max_drop"] <= 1e-9
    gate = ok[ok["split"] == "gate"]
    b = gate[gate["is_robust"].notna()]
    report = {"stage1_kept": int(len(rows)), "complete": int(len(ok)),
              "rule_a": agreement(list(gate["official"].astype(bool)), list(gate["rule_a"])),
              "rule_b": agreement(list(b["official"].astype(bool)), list(b["is_robust"].astype(bool)))}
    ra = report["rule_a"]
    report["pass"] = bool(ra.get("n") and ra["agreement"] >= 0.70 and (ra["kappa"] or 0) >= 0.30)
    new = ok[ok["split"] == "new"]
    report["new_labelled"], report["new_robust"] = int(len(new)), int(new["rule_a"].sum())
    pd.DataFrame({"model_id": DEEPSEEK, "original_problem": new["problem"], "model_is_robust": new["rule_a"],
                  "max_drop": new["max_drop"], "base_acc": new["base_acc"], "problem_id": new["problem_id"]}
                 ).to_parquet(args.out / "labels_new.parquet", index=False)
    (args.out / "gate_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
