"""T-014: H100 hours for the full run, extrapolated from the dry-run token counts.

    python scripts/label_gen/estimate_h100.py --run <dryrun out dir> --out results/grok/T-014/h100_estimate.json

Full run: 173 problems x 4 models; stage 1 = 4 samples, 12288 cap; stage 2 =
gpt-oss-120b, 5 types per problem kept by any model; stage 3 = 2 samples per
variant per kept pair, 12288 cap. Dry run: Skywork only, 4096 cap.

Tokens per solve generation: dry-run generations that stopped keep their length;
capped ones (4096) are bounded between 4096 (low) and 12288 (high), with
``mid`` = 8192. Kept fraction = dry-run kept fraction (also given at 0.5 / 0.8).
Throughput is not measurable on a T4; we bracket vLLM aggregate decode on one
H100 at 1.5k / 3k / 5k tok/s for the 4-8B solvers (plan assumption 3-6k) and
2k tok/s for gpt-oss-120b, and add model-load / download overhead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

N_PROBLEMS, N_MODELS, S1, S3, TYPES, CAP = 173, 4, 4, 2, 5, 12288


def recs(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip().startswith("{") and '"key"' in l
            and not l.startswith('{"key": "TORN')]


def per_gen(rows, cap_dry):
    stopped = [r["tokens"] for r in rows if r["finish_reason"] != "length"]
    n_cap = sum(r["finish_reason"] == "length" for r in rows)
    n = len(rows)
    base = sum(stopped)
    return {"n": n, "n_capped": n_cap, "mean_stopped": round(base / max(1, len(stopped)), 1),
            "low": (base + n_cap * cap_dry) / n, "mid": (base + n_cap * 8192) / n, "high": (base + n_cap * CAP) / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-cap", type=int, default=4096)
    ap.add_argument("--writer-tokens", type=float, default=1500.0,
                    help="gpt-oss-120b tokens per variant (reasoning + rewrite), assumed")
    args = ap.parse_args()
    s1 = recs(args.run / "stage1_generations.jsonl")
    s3_path = args.run / "stage3_generations.jsonl"
    s3 = recs(s3_path) if s3_path.exists() else []
    summary = [json.loads(l) for l in (args.run / "stage1_summary.jsonl").read_text().splitlines() if l.strip()]
    kept_frac = sum(r["kept"] for r in summary) / len(summary)
    g = per_gen(s1 + s3, args.dry_cap)
    out = {"dry_run": {"stage1_gens": len(s1), "stage3_gens": len(s3), "per_gen_tokens": g,
                       "kept_fraction": kept_frac}, "scenarios": {}}
    for kf_name, kf in (("dry_run_kept", kept_frac), ("kept_0.5", 0.5), ("kept_0.8", 0.8)):
        n1 = N_PROBLEMS * N_MODELS * S1
        pairs = N_PROBLEMS * N_MODELS * kf
        n3 = pairs * TYPES * S3
        kept_any = min(N_PROBLEMS, N_PROBLEMS * (1 - (1 - kf) ** 2))  # models correlated: between kf and 1-(1-kf)^4
        n2 = kept_any * TYPES * 1.3  # 30% retries
        sc = {"gens_stage1": n1, "gens_stage3": round(n3), "writer_gens": round(n2)}
        for tl in ("low", "mid", "high"):
            tok = (n1 + n3) * g[tl]
            sc[f"solver_Mtok_{tl}"] = round(tok / 1e6, 2)
            for tps in (1500, 3000, 5000):
                sc[f"solver_h_{tl}_{tps}tps"] = round(tok / tps / 3600, 2)
        sc["writer_Mtok"] = round(n2 * args.writer_tokens / 1e6, 2)
        sc["writer_h_2000tps"] = round(n2 * args.writer_tokens / 2000 / 3600, 2)
        sc["overhead_h"] = 1.0  # 8 solver engine starts + gpt-oss-120b download (~65 GB) and load + env setup
        sc["total_h_mid_3000tps"] = round(sc["solver_h_mid_3000tps"] + sc["writer_h_2000tps"] + sc["overhead_h"], 2)
        sc["total_h_high_1500tps"] = round(sc["solver_h_high_1500tps"] + sc["writer_h_2000tps"] + sc["overhead_h"], 2)
        sc["total_h_low_5000tps"] = round(sc["solver_h_low_5000tps"] + sc["writer_h_2000tps"] + sc["overhead_h"], 2)
        out["scenarios"][kf_name] = sc
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
