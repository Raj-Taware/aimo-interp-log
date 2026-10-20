"""T-017: sanity check of the amended canary comparator in make_kernel_t017.py.

    python scripts/check_canary_t017.py --new <run-1 trainv2__deepseek...npz>

Uses the run-1 DeepSeek trainv2 dump (attnx-only fp32 reduction-order diffs) against the
T-013 reference from origin/grok/T-013: it must PASS under the amended rule, and fail when
(a) one non-attnx cell moves by one float32 ulp, (b) one attnx cell moves by 2e-6 relative,
(c) one NaN position changes.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_kernel_t017 import REPO, canary_compare  # noqa: E402

REF = "self-perturb/results/grok/T-013/dumps/trainv2__deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, required=True)
    args = parser.parse_args()
    raw = subprocess.run(["git", "-C", str(REPO), "show", f"origin/grok/T-013:{REF}"],
                         check=True, capture_output=True).stdout
    ref = dict(np.load(io.BytesIO(raw), allow_pickle=True))
    new = dict(np.load(args.new, allow_pickle=True))
    names = [str(n) for n in ref["names"]]
    attnx = [i for i, n in enumerate(names) if n.startswith("distract_attnx_")]
    other = [i for i, n in enumerate(names) if not n.startswith("distract_attnx_")]
    feats = new["features"]

    def with_features(f):
        return {**new, "features": f}

    def finite_cell(cols):
        for r in range(feats.shape[0]):
            for c in cols:
                if np.isfinite(feats[r, c]) and ref["features"][r, c] != 0:
                    return r, c
        raise SystemExit("no finite cell")

    results = {"run1_as_is": canary_compare(new, ref)}
    f = feats.copy(); r, c = finite_cell(other)
    f[r, c] = np.nextafter(f[r, c], np.inf, dtype=f.dtype)
    results["non_attnx_one_ulp"] = canary_compare(with_features(f), ref)
    f = feats.copy(); r, c = finite_cell(attnx)
    f[r, c] = f.dtype.type(float(ref["features"][r, c]) * (1 + 2e-6))
    results["attnx_2e-6_rel"] = canary_compare(with_features(f), ref)
    f = feats.copy(); r, c = finite_cell(other)
    f[r, c] = np.nan
    results["nan_moved"] = canary_compare(with_features(f), ref)

    expect = {"run1_as_is": True, "non_attnx_one_ulp": False, "attnx_2e-6_rel": False, "nan_moved": False}
    ok = True
    for key, res in results.items():
        good = res["pass"] == expect[key]
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {key}: pass={res['pass']} "
              f"attnx max abs={res.get('attnx_max_abs_diff')} rel={res.get('attnx_max_rel_diff')} "
              f"non-attnx max abs={res.get('non_attnx_max_abs_diff')} cells attnx/other="
              f"{res.get('attnx_cells_differ')}/{res.get('non_attnx_cells_differ')}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
