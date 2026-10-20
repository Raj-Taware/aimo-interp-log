"""T-017: generate the self-contained Kaggle kernel (per-head distractor attention, train-main-v2).

    python scripts/make_kernel_t017.py --out /workspace/kaggle-aimo-t017/aimo-t017-heads
    kaggle kernels push -p /workspace/kaggle-aimo-t017/aimo-t017-heads --accelerator NvidiaTeslaT4

Order: rule-3 canary (DeepSeek, collab/canary.jsonl vs T-008), then the
train-main-v2 geometry dump per model (DeepSeek, Qwen3.5-4B, Skywork-OR1-Math-7B,
Olmo-3-7B-Think), then ``eval_heads.py`` on the four dumps (JSON saved as
``eval_heads.json``). After each model's dump its 182 features are compared with
the embedded T-013 trainv2 dump under the amended T-017 canary rule (Claude,
2026-10-20): every feature except ``distract_attnx_*`` bit-exact (max |diff| 0),
``distract_attnx_*`` within 1e-6 relative, NaN positions equal everywhere, same
problems and feature names. On any mismatch the kernel stops before the next
model and eval_heads.py is not run. Torch is pinned to the
T-012/T-013 build (2.11.0+cu128) if the image differs. Each step is its own
process with its stdout kept under logs/; the HF cache is cleared between models.

Rows: the same train-main-v2 rows as T-012/T-013. The live parquet has since
gained 12 Skywork rows (3 -> 15), so pass the parquet T-013 embedded via
``--trainv2`` (sha256 checked against TRAINV2_T013_SHA256). The MATH and train-main-v2 parquets are embedded (pinned to
what the dump scripts' default URLs served at generation time), so the kernel
reads no dataset from the network. Embeds no secrets -- every checkpoint is public.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
TEXT = ["src/selfperturb/__init__.py", "src/selfperturb/perturbations.py", "src/selfperturb/features.py",
        "src/selfperturb/geometry.py", "scripts/dump_features.py", "scripts/dump_geometry.py",
        "scripts/canary.py", "scripts/evaluate.py", "scripts/eval_geometry.py", "scripts/eval_heads.py"]
BINARY = {
    "ref/deepseek_t008.npz": ROOT / "results/grok/T-008/dumps/deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz",
    "data/canary.jsonl": REPO / "collab/canary.jsonl",
    **{f"ref/t013/trainv2__{m}.npz": f"T013:self-perturb/results/grok/T-013/dumps/trainv2__{m}.npz"
       for m in ["deepseek-ai_DeepSeek-R1-0528-Qwen3-8B", "Qwen_Qwen3.5-4B",
                 "Skywork_Skywork-OR1-Math-7B", "allenai_Olmo-3-7B-Think"]},
}
URLS = {
    "data/trainv2.parquet": "https://huggingface.co/api/datasets/aimo-interp/train-main-v2/parquet/default/train/0.parquet",
}
TRAINV2_T013_SHA256 = "bedfb551ea0f4d95"  # prefix of the T-012/T-013 embedded parquet
DEEPSEEK = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODELS = [DEEPSEEK, "Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think"]
ATTNX_PREFIX = "distract_attnx_"
ATTNX_REL_TOL = 1e-6


def canary_compare(new, ref, attnx_prefix="distract_attnx_", rel_tol=1e-6):
    """Amended T-017 canary (Claude, 2026-10-20). ``new``/``ref`` are npz-like mappings with
    ``features`` (rows x 182), ``problems`` and ``names``. Non-attnx features must be bit-exact;
    ``distract_attnx_*`` must satisfy |new - ref| <= rel_tol * |ref| (ref == 0 => exact);
    NaN positions must be equal everywhere. Returns a JSON-able dict with ``pass``."""
    import numpy as np
    a, b = np.asarray(new["features"], dtype=np.float64), np.asarray(ref["features"], dtype=np.float64)
    names_new, names_ref = [str(x) for x in new["names"]], [str(x) for x in ref["names"]]
    res = {"rule": f"non-attnx exact; {attnx_prefix}* rel<={rel_tol:g}", "rows_new": int(len(a)),
           "rows_ref": int(len(b)), "n_features": int(a.shape[1]) if a.ndim == 2 else None,
           "problems_equal": [str(x) for x in new["problems"]] == [str(x) for x in ref["problems"]],
           "names_equal": names_new == names_ref}
    ok = a.shape == b.shape and res["problems_equal"] and res["names_equal"]
    if a.shape == b.shape and a.ndim == 2:
        attnx = np.array([n.startswith(attnx_prefix) for n in names_ref])
        nan_a, nan_b = np.isnan(a), np.isnan(b)
        nan_same = bool((nan_a == nan_b).all())
        diff = np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0))
        scale = np.abs(np.nan_to_num(b, nan=0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(diff == 0, 0.0, np.where(scale > 0, diff / scale, np.inf))
        d_other, d_attnx, r_attnx = diff[:, ~attnx], diff[:, attnx], rel[:, attnx]
        other_ok = float(d_other.max(initial=0.0)) == 0.0
        attnx_ok = bool((d_attnx <= rel_tol * scale[:, attnx]).all())
        res.update(nan_positions_equal=nan_same, n_attnx_features=int(attnx.sum()),
                   max_abs_diff=float(diff.max(initial=0.0)),
                   non_attnx_max_abs_diff=float(d_other.max(initial=0.0)),
                   non_attnx_cells_differ=int((d_other > 0).sum()),
                   attnx_max_abs_diff=float(d_attnx.max(initial=0.0)),
                   attnx_max_rel_diff=float(r_attnx.max(initial=0.0)),
                   attnx_cells_differ=int((d_attnx > 0).sum()),
                   n_cells_differ=int((diff > 0).sum() + (nan_a != nan_b).sum()),
                   non_attnx_exact=other_ok, attnx_within_tol=attnx_ok)
        ok = ok and nan_same and other_ok and attnx_ok
    res["pass"] = bool(ok)
    return res

RUNNER = '''\
"""Generated by scripts/make_kernel_t017.py -- edit the generator, not this file."""
import base64, json, os, shutil, subprocess, sys, time
import numpy as np
from pathlib import Path
T_START = time.time()

{compare_src}

DEEPSEEK = {deepseek!r}
MODELS = {models!r}
ATTNX_PREFIX = {attnx_prefix!r}
ATTNX_REL_TOL = {attnx_rel_tol!r}
TEXT = json.loads({text!r})
BINARY = json.loads({binary!r})

work = Path("/kaggle/working/repo")
for rel, content in TEXT.items():
    p = work / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content, encoding="utf-8")
for rel, blob in BINARY.items():
    p = work / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode(blob))

import torch as _t
if _t.__version__ != "2.11.0+cu128" and not os.environ.get("T017_PINNED"):
    print("PIN torch 2.11.0+cu128 (image has", _t.__version__, ")", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.11.0",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    del _t
    os.environ["T017_PINNED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==5.13.0", "accelerate==1.13.0"], check=False)
import torch, transformers, accelerate
env = {{"torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}}
print("ENV", json.dumps(env), flush=True)

out = Path("/kaggle/working/T-017")
(out / "logs").mkdir(parents=True, exist_ok=True)
dumps = out / "dumps"
timings = {{}}
checks = {{}}
eval_heads = None
report_path = out / "kernel_report.json"

def write_report():
    wall = round(time.time() - T_START, 1)
    report_path.write_text(json.dumps({{"env": env, "timings": timings, "wall_seconds": wall,
                                       "wall_hours": round(wall / 3600, 3), "trainv2_canary": checks,
                                       "eval_heads": eval_heads}}, indent=2))

def step(key, cmd):
    t0 = time.time()
    print("STEP", key, flush=True)
    log = out / "logs" / (key.replace("/", "_").replace("::", "__") + ".log")
    with open(log, "w") as handle:
        proc = subprocess.Popen(cmd, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True); handle.write(line)
        rc = proc.wait()
    timings[key] = {{"seconds": round(time.time() - t0, 1), "returncode": rc}}
    print("DONE", key, json.dumps(timings[key]), flush=True)
    write_report()

def clear_cache():
    shutil.rmtree(Path.home() / ".cache" / "huggingface" / "hub", ignore_errors=True)
    print("DISK", shutil.disk_usage("/kaggle/working"), flush=True)

py = sys.executable
tv2_src = ["--source", "data/trainv2.parquet"]

def compare(model):
    safe = model.replace("/", "_")
    new = np.load(dumps / f"trainv2__{{safe}}.npz", allow_pickle=True)
    ref = np.load(f"{{work}}/ref/t013/trainv2__{{safe}}.npz", allow_pickle=True)
    res = canary_compare(new, ref, ATTNX_PREFIX, ATTNX_REL_TOL)
    ok = res["pass"]
    if "profile__distract__attnx_heads" in new.files:
        res["heads_shape"] = list(new["profile__distract__attnx_heads"].shape)
        res["heads_case_rows"] = int(len(new["profile__distract__attnx_heads_case"]))
    res["pass"] = bool(ok)
    checks[model] = res
    print("CANARY", model, json.dumps(res), flush=True)
    write_report()
    return ok

step("canary", [py, "scripts/canary.py", "--canary", "data/canary.jsonl",
                "--reference", "ref/deepseek_t008.npz", "--out", str(out / "canary")])
for model in MODELS:
    step(f"trainv2::{{model}}", [py, "scripts/dump_geometry.py", "--kind", "trainv2", "--model", model,
                                *tv2_src, "--out", str(dumps)])
    clear_cache()
    if timings[f"trainv2::{{model}}"]["returncode"] != 0 or not compare(model):
        print("STOP canary mismatch or dump failure at", model, flush=True)
        checks["stopped_at"] = model
        break
else:
    files = [str(dumps / f"trainv2__{{m.replace('/', '_')}}.npz") for m in MODELS]
    t0 = time.time()
    print("STEP eval_heads", flush=True)
    with open(out / "logs" / "eval_heads.stderr.log", "w") as err:
        proc = subprocess.run([py, "scripts/eval_heads.py", "--tv2", *files], cwd=work,
                              stdout=subprocess.PIPE, stderr=err, text=True)
    (out / "logs" / "eval_heads.stdout.log").write_text(proc.stdout)
    timings["eval_heads"] = {{"seconds": round(time.time() - t0, 1), "returncode": proc.returncode}}
    print(proc.stdout, flush=True)
    try:
        eval_heads = json.loads(proc.stdout)
        (out / "eval_heads.json").write_text(json.dumps(eval_heads, indent=2))
    except Exception as exc:
        print("EVAL_HEADS parse failure", repr(exc), flush=True)
    print("DONE eval_heads", json.dumps(timings["eval_heads"]), flush=True)

write_report()
print("REPORT", report_path.read_text(), flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/workspace/kaggle-aimo-t017/aimo-t017-heads"))
    parser.add_argument("--slug", default="aimo-t017-heads")
    parser.add_argument("--user", default="rajtaware10")
    parser.add_argument("--trainv2", type=Path, help="pinned train-main-v2 parquet (T-013's)")
    arguments = parser.parse_args()
    text = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in TEXT}
    blobs = {}
    for rel, path in BINARY.items():
        if isinstance(path, str) and path.startswith("T013:"):  # T-013 reference dumps from their branch
            blobs[rel] = subprocess.run(["git", "-C", str(REPO), "show", f"origin/grok/T-013:{path[5:]}"],
                                        check=True, capture_output=True).stdout
        else:
            blobs[rel] = path.read_bytes()
    if arguments.trainv2:
        data = arguments.trainv2.read_bytes()
        assert hashlib.sha256(data).hexdigest().startswith(TRAINV2_T013_SHA256), "not the T-013 parquet"
        blobs["data/trainv2.parquet"] = data
    for rel, url in URLS.items():
        if rel in blobs:
            continue
        with urllib.request.urlopen(url) as response:
            blobs[rel] = response.read()
    binary = {rel: base64.b64encode(data).decode() for rel, data in blobs.items()}
    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "run.py").write_text(RUNNER.format(
        deepseek=DEEPSEEK, models=MODELS, attnx_prefix=ATTNX_PREFIX, attnx_rel_tol=ATTNX_REL_TOL,
        compare_src=inspect.getsource(canary_compare), text=json.dumps(text), binary=json.dumps(binary)))
    (arguments.out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{arguments.user}/{arguments.slug}", "title": arguments.slug, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": []}, indent=2))
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
