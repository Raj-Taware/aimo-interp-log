"""T-016 phases on Kaggle T4x2 (run by the kernel from make_kernel_t016.py).

step0 : vLLM 0.31.0 on T4 -- Qwen3.5-4B (1 GPU, --text-only) and DeepSeek-R1-0528-Qwen3-8B (TP 2),
        8 gate prompts, 2048 tokens, --chunk-size 64; attention-backend fallbacks; tok/s.
gate  : stage 0 gold (DeepSeek, 8 samples, >=6 agree) -> stage 1 (4 models x 4 samples)
        -> Claude-written variants -> stage 3 (2 samples) -> gate_check. Resumable via .done markers.
"""
from __future__ import annotations

import json, os, subprocess, sys, time, urllib.request
from pathlib import Path

OUT = Path("/kaggle/working/T-016")
TV2_URL = "https://huggingface.co/api/datasets/aimo-interp/train-main-v2/parquet/default/train/0.parquet"
MODELS = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
          "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
GOLD = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
BACKENDS = [None, "TRITON_ATTN", "FLEX_ATTENTION"]
T0 = time.time()
HARD_STOP = float(os.environ.get("T016_HARD_STOP_H", "11.3")) * 3600


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def sh(cmd, env=None, timeout=None):
    log("$", " ".join(map(str, cmd)))
    try:
        return subprocess.run(list(map(str, cmd)), env={**os.environ, **(env or {})}, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        log("TIMEOUT"); return 124


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    rc = sh([sys.executable, "-m", "pip", "install", "-q", "vllm==0.31.0", "transformers==5.13.0", "pandas", "pyarrow"])
    # Kaggle image ships torchaudio built for cu128; vLLM 0.31 pulls torch cu130 -> transformers import fails.
    sh([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"])
    env = subprocess.run([sys.executable, "-c", "import json,torch,vllm,transformers;print(json.dumps({'torch':torch.__version__,"
                          "'cuda':torch.version.cuda,'vllm':vllm.__version__,'transformers':transformers.__version__,"
                          "'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
                          "'cc':list(torch.cuda.get_device_capability(0))}))"], capture_output=True, text=True)
    info = {"pip_rc": rc, "env": env.stdout.strip(), "env_err": env.stderr[-3000:]}
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout
    info["nvidia_smi"] = smi
    log("ENV", json.dumps(info)); (OUT / "env.json").write_text(json.dumps(info, indent=2))
    tv2 = OUT / "train-main-v2.parquet"
    if not tv2.exists():
        urllib.request.urlretrieve(TV2_URL, tv2)
    return tv2


def flags(model):
    # Qwen3.5-4B (hybrid GDN) on one T4: only ~157-216 Mamba cache blocks -> cap concurrency at 128.
    return (["--text-only", "--max-num-seqs", "128"] if model == "Qwen/Qwen3.5-4B"
            else ["--tensor-parallel-size", "2"])


def run_with_fallback(script, model, args, tag, timeout=None):
    """Try default attention backend, then TRITON_ATTN, FLEX_ATTENTION. Record which worked."""
    rec = OUT / "backend_used.json"
    used = json.loads(rec.read_text()) if rec.exists() else {}
    order = BACKENDS
    if model in used:  # reuse what worked
        order = [used[model]] + [b for b in BACKENDS if b != used[model]]
    for b in order:
        env = {"VLLM_ATTENTION_BACKEND": b} if b else {}
        rc = sh([sys.executable, f"scripts/label_gen/{script}", "--backend", "vllm", "--model", model,
                 *flags(model), *args], env=env, timeout=timeout)
        log(f"{tag} {model} backend={b} rc={rc}")
        if rc == 0:
            used[model] = b; rec.write_text(json.dumps(used, indent=2)); return b, rc
    return None, rc


def step0():
    tv2 = setup()
    sys.path.insert(0, "scripts/label_gen")
    from stage0_gold import gate_problems
    from common import write_jsonl, read_jsonl
    probs = [{**p, "answer": ""} for p in gate_problems(tv2)[:8]]
    pf = OUT / "step0_problems.jsonl"; write_jsonl(pf, probs)
    rp = OUT / "step0_report.json"
    res = json.loads(rp.read_text()) if rp.exists() else {}
    runs = [(m, 1, "") for m in ["Qwen/Qwen3.5-4B", GOLD]] + [(m, 8, "@64seq") for m in ["Qwen/Qwen3.5-4B", GOLD]]
    for m, ns, tag in runs:
        d = OUT / "step0" / (m.replace("/", "_") + tag.replace("@", "_"))
        t = time.time()
        b, rc = run_with_fallback("stage1_solve.py", m, ["--problems", pf, "--out", d, "--samples", str(ns),
                                  "--max-new-tokens", "2048", "--chunk-size", "64"], "step0", timeout=2400)
        r = {"rc": rc, "backend": b, "wall_s": round(time.time() - t, 1)}
        g = d / "stage1_generations.jsonl"
        if g.exists():
            recs = read_jsonl(g); tok = sum(x["tokens"] for x in recs); sec = max(x["batch_seconds"] for x in recs)
            r.update(n=len(recs), tokens=tok, gen_seconds=sec, tok_per_s=round(tok / sec, 1) if sec else None,
                     n_hit_cap=sum(x["finish_reason"] == "length" for x in recs))
        res[m + tag] = r; log("STEP0", m + tag, r)
        (OUT / "step0_report.json").write_text(json.dumps(res, indent=2))


def gate():
    tv2 = setup()
    G = OUT / "gate"; D = OUT / ".done"; D.mkdir(parents=True, exist_ok=True)
    py = [sys.executable]
    def step(k):
        return not (D / k).exists() and time.time() - T0 < HARD_STOP
    if step("gold"):
        b, rc = run_with_fallback("stage0_gold.py", GOLD, ["--samples", "8", "--min-agree", "6", "--max-new-tokens",
                                  "12288", "--trainv2", tv2, "--out", G, "--dtype", "float16"], "gold")
        if rc: sys.exit(rc)
        (D / "gold").touch()
    for m in MODELS:
        k = "s1_" + m.replace("/", "_")
        if step(k):
            b, rc = run_with_fallback("stage1_solve.py", m, ["--problems", G / "problems.jsonl", "--out", G,
                                      "--samples", "4", "--max-new-tokens", "12288"], "s1")
            if rc == 0: (D / k).touch()
    if step("variants"):
        if sh(py + ["scripts/label_gen/claude_variants_gate.py", "--problems", G / "problems.jsonl", "--out", G]) == 0:
            (D / "variants").touch()
    for m in MODELS:
        k = "s3_" + m.replace("/", "_")
        if step(k) and (D / "variants").exists():
            b, rc = run_with_fallback("stage3_solve_variants.py", m, ["--problems", G / "problems.jsonl",
                                      "--stage1", G / "stage1_summary.jsonl", "--variants", G / "stage2_variants.jsonl",
                                      "--out", G, "--samples", "2", "--max-new-tokens", "12288",
                                      "--source", "self-labelled gate (T-016, Kaggle T4x2)"], "s3")
            if rc == 0: (D / k).touch()
    if (G / "labels.jsonl").exists():
        sh(py + ["scripts/label_gen/gate_check.py", "--trainv2", tv2, "--labels", G / "labels.jsonl",
                 "--out", G / "gate_report.json"])
    log("done markers", sorted(p.name for p in D.iterdir()))


if __name__ == "__main__":
    {"step0": step0, "gate": gate}[sys.argv[1]]()
