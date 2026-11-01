"""T-023/T-024: generate the Kaggle kernel for a forward-only or gradient sample on the labelled rows.

    python scripts/make_kernel_t023.py --out <dir>
    python scripts/make_kernel_t023.py --out <dir> --task T-024 --dump dump_patch.py --slug aimo-t024-patch
    kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4

Same environment as T-019/T-021 (torch pin, transformers 5.13.0, fp16). Runs dump_saliency.py on
the train-main-v2 and self-domain problems for all four small models (smoke of 3 rows on the
first), clearing the HF cache between models. Writes T-023/saliency__<model>.npz, logs/,
kernel_report.json. No secrets; rows carry problems only, labels stay offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from make_kernel_t022 import MODELS, RUNNER, ROOT, TORCH_PIN

STATES = ROOT / "results/grok/T-019/states"
SELFDOMAIN = ROOT / "results/grok/T-021/dump/selfdomain__deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.npz"
TEXT = ["src/selfperturb/__init__.py", "src/selfperturb/perturbations.py", "src/selfperturb/features.py",
        "src/selfperturb/geometry.py"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--slug", default="aimo-t023-saliency")
    parser.add_argument("--task", default="T-023")
    parser.add_argument("--dump", default="dump_saliency.py")
    parser.add_argument("--user", default="rajtaware10")
    args = parser.parse_args()
    text = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in [*TEXT, f"scripts/{args.dump}"]}
    problems = list(np.load(SELFDOMAIN, allow_pickle=True)["problems"])
    for f in sorted(STATES.glob("trainv2__*__states.npz")):
        problems += list(np.load(f, allow_pickle=True)["problems"])
    text["rows.jsonl"] = "".join(json.dumps({"problem": p}) + "\n" for p in dict.fromkeys(map(str, problems)))
    source = (RUNNER.replace("make_kernel_t022", "make_kernel_t023").replace("T-022", args.task).replace("T022", "T02X")
              .replace("dump_recall.py", args.dump).replace('"--pool"', '"--rows"')
              .replace("results/grok/T-014/problems.jsonl", "rows.jsonl")
              .replace("@@TEXT@@", repr(json.dumps(text))).replace("@@MODELS@@", repr(json.dumps(MODELS)))
              .replace("@@TORCH_PIN@@", repr(TORCH_PIN)))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.py").write_text(source)
    (args.out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{args.user}/{args.slug}", "title": args.slug, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
        "model_sources": []}, indent=2))
    print("wrote", args.out, "rows", text["rows.jsonl"].count("\n"))


if __name__ == "__main__":
    main()
