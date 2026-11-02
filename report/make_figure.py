"""Figure 1 of the report, from committed results only.

    python report/make_figure.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / "self-perturb/results/grok"
LABEL = {"Qwen_Qwen3.5-4B": "Qwen3.5-4B", "Skywork_Skywork-OR1-Math-7B": "Skywork-OR1-7B",
         "allenai_Olmo-3-7B-Think": "Olmo-3-7B", "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-8B"}

fig, (left, right) = plt.subplots(1, 2, figsize=(5.5, 2.1), constrained_layout=True)

scores = {"Full v2b": 10 / 14,
          "White-box\nonly": json.load(open(RUN / "ablation-v2b/wbonly-1026158/scores.json"))["accuracy"],
          "No\nwhite-box": json.load(open(RUN / "ablation-v2b/nowb-1019496/scores.json"))["accuracy"]}
left.bar(list(scores), list(scores.values()), color=["#2f5597", "#2f5597", "#b7b7b7"], width=0.6)
left.axhline(0.5, color="black", lw=0.8, ls="--")
left.text(2.42, 0.51, "chance", ha="right", va="bottom", fontsize=7)
left.set_ylim(0, 1)
left.set_ylabel("Public test accuracy", fontsize=8)
left.tick_params(labelsize=7)

for safe, name in LABEL.items():
    r = np.load(RUN / f"T-024/patch__{safe}.npz", allow_pickle=True)
    pert, patch = r["kl_pert"].astype(float), r["kl_patch"].astype(float)
    removed = 100 * (1 - np.nanmedian(patch, (0, 1)) / np.nanmedian(pert))
    right.plot(r["depths"].astype(float), removed, marker="o", ms=3, lw=1.2, label=name)
right.set_xlabel("Patched depth (fraction of layers)", fontsize=8)
right.set_ylabel("Output shift removed (%)", fontsize=8)
right.set_ylim(-5, 100)
right.tick_params(labelsize=7)
right.legend(fontsize=6, frameon=False)

fig.savefig(ROOT / "figure1.pdf")
