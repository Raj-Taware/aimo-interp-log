# T-015 run plan — Claude operates Lightning H100 (2026-10-12)

Raj: Claude runs T-015 (Grok back Tue 13 Oct 12:00 IST). Budget 2 x 5 h
slots. Raw traces -> private HF dataset `Raj-taware/aimo-t015-raw`.

## Why this run matters (what each output decides)

| output | decides |
| --- | --- |
| labels on 173 AIME/AMC pairs (competition-like) | **nomination**: v2 vs v4 (vs v5) scored on the closest thing to the hidden test |
| geometry dumps on those pairs | pre-registered H1 confirmation -> v5 or not |
| ~7k graded generations + 64-token heads | trains Phase-2 early-correctness probe |

If the run only half-finishes, the order below still leaves the most
valuable thing done first.

## Finding before start: the gate has no gold answers

`train-main-v2` columns: `model_id, reasoning_effort, problem, is_robust,
max_drop` — **no answer**. 23 small-track problems, 34 labelled pairs, none
in T-006 or the AIME/AMC pool. Fix (stage 0): gpt-oss-120b (high effort)
solves each gate problem 8 times; gold = majority answer if >= 6/8 agree,
else the problem is dropped from the gate (report how many).

## Access (no secrets through Claude)

- Lightning: `lightning-sdk` on Raj's PC; **Raj runs `lightning login`
  himself** (browser). Claude then drives one Studio via the SDK.
- HF: Studio never gets a token. Claude downloads raw outputs from the
  Studio to this PC and uploads with Raj's existing local `hf` login.
- Code: uploaded from the local clone (repo is private; no GitHub creds in
  the Studio).

## Slot 1 (hard stop 4 h 45 m)

| step | what | est. |
| --- | --- | --- |
| 1 | Studio on H100 80GB; check driver (>=580 else `+cu129` wheel), disk (>=200 GB); venv vLLM 0.31.0 + transformers 5.13.0; start all 5 model downloads in background | 0.3 h |
| 2 | stage 0: gpt-oss gold for 23 gate problems | 0.15 h |
| 3 | gate: stage 1 -> 2 -> 3 on gate problems, 4 models | 0.6 h |
| 4 | **gate check** (Claude): agreement >= 0.70 and kappa >= 0.30 on labelled pairs, max_drop correlation; record tok/s. **Fail -> stop, report, no pool run.** | — |
| 5 | budget: projected total from measured tok/s; if > 9 h, stage-3 samples 2 -> 1 (then cap) | — |
| 6 | stage 1 on 173 pool problems, 4 models | ~1.5 h |
| 7 | stage 2: gpt-oss writes pool variants; spot-check file pulled and reviewed | 0.3 h |
| 8 | sync to PC + HF; stop Studio | — |

## Slot 2

| step | what |
| --- | --- |
| 1 | stage 3 on kept pool pairs, 4 models |
| 2 | geometry dump (fp16, transformers 5.13.0, train-main-v2 schema) for every labelled pair |
| 3 | 64-token heads export; sync; stop Studio |

## Code changes before the run (small, committed first)

1. Store `head_ids` (first 64 generated token ids) in stage 1 and stage 3
   records (both backends) — T-015 amendment 2.
2. `stage0_gold.py`: majority-vote gold answers -> gate `problems.jsonl`.
3. `run_t015.sh`: one resumable driver per slot; each step writes a
   `.done` marker, so a dropped connection or a stopped Studio just re-runs
   the script; hard stop at 4 h 45 m.

## Pre-registered before stage 3 labels exist (committed in slot 1)

- H1 (as already registered) and H1-head: per-head excess attention, heads
  chosen without labels (top-k by mean excess on all pool pairs, k = 5% of
  attention heads), score = mean over those heads; same AUC > 0.60, p < 0.05
  bar.
- Nomination rule: run v2, v4 inference on the labelled pool pairs;
  nominate the higher balanced accuracy; tie (diff < 0.03) -> v2 (simpler).

## Risks

| risk | fallback |
| --- | --- |
| Qwen3.5-4B fp16 GDN fails | `--dtype bfloat16` for generation only |
| gpt-oss-120b OOM / slow at 80 GB | `--max-model-len 16384`, `--reasoning-effort medium` |
| H100 not available / queue | wait; nothing spent |
| traces longer than estimate (11.5 h case) | budget rule step 5 |
| Claude's session dies mid-slot | driver keeps running in Studio; Grok can resume Tue with the same script |

## Update 2026-10-13 — Kaggle-only (Lightning blocked)

Raj has $8 Lightning credit and every GPU start is refused ("Insufficient
balance"; L40S/L4 403). Studio `aimo-t015` stays prepared on disk
(vLLM venv + gpt-oss-120b weights) in case credit appears. New route
(T-016, Grok, Kaggle T4x2): gold answers by DeepSeek-R1-0528 majority (>= 6/8),
variants written by Claude (gpt-oss-120b does not fit a T4), small models via
vLLM on T4. Methodological change: variants written by a different LLM than
the organizers' — the gate measures whether that matters.

## Decision 2026-10-19 — labelling stopped

T-016: vLLM works on T4 but long traces run at ~60 tok/s; DeepSeek hit the
12288 cap on 176/184 gold attempts (gold for 1/23; tv2-20 = 277, checked by
hand). Pool needs 25-55M tokens = 115-255 T4 h vs ~30 Kaggle h/week before
Nov 1. Raj chose: stop labelling; nominate from existing evidence; spend the
time on the write-up plus one exploratory per-head attention test (T-017).
The pre-registered H1 / H1-head confirmation and nomination rule are not run
(no labels) — the write-up says so.
