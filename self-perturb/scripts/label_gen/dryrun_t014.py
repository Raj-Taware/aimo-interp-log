"""T-014 dry run: run stages 1 -> 2 -> 3 end to end, SIGKILL each stage once
mid-run, resume it, then re-run it once more (must be a no-op).

    python scripts/label_gen/dryrun_t014.py --problems results/grok/T-014/problems.jsonl \
        --problem-id aime-0 ... --model Skywork/Skywork-OR1-Math-7B --writer Skywork/Skywork-OR1-Math-7B \
        --out /kaggle/working/T-014/run

Per stage:
  1. launch; as soon as the first batch is persisted (raw JSONL gains a line),
     SIGKILL the whole process group -- the next batch is then in flight;
  2. (stage 1 only) append a torn, newline-less JSON fragment to the raw file,
     simulating a kill in the middle of a write;
  3. relaunch the identical command and let it finish;
  4. relaunch again -- must report 0 pending and exit without loading a model.
Checks: every record present at kill time is unchanged at the end, no duplicate
keys, final count equals the expected count. Results -> <out>/dryrun_report.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def valid_records(path: Path) -> dict[str, dict]:
    recs = {}
    if not path.exists():
        return recs
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs.setdefault(rec["key"], rec)
    return recs


def count_lines(path: Path) -> int:
    return len(valid_records(path))


def run(cmd, log_path: Path, kill_on: Path | None = None, timeout: float = 4 * 3600) -> dict:
    t0 = time.time()
    before = count_lines(kill_on) if kill_on else 0
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=HERE)
        killed = False
        while proc.poll() is None:
            time.sleep(2)
            if kill_on is not None and count_lines(kill_on) > before:
                os.killpg(proc.pid, signal.SIGKILL)
                killed = True
                proc.wait()
                break
            if time.time() - t0 > timeout:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise RuntimeError(f"timeout: {cmd}")
    text = log_path.read_text()
    return {"seconds": round(time.time() - t0, 1), "returncode": proc.returncode, "killed": killed,
            "pending_lines": [l.split(" ", 1)[1] for l in text.splitlines() if "pending" in l],
            "loaded_model": ("hf: loaded" in text) or ("vllm: loaded" in text)}


def stage_test(name, cmd, raw: Path, logs: Path, expected_fn, inject_torn=False) -> dict:
    print(f"=== {name}: kill run", flush=True)
    res = {"stage": name, "kill_run": run(cmd, logs / f"{name}_1_kill.log", kill_on=raw)}
    at_kill = valid_records(raw)
    res["records_at_kill"] = len(at_kill)
    if inject_torn:
        with raw.open("a", encoding="utf-8") as handle:
            handle.write('{"key": "TORN-WRITE-SIMULATION", "stage": 1, "text": "half a rec')
        res["injected_torn_line"] = True
    print(f"=== {name}: resume run ({len(at_kill)} records at kill)", flush=True)
    res["resume_run"] = run(cmd, logs / f"{name}_2_resume.log")
    print(f"=== {name}: no-op rerun", flush=True)
    res["noop_run"] = run(cmd, logs / f"{name}_3_noop.log")
    final = valid_records(raw)
    lines = [l for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = []
    for l in lines:
        try:
            keys.append(json.loads(l)["key"])
        except json.JSONDecodeError:
            pass
    expected = expected_fn()
    res.update({
        "records_final": len(final), "expected": expected,
        "duplicate_keys": len(keys) - len(set(keys)),
        "torn_lines_in_file": len(lines) - len(keys),
        "killed_records_unchanged": all(final.get(k) == v for k, v in at_kill.items()),
        "resumed_skipped_done": res["resume_run"]["pending_lines"],
        "noop_loaded_model": res["noop_run"]["loaded_model"],
    })
    res["pass"] = bool(res["kill_run"]["killed"] and res["resume_run"]["returncode"] == 0
                       and res["noop_run"]["returncode"] == 0 and not res["noop_loaded_model"]
                       and res["duplicate_keys"] == 0 and res["killed_records_unchanged"]
                       and len(final) == expected and 0 < len(at_kill) < expected)
    print(json.dumps({k: v for k, v in res.items() if k not in ("kill_run", "resume_run", "noop_run")}), flush=True)
    if res["resume_run"]["returncode"] != 0:
        raise SystemExit(f"{name}: resume run failed, see {logs / (name + '_2_resume.log')}")
    return res


def token_stats(path: Path, field="tokens") -> dict:
    recs = list(valid_records(path).values())
    toks = sorted(r[field] for r in recs if field in r)
    if not toks:
        return {"n": 0}
    secs = {}
    for r in recs:  # each record carries its batch's wall time; count every batch once
        secs[(r.get("model", r.get("writer")), r["batch_seconds"], r["finished_at"][:16])] = r["batch_seconds"]
    return {"n": len(toks), "sum": sum(toks), "mean": round(sum(toks) / len(toks), 1),
            "median": toks[len(toks) // 2], "max": toks[-1],
            "n_hit_cap": sum(r["finish_reason"] == "length" for r in recs),
            "gen_seconds_approx": round(sum(secs.values()), 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--problems", type=Path, required=True)
    p.add_argument("--problem-id", action="append", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--writer", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--variant-samples", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--writer-max-new-tokens", type=int, default=3072)
    p.add_argument("--min-base-acc", type=float, default=0.5)
    p.add_argument("--batch1", type=int, default=6)
    p.add_argument("--batch-cap", type=int, default=10)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--backend", default="hf")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--extra", nargs="*", default=[], help="extra args passed to every stage")
    args = p.parse_args()

    args.problems, args.out = args.problems.resolve(), args.out.resolve()
    out = args.out
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    pids = sum((["--problem-id", x] for x in args.problem_id), [])
    common = ["--backend", args.backend, "--dtype", args.dtype, "--out", str(out), *args.extra]
    report = {"args": vars(args) | {"problems": str(args.problems), "out": str(out)}, "stages": []}

    def save():
        (out / "dryrun_report.json").write_text(json.dumps(report, indent=2, default=str))

    # ---------------- stage 1
    s1 = [py, "stage1_solve.py", "--problems", str(args.problems), *pids, "--model", args.model,
          "--samples", str(args.samples), "--max-new-tokens", str(args.max_new_tokens),
          "--min-base-acc", str(args.min_base_acc), "--batch-size", str(args.batch1), "--chunk-size", str(args.batch1), *common]
    report["stages"].append(stage_test("stage1", s1, out / "stage1_generations.jsonl", logs,
                                       lambda: len(args.problem_id) * args.samples, inject_torn=True))
    summary = [json.loads(l) for l in (out / "stage1_summary.jsonl").read_text().splitlines() if l.strip()]
    kept = [r["problem_id"] for r in summary if r["kept"]]
    report["stage1_summary"] = [{k: r[k] for k in ("problem_id", "base_acc", "kept", "tokens", "extracted")}
                                for r in summary]
    report["stage1_tokens"] = token_stats(out / "stage1_generations.jsonl")
    save()
    if not kept:
        report["stopped"] = "no problem kept in stage 1; stages 2-3 have nothing to do"
        save()
        print("STOP: nothing kept", flush=True)
        return

    # ---------------- stage 2
    n2 = len(kept) * 5
    b2 = max(1, min(args.batch_cap, math.ceil(n2 / 2)))
    s2 = [py, "stage2_variants.py", "--problems", str(args.problems), "--stage1", str(out / "stage1_summary.jsonl"),
          "--writer", args.writer, "--attempts", str(args.attempts),
          "--max-new-tokens", str(args.writer_max_new_tokens), "--batch-size", str(b2), "--chunk-size", str(b2), *common]

    def expected2():
        variants = [json.loads(l) for l in (out / "stage2_variants.jsonl").read_text().splitlines() if l.strip()]
        raw = valid_records(out / "stage2_raw.jsonl")
        report["stage2_variant_count"] = len(variants)
        # every (problem, type) has either a valid attempt or all attempts used; count those attempts
        per = {}
        for r in raw.values():
            per.setdefault((r["problem_id"], r["type"]), []).append(r)
        exp = 0
        for recs in per.values():
            recs.sort(key=lambda r: r["attempt"])
            first_ok = next((i for i, r in enumerate(recs) if r["valid"]), None)
            exp += (first_ok + 1) if first_ok is not None else args.attempts
        assert len(per) == n2, (len(per), n2)
        return exp

    report["stages"].append(stage_test("stage2", s2, out / "stage2_raw.jsonl", logs, expected2))
    report["stage2_tokens"] = token_stats(out / "stage2_raw.jsonl", "writer_tokens")
    raw2 = valid_records(out / "stage2_raw.jsonl")
    report["stage2_reasons"] = {}
    for r in raw2.values():
        report["stage2_reasons"][r["reason"]] = report["stage2_reasons"].get(r["reason"], 0) + 1
    save()
    variants = [json.loads(l) for l in (out / "stage2_variants.jsonl").read_text().splitlines() if l.strip()]
    if not variants:
        report["stopped"] = "stage 2 produced no valid variant"
        save()
        print("STOP: no variants", flush=True)
        return

    # ---------------- stage 3
    n3 = sum(1 for v in variants if v["problem_id"] in kept) * args.variant_samples
    b3 = max(1, min(args.batch_cap, math.ceil(n3 / 2)))
    s3 = [py, "stage3_solve_variants.py", "--problems", str(args.problems),
          "--stage1", str(out / "stage1_summary.jsonl"), "--variants", str(out / "stage2_variants.jsonl"),
          "--model", args.model, "--samples", str(args.variant_samples),
          "--max-new-tokens", str(args.max_new_tokens), "--batch-size", str(b3), "--chunk-size", str(b3), *common]
    report["stages"].append(stage_test("stage3", s3, out / "stage3_generations.jsonl", logs, lambda: n3))
    report["stage3_tokens"] = token_stats(out / "stage3_generations.jsonl")
    report["labels"] = [json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines() if l.strip()]
    for row in report["labels"]:
        row.pop("problem", None)
    report["all_pass"] = all(s["pass"] for s in report["stages"])
    save()
    print("DRYRUN DONE all_pass=", report["all_pass"], flush=True)


if __name__ == "__main__":
    main()
