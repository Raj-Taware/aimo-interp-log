"""Shared pieces of the olympiad self-labelling pipeline (T-014).

Three stages (see collab/tasks/T-014.md):

1. ``stage1_solve.py``          small model x original problem -> base_acc
2. ``stage2_variants.py``       writer model -> one variant per perturbation type
3. ``stage3_solve_variants.py`` small model x variant -> perm_acc, max_drop, label

Every stage writes append-only JSONL keyed per *single sample*, re-reads it on
start and skips keys already present, so a killed run (H100 time comes in
5-hour slots) resumes losing at most the batch in flight. A torn last line
(SIGKILL mid-write) is tolerated: it is skipped on read and the next append
starts on a fresh line.

Two interchangeable generation backends:

* ``hf``   plain ``transformers`` ``generate`` (Kaggle T4 dry run; T-009 protocol)
* ``vllm`` vLLM offline ``LLM.generate`` (H100)

Both receive *token ids* built by our own ``load_tokenizer`` (which repairs the
DeepSeek tokenizer under transformers 5.13.0) and decode the sampled ids with
that same tokenizer, so prompts and texts are identical across backends.
Sampling follows T-009: temperature / top_p / top_k from the checkpoint's
``generation_config.json`` when set there, else 0.6 / 0.95 / none; the stop
tokens are never overridden (each backend reads the generation config).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent.parent / "src"  # self-perturb/src

INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
PERTURBATION_TYPES = ("domain", "distract", "rephrase", "rename", "typos")
SMALL_MODELS = (
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
)
# organizers' rule (aimo-interp/train-main-v2 README)
ROBUST_MAX_DROP = 0.10
SPURIOUS_MAX_DROP = 0.25


def log(*parts) -> None:
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def stable_seed(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2**31 - 1)


# --------------------------------------------------------------------------- io
class JsonlStore:
    """Append-only JSONL; one record per finished sample, identified by ``key``."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        self.bad_lines = 0
        self.duplicate_keys = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        self.bad_lines += 1  # torn write from a kill
                        continue
                    if rec["key"] in self.records:
                        self.duplicate_keys += 1
                        continue
                    self.records[rec["key"]] = rec

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def append(self, recs: list[dict]) -> None:
        if not recs:
            return
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                needs_newline = handle.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as handle:
            if needs_newline:
                handle.write("\n")
            for rec in recs:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for rec in recs:
            self.records[rec["key"]] = rec

    def values(self) -> list[dict]:
        return list(self.records.values())


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Atomic rewrite of a *derived* file (summaries, labels) -- not raw outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_problems(path: Path, ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    rows = read_jsonl(path)
    if ids:
        wanted = set(ids)
        rows = [r for r in rows if r["problem_id"] in wanted]
    if limit:
        rows = rows[:limit]
    return rows


# ------------------------------------------------------------------- answers
def extract_last_boxed(text: str) -> str | None:
    opener = "\\boxed{"
    last = None
    i = 0
    while True:
        i = text.find(opener, i)
        if i < 0:
            return last
        start = i + len(opener)
        depth, j = 1, start
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            last = text[start : j - 1]
            i = j
        else:
            i = start


def extract_answer(text: str) -> tuple[str | None, str]:
    """T-009 cascade: last \\boxed{} -> last **bold** -> last number after "answer is"."""
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed, "boxed"
    bold = list(re.finditer(r"\*\*([^*]+)\*\*", text))
    if bold:
        return bold[-1].group(1).strip(), "bold"
    num = list(re.finditer(r"answer is[^\d\-]*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE))
    if num:
        return num[-1].group(1), "answer_is"
    return None, "none"


def normalize_answer(ans) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:mathrm|textbf|mathbf)\{([^{}]*)\}", r"\1", s)
    s = s.replace("$", "").replace(" ", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "").rstrip(".")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    try:
        val = float(s)
        if val == int(val):
            return str(int(val))
        return repr(val)
    except (ValueError, OverflowError):
        return s


def answers_match(pred, gold) -> bool:
    p, g = normalize_answer(pred), normalize_answer(gold)
    return bool(p) and p == g


# ------------------------------------------------------------------ sampling
def resolve_sampling(model_id: str, overrides: dict | None = None) -> dict:
    """T-009 rule, read from the raw generation_config.json (backend independent)."""
    cfg = {}
    try:
        from huggingface_hub import hf_hub_download

        local_only = os.environ.get("HF_HUB_OFFLINE") == "1"
        path = hf_hub_download(model_id, "generation_config.json", local_files_only=local_only)
        cfg = json.loads(Path(path).read_text())
    except Exception:  # no generation_config.json (Qwen3.5-4B, DeepSeek-R1-0528-Qwen3-8B)
        cfg = {}
    temperature = cfg.get("temperature")
    top_p = cfg.get("top_p")
    top_k = cfg.get("top_k")
    used = {
        "temperature": float(temperature) if temperature and temperature > 0 else 0.6,
        "top_p": float(top_p) if top_p and top_p > 0 else 0.95,
        "top_k": int(top_k) if isinstance(top_k, int) and top_k > 0 else None,
        "source": "generation_config.json" if cfg else "defaults",
    }
    for name, value in (overrides or {}).items():
        if value is not None:
            used[name] = value
            used["source"] = "override"
    return used


# ------------------------------------------------------------------- prompts
def load_tokenizer(model_id: str):
    sys.path.insert(0, str(SRC_DIR))
    from selfperturb.features import load_tokenizer as _load

    tokenizer = _load(model_id, os.environ.get("HF_HUB_OFFLINE") == "1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def chat_ids(tokenizer, user: str, **template_kwargs) -> list[int]:
    """Chat-templated prompt ids. The template already carries any BOS, so no
    special tokens are added on top (T-009 used the tokenizer's default; the four
    small models' tokenizers add none there, see report)."""
    messages = [{"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **template_kwargs)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def solve_prompt(problem: str) -> str:
    return problem + INSTRUCTION


def is_looping(text: str) -> bool:
    """T-009: the last 4000 characters repeat a 40-character chunk >= 4 times."""
    tail = text[-4000:]
    if len(tail) < 160:
        return False
    counts = Counter(tail[i : i + 40] for i in range(len(tail) - 39))
    return counts.most_common(1)[0][1] >= 4


# ------------------------------------------------------------------ backends
class Request:
    __slots__ = ("key", "prompt_ids", "seed", "meta")

    def __init__(self, key, prompt_ids, seed, meta):
        self.key, self.prompt_ids, self.seed, self.meta = key, prompt_ids, seed, meta


class Result:
    __slots__ = ("text", "n_tokens", "finish_reason", "seconds")

    def __init__(self, text, n_tokens, finish_reason, seconds):
        self.text, self.n_tokens, self.finish_reason, self.seconds = text, n_tokens, finish_reason, seconds


def add_backend_args(parser) -> None:
    parser.add_argument("--backend", choices=("hf", "vllm"), default="vllm")
    parser.add_argument("--dtype", default="float16", help="float16 (rule 2) or bfloat16")
    parser.add_argument("--batch-size", type=int, default=4, help="hf: sequences per generate call")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="vllm: requests per generate call (= resume granularity)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enforce-eager", action="store_true", help="vllm: no CUDA graphs / compilation")
    parser.add_argument("--text-only", action="store_true",
                        help="vllm: limit_mm_per_prompt image/video = 0 (Qwen3.5-4B is a vision-language "
                             "checkpoint; skips vision-encoder memory profiling)")


def make_backend(args, model_id: str, tokenizer, sampling: dict, max_new_tokens: int):
    if args.backend == "hf":
        return HFBackend(model_id, tokenizer, sampling, max_new_tokens, args.batch_size, args.dtype)
    return VLLMBackend(model_id, tokenizer, sampling, max_new_tokens, args.chunk_size, args.dtype,
                       args.gpu_memory_utilization, args.tensor_parallel_size, args.max_model_len,
                       args.enforce_eager, args.text_only)


class HFBackend:
    name = "hf"

    def __init__(self, model_id, tokenizer, sampling, max_new_tokens, batch_size, dtype):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.tokenizer = tokenizer
        self.sampling = sampling
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto" if torch.cuda.is_available() else None,
            **{dtype_key: getattr(torch, dtype) if torch.cuda.is_available() else torch.float32})
        self.model.eval()
        self.load_seconds = round(time.time() - t0, 1)
        self.device = next(self.model.parameters()).device
        log(f"hf: loaded {model_id} in {self.load_seconds}s; gen_config eos="
            f"{getattr(self.model.generation_config, 'eos_token_id', None)} tokenizer eos={tokenizer.eos_token_id}")

    def generate(self, requests: list[Request]):
        """Yields (requests, results) per batch so the caller can persist each batch."""
        torch = self.torch
        from transformers import set_seed

        kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=True,
                      temperature=self.sampling["temperature"], top_p=self.sampling["top_p"],
                      pad_token_id=self.tokenizer.pad_token_id)  # eos deliberately not passed
        if self.sampling.get("top_k"):
            kwargs["top_k"] = self.sampling["top_k"]
        i = 0
        while i < len(requests):
            batch = requests[i : i + self.batch_size]
            width = max(len(r.prompt_ids) for r in batch)
            pad = self.tokenizer.pad_token_id
            ids = [[pad] * (width - len(r.prompt_ids)) + r.prompt_ids for r in batch]
            mask = [[0] * (width - len(r.prompt_ids)) + [1] * len(r.prompt_ids) for r in batch]
            input_ids = torch.tensor(ids, device=self.device)
            attention_mask = torch.tensor(mask, device=self.device)
            set_seed(batch[0].seed)
            t0 = time.time()
            try:
                with torch.no_grad():
                    out = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise
                self.batch_size = max(1, self.batch_size // 2)
                log(f"hf: OOM -> batch_size {self.batch_size}")
                continue
            seconds = time.time() - t0
            results = []
            for row in range(len(batch)):
                new = out[row, width:].tolist()
                while new and new[-1] == pad:
                    new.pop()
                finish = "length" if len(new) >= self.max_new_tokens else "stop"
                text = self.tokenizer.decode(new, skip_special_tokens=True)
                results.append(Result(text, len(new), finish, round(seconds, 1)))
            yield batch, results
            i += len(batch)


class VLLMBackend:
    name = "vllm"

    def __init__(self, model_id, tokenizer, sampling, max_new_tokens, chunk_size, dtype,
                 gpu_memory_utilization, tensor_parallel_size, max_model_len, enforce_eager=False,
                 text_only=False):
        from vllm import LLM

        self.tokenizer = tokenizer
        self.sampling = sampling
        self.max_new_tokens = max_new_tokens
        self.chunk_size = chunk_size
        extra = {"limit_mm_per_prompt": {"image": 0, "video": 0}} if text_only else {}
        t0 = time.time()
        self.llm = LLM(model=model_id, dtype=dtype, seed=0, **extra,
                       max_model_len=max_model_len or (max_new_tokens + 4096),
                       gpu_memory_utilization=gpu_memory_utilization,
                       tensor_parallel_size=tensor_parallel_size,
                       enable_prefix_caching=True, enforce_eager=enforce_eager,
                       generation_config="auto")  # stop tokens from generation_config
        self.load_seconds = round(time.time() - t0, 1)
        log(f"vllm: loaded {model_id} in {self.load_seconds}s")

    def generate(self, requests: list[Request]):
        from vllm import SamplingParams

        for i in range(0, len(requests), self.chunk_size):
            chunk = requests[i : i + self.chunk_size]
            params = []
            for r in chunk:
                kw = dict(temperature=self.sampling["temperature"], top_p=self.sampling["top_p"],
                          max_tokens=self.max_new_tokens, seed=r.seed, skip_special_tokens=True)
                if self.sampling.get("top_k"):
                    kw["top_k"] = self.sampling["top_k"]
                params.append(SamplingParams(**kw))
            prompts = [{"prompt_token_ids": r.prompt_ids} for r in chunk]
            t0 = time.time()
            outs = self.llm.generate(prompts, params, use_tqdm=True)
            seconds = time.time() - t0
            results = []
            for out in outs:
                comp = out.outputs[0]
                ids = list(comp.token_ids)
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                results.append(Result(text, len(ids), comp.finish_reason or "stop", round(seconds, 1)))
            yield chunk, results


def runtime_env() -> dict:
    env = {"python": sys.version.split()[0]}
    try:
        import torch
        import transformers

        env.update(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                   gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except Exception:
        pass
    try:
        import vllm

        env["vllm"] = vllm.__version__
    except Exception:
        pass
    return env


def free_backend(backend) -> None:
    import gc

    del backend
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
