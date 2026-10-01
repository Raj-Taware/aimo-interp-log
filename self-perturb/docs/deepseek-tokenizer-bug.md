# DeepSeek-R1-0528-Qwen3-8B loses all whitespace under transformers 5.13.0

Found 2026-10-02 while reviewing Grok's T-007 generations, whose DeepSeek
outputs decoded as `dĠpointĠ(8,17):ĠĠĊ`.

## Measurement (transformers 5.13.0, tokenizers 0.22.2 — the grader's versions)

| model | `AutoTokenizer` class | round-trips `"Find the point (8,17).\nThen x + y = 25."` |
| --- | --- | --- |
| DeepSeek-R1-0528-Qwen3-8B | `LlamaTokenizer` | **no** — encodes `Findthepoint(8,17).Thenx+y=25.` |
| Qwen3.5-4B | `Qwen2Tokenizer` | yes |
| Skywork-OR1-Math-7B | `Qwen2Tokenizer` | yes |
| Olmo-3-7B-Think | `TokenizersBackend` | yes |

`PreTrainedTokenizerFast.from_pretrained(<same id>)` loads the same files
correctly (round-trips, keeps the chat template, single BOS):

```text
broken: '<｜User｜>Find the point (8,17).<｜Assistant｜>' -> ids decode to 'oint(8,17).<｜Assistant｜>'
fixed:  same prompt                                     -> ids decode to ' point (8,17).<｜Assistant｜>'
```

## Impact

- Every DeepSeek feature dump before the fix (`results/feature-dumps/*`, the
  T-002 extras) was computed on space-stripped text. Train and eval were
  equally affected, so submissions were internally consistent but measured
  the wrong input. The other three models are unaffected.
- The T-003 to T-005 conclusions rest on DeepSeek features and must be
  re-run. "Length alone predicts the label (AUC 0.744)" does not involve the
  model and stands.
- Any team calling `AutoTokenizer` for this checkpoint in the official
  environment — including the organizers' starter baselines — feeds it the
  same stripped text.

## Fix

`selfperturb.features.load_tokenizer` loads via `AutoTokenizer`, checks a
whitespace round-trip, and reloads via `PreTrainedTokenizerFast` only if the
check fails. Same code on Kaggle and on the grader.
