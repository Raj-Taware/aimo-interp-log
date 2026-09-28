"""Deterministic, dependency-free perturbations of a math problem statement.

The competition labels a (model, problem) pair as robust when the model's answer
survives the organizers' hidden perturbations. Those perturbed statements are
never shown to a submission -- but nothing stops us from *recreating* the same
families of perturbation ourselves, at inference time, with plain string
manipulation. We can then measure how much the model's internal state moves
between the original and each self-perturbation.

Design constraints, all forced by the evaluation container:

* standard library only (no network, no extra packages);
* deterministic given (text, seed) so repeated runs score identically;
* cheap -- a perturbation is a string edit, not a model call;
* LaTeX-aware -- prose edits mask math spans out and restore them after.
  ``rename`` and ``numerals`` edit *inside* math on purpose, and only in
  answer-preserving ways (consistent variable renaming, digits to words).
  ``number_jitter`` edits numbers wherever they live, because changing a
  quantity is the whole point of it.

Olympiad statements carry their entities as math symbols ($r$, $m$, points
$A$, $B$), not as named people and objects, so renaming works on the symbols.
A first version renamed only English names and nouns and left 96% of real
problems untouched -- a no-op perturbation is a constant feature.

Two kinds of perturbation live here, and the distinction matters:

* **Answer-preserving** (rename, numerals, typos, distract, rephrase): the
  deductive chain is untouched, so a robust model's answer should not change.
  They mirror the organizers' categories except ``domain``, which needs a
  language model to do honestly and is not attempted.
* **Answer-changing** (number_jitter): the answer legitimately differs. Useless
  as a correctness check but valuable as a *sensitivity probe* -- we measure
  representation drift, never correctness, so a model that barely reacts to a
  changed quantity is telling us it is not reading the numbers.
"""

from __future__ import annotations

import random
import re

# Math spans are replaced by these placeholders while the prose is edited.
_MATH_PATTERN = re.compile(
    r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\]|\\begin\{.*?\}.*?\\end\{.*?\})",
    re.DOTALL,
)
_MASK = "\x00MATH{}\x00"

# Inside math, a lone letter not glued to a LaTeX command or another letter is
# a symbol we may rename: the m in \frac{m}{n}, the x in \sin x.
_SYMBOL = re.compile(r"(?<![\\A-Za-z])[A-Za-z](?![A-Za-z])")
# Letters whose meaning is fixed by convention, never renamed.
_RESERVED = set("deiEIO")
# Arguments of these commands are names, not variables: \mathbb{R}, \text{cm}.
_LITERAL_ARGUMENT = re.compile(
    r"\\(?:mathbb|mathrm|text|operatorname|mathcal|mathfrak)\s*\{[^{}]*\}"
)

_NAMES = {
    "Alice": "Priya", "Bob": "Tomas", "Carol": "Ingrid", "Dave": "Kenji",
    "Eve": "Noor", "Ann": "Zara", "Tom": "Rafael", "John": "Hiroshi",
    "Mary": "Lucia", "Sarah": "Amara", "Mike": "Dmitri", "Anna": "Fatima",
}
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
).split()
_DISTRACTORS = (
    "The recorded measurements were taken on a Tuesday.",
    "An unrelated note in the margin mentions a value of 37.",
    "The archivist who filed this problem retired three years ago.",
    "A second copy of the sheet was stored in a different building.",
    "The ambient temperature that day was 19 degrees.",
)
_PREFIXES = (
    "Consider the following problem.",
    "Work through the problem stated below.",
    "You are given the problem below.",
)
_SUFFIXES = (
    "Give the final answer.",
    "State the resulting value.",
    "Report the answer you obtain.",
)


def _mask_math(text: str) -> tuple[str, list[str]]:
    """Replace math spans with opaque placeholders, returning them separately."""
    spans: list[str] = []

    def _swap(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return _MASK.format(len(spans) - 1)

    return _MATH_PATTERN.sub(_swap, text), spans


def _unmask_math(text: str, spans: list[str]) -> str:
    """Put the original math spans back where their placeholders sit."""
    for index, span in enumerate(spans):
        text = text.replace(_MASK.format(index), span)
    return text


def _apply_table(text: str, table: dict[str, str]) -> str:
    """Whole-word substitution."""
    for source, target in table.items():
        text = re.sub(rf"\b{re.escape(source)}\b", target, text)
    return text


def _prose_only(function):
    """Run a perturbation on the prose, leaving math spans untouched."""

    def wrapper(text: str, seed: int = 0) -> str:
        masked, spans = _mask_math(text)
        return _unmask_math(function(masked, seed), spans)

    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper


def _edit_math(text: str, edit) -> str:
    """Apply ``edit`` to each math span, shielding literal command arguments."""

    def _span(match: re.Match[str]) -> str:
        span = match.group(0)
        shields: list[str] = []

        def _shield(inner: re.Match[str]) -> str:
            shields.append(inner.group(0))
            return f"\x01{len(shields) - 1}\x01"

        span = edit(_LITERAL_ARGUMENT.sub(_shield, span))
        for index, original in enumerate(shields):
            span = span.replace(f"\x01{index}\x01", original)
        return span

    return _MATH_PATTERN.sub(_span, text)


def rename(text: str, seed: int = 0) -> str:
    """Consistently rename math symbols (and common names). Answer-preserving.

    Every occurrence of a symbol maps to the same fresh letter of the same case,
    chosen from letters the problem never uses, so the structure is unchanged.
    """
    used: set[str] = set()
    _edit_math(text, lambda span: used.update(_SYMBOL.findall(span)) or span)
    renamable = sorted(letter for letter in used if letter not in _RESERVED)

    rng = random.Random(seed)
    mapping: dict[str, str] = {}
    for letter in renamable:
        pool = [
            candidate
            for candidate in (
                "abcfghjkmnpqrstuvwxyz" if letter.islower() else "ABCFGHJKMNPQRSTUVWXYZ"
            )
            if candidate not in used and candidate not in mapping.values()
        ]
        if pool:
            mapping[letter] = rng.choice(pool)

    renamed = _edit_math(
        text, lambda span: _SYMBOL.sub(lambda m: mapping.get(m.group(0), m.group(0)), span)
    )
    masked, spans = _mask_math(renamed)
    return _unmask_math(_apply_table(masked, _NAMES), spans)


_WORD_VALUES = {word: value for value, word in enumerate(_NUMBER_WORDS) if value >= 2}
# Either a small integer or its word, standing alone: not part of a decimal,
# fraction, or percentage. "one" and 0/1 are skipped -- too often structural
# ("no one", "at least 1").
_PROSE_NUMBER = re.compile(
    r"(?<![\d.,/])\b(\d{1,2}|" + "|".join(_WORD_VALUES) + r")\b(?![\d%]|\.\d|/)",
    re.IGNORECASE,
)


def numerals(text: str, seed: int = 0) -> str:
    """Swap small numbers between digits and words. Answer-preserving.

    Most statements write quantities in prose ("Ten treeks", "the digit 6"),
    not as bare ``$3$`` spans, so both directions are swapped in one pass:
    6 -> six, Ten -> 10, and ``$3$`` -> three.
    """

    def _span_to_word(match: re.Match[str]) -> str:
        value = int(match.group(1))
        return _NUMBER_WORDS[value] if 2 <= value <= 20 else match.group(0)

    def _swap(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.isdigit():
            value = int(token)
            return _NUMBER_WORDS[value] if 2 <= value <= 20 else token
        return str(_WORD_VALUES[token.lower()])

    text = re.sub(r"\$(\d{1,2})\$", _span_to_word, text)
    masked, spans = _mask_math(text)
    return _unmask_math(_PROSE_NUMBER.sub(_swap, masked), spans)


@_prose_only
def typos(text: str, seed: int = 0) -> str:
    """Transpose two letters inside a few long words. Answer-preserving."""
    rng = random.Random(seed)
    words = text.split(" ")
    # Only touch alphabetic words of 5+ chars: short words and anything holding
    # digits or placeholders stay intact.
    candidates = [
        index
        for index, word in enumerate(words)
        if len(word) >= 5 and word.isalpha()
    ]
    for index in rng.sample(candidates, min(3, len(candidates))):
        word = words[index]
        position = rng.randrange(1, len(word) - 2)
        words[index] = (
            word[:position] + word[position + 1] + word[position] + word[position + 2:]
        )
    return " ".join(words)


@_prose_only
def distract(text: str, seed: int = 0) -> str:
    """Insert one irrelevant but plausible sentence. Answer-preserving."""
    rng = random.Random(seed)
    sentence = rng.choice(_DISTRACTORS)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) < 2:
        return f"{text.strip()} {sentence}"
    # Drop it mid-statement, where it competes with the real premises.
    position = max(1, len(sentences) // 2)
    sentences.insert(position, sentence)
    return " ".join(sentences)


@_prose_only
def rephrase(text: str, seed: int = 0) -> str:
    """Wrap the statement in neutral framing. Answer-preserving."""
    rng = random.Random(seed)
    return f"{rng.choice(_PREFIXES)}\n\n{text.strip()}\n\n{rng.choice(_SUFFIXES)}"


def number_jitter(text: str, seed: int = 0) -> str:
    """Nudge standalone integers. NOT answer-preserving -- a sensitivity probe.

    Used only to measure how far the model's representation moves when a
    quantity it must actually read is changed. A model whose state barely
    reacts is not reading the numbers.
    """
    rng = random.Random(seed)

    def _bump(match: re.Match[str]) -> str:
        value = int(match.group(0))
        # Leave 0/1 alone: they are usually structural, not quantities.
        if value <= 1:
            return match.group(0)
        return str(value + rng.choice((1, 2, -1)))

    return re.sub(r"(?<![\d.\\])\b\d{1,3}\b(?![\d.])", _bump, text)


# Ordered so feature vectors keep a stable column order across runs.
PERTURBATIONS = {
    "rename": rename,
    "numerals": numerals,
    "typos": typos,
    "distract": distract,
    "rephrase": rephrase,
    "number_jitter": number_jitter,
}

ANSWER_PRESERVING = ("rename", "numerals", "typos", "distract", "rephrase")


def perturb_all(text: str, seed: int = 0) -> dict[str, str]:
    """Return every perturbation of ``text``, keyed by name."""
    return {name: function(text, seed) for name, function in PERTURBATIONS.items()}
