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
* LaTeX-safe -- math spans are masked out before editing and restored after,
  so the answer-preserving perturbations never touch mathematical content.
  ``number_jitter`` is the deliberate exception: it edits numbers wherever they
  live, math included, because changing a quantity is the whole point of it.

Two kinds of perturbation live here, and the distinction matters:

* **Answer-preserving** (rename, typos, distract, rephrase, domain): the
  deductive chain is untouched, so a robust model's answer should not change.
  These mirror the organizers' categories.
* **Answer-changing** (number_jitter): the answer legitimately differs. This is
  useless as a correctness check but valuable as a *sensitivity probe* -- we
  measure representation drift, never correctness, so a model that barely reacts
  to a changed quantity is telling us it is pattern-matching rather than reading
  the numbers.
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

# Substitution tables. Each is applied whole-word and case-sensitively on the
# capitalized form, so "Ann" -> "Zara" but "annual" is left alone.
_NAMES = {
    "Alice": "Priya", "Bob": "Tomas", "Carol": "Ingrid", "Dave": "Kenji",
    "Eve": "Noor", "Ann": "Zara", "Tom": "Rafael", "John": "Hiroshi",
    "Mary": "Lucia", "Sarah": "Amara", "Mike": "Dmitri", "Anna": "Fatima",
}
_OBJECTS = {
    "apples": "crates", "apple": "crate", "balls": "tokens", "ball": "token",
    "boxes": "bins", "box": "bin", "cards": "tiles", "card": "tile",
    "coins": "beads", "coin": "bead", "marbles": "pebbles", "marble": "pebble",
    "students": "engineers", "student": "engineer", "books": "ledgers",
    "book": "ledger", "candies": "rations", "candy": "ration",
}
# Domain shift shoves the same structure into another setting.
_DOMAIN = {
    "school": "shipyard", "class": "crew", "classroom": "workshop",
    "teacher": "foreman", "shop": "depot", "store": "depot",
    "garden": "reactor bay", "farm": "refinery", "game": "trial",
    "player": "operator", "team": "squadron", "race": "launch sequence",
}
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
    """Whole-word substitution, preserving leading capitalization."""
    for source, target in table.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b")
        text = pattern.sub(target, text)
        capitalized = re.compile(rf"\b{re.escape(source.capitalize())}\b")
        text = capitalized.sub(target.capitalize(), text)
    return text


def _prose_only(function):
    """Run a perturbation on the prose, leaving math spans untouched."""

    def wrapper(text: str, seed: int = 0) -> str:
        masked, spans = _mask_math(text)
        return _unmask_math(function(masked, seed), spans)

    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper


@_prose_only
def rename(text: str, seed: int = 0) -> str:
    """Swap people and objects for different ones. Answer-preserving."""
    return _apply_table(_apply_table(text, _NAMES), _OBJECTS)


@_prose_only
def domain(text: str, seed: int = 0) -> str:
    """Move the same structure into an unrelated setting. Answer-preserving."""
    return _apply_table(_apply_table(text, _DOMAIN), _OBJECTS)


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
    "domain": domain,
    "typos": typos,
    "distract": distract,
    "rephrase": rephrase,
    "number_jitter": number_jitter,
}

ANSWER_PRESERVING = ("rename", "domain", "typos", "distract", "rephrase")


def perturb_all(text: str, seed: int = 0) -> dict[str, str]:
    """Return every perturbation of ``text``, keyed by name."""
    return {name: function(text, seed) for name, function in PERTURBATIONS.items()}
