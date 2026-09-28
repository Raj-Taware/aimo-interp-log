"""Smoke checks for hybrid-cascade decision policies (conservative + symmetric)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION_ROOT = ROOT / "solutions" / "hybrid-cascade"
if str(SOLUTION_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLUTION_ROOT))

from cascade import (  # noqa: E402
    CASCADE_TRUE_RATE_CAPS,
    CONSERVATIVE_MODE,
    MAX_STAGE_TRUE_RATE,
    all_false_fallback,
    reject_overconfident_votes,
    run_cascade,
)
from probe_infer import (  # noqa: E402
    CONSERVATIVE,
    DECISION_POLICY,
    FALSE_MARGIN_THRESHOLD,
    MIN_AGREE_FRAC,
    MIN_POSITIVE_MARGIN_FRAC,
    POLICY_THRESHOLDS,
    TRUE_MARGIN_THRESHOLD,
    decide_robust_from_margins,
)


class ActivePolicyDefaultsTests(unittest.TestCase):
    def test_active_policy_is_symmetric_confident(self) -> None:
        self.assertEqual(DECISION_POLICY, "symmetric_confident")
        self.assertTrue(CONSERVATIVE)  # non-legacy
        self.assertTrue(CONSERVATIVE_MODE)
        self.assertAlmostEqual(TRUE_MARGIN_THRESHOLD, 0.30)
        self.assertAlmostEqual(FALSE_MARGIN_THRESHOLD, -0.30)
        self.assertAlmostEqual(MIN_AGREE_FRAC, 0.90)
        self.assertAlmostEqual(MIN_POSITIVE_MARGIN_FRAC, 0.90)
        self.assertAlmostEqual(MAX_STAGE_TRUE_RATE, 0.50)
        self.assertIn("conservative", POLICY_THRESHOLDS)
        self.assertIn("symmetric_confident", POLICY_THRESHOLDS)
        self.assertAlmostEqual(CASCADE_TRUE_RATE_CAPS["conservative"], 0.35)
        self.assertAlmostEqual(CASCADE_TRUE_RATE_CAPS["symmetric_confident"], 0.50)

    def test_conservative_policy_thresholds_preserved(self) -> None:
        cfg = POLICY_THRESHOLDS["conservative"]
        self.assertAlmostEqual(float(cfg["true_margin_threshold"]), 0.5)
        self.assertIsNone(cfg["false_margin_threshold"])
        self.assertAlmostEqual(float(cfg["min_agree_frac"]), 0.75)


class ConservativeProbeDecisionTests(unittest.TestCase):
    def test_legacy_mean_geq_zero_would_be_true_but_conservative_prefers_false(self) -> None:
        # Mean margin slightly positive, but weak: below raised threshold and
        # mixed seed signs → must be False under conservative policy.
        margins = [0.05, 0.02, -0.01, 0.04]
        self.assertGreater(sum(margins) / len(margins), 0.0)
        self.assertFalse(decide_robust_from_margins(margins, policy="conservative"))
        self.assertTrue(decide_robust_from_margins(margins, policy="legacy"))
        # Back-compat kwargs
        self.assertFalse(decide_robust_from_margins(margins, conservative=True))
        self.assertTrue(decide_robust_from_margins(margins, conservative=False))

    def test_high_confidence_agreement_is_true_under_conservative(self) -> None:
        margins = [1.2, 0.9, 1.1, 1.0, 0.8]
        self.assertTrue(decide_robust_from_margins(margins, policy="conservative"))

    def test_empty_margins_are_false(self) -> None:
        self.assertFalse(decide_robust_from_margins([], policy="conservative"))
        self.assertFalse(decide_robust_from_margins([], policy="symmetric_confident"))


class SymmetricConfidentDecisionTests(unittest.TestCase):
    def test_high_pos_with_strong_seed_agreement_is_true(self) -> None:
        # 12 seeds, 11/12 >= 0 → pos_frac ≈ 0.9167 >= 0.90; mean ≈ 0.45 >= 0.30
        margins = [0.5] * 11 + [-0.05]
        self.assertTrue(
            decide_robust_from_margins(margins, policy="symmetric_confident")
        )

    def test_high_pos_but_weak_seed_agreement_is_false(self) -> None:
        # mean high enough, but only 75% seeds positive → below 0.90 bar
        margins = [0.8] * 9 + [-0.1] * 3  # pos_frac=0.75, mean≈0.575
        self.assertGreater(sum(margins) / len(margins), 0.30)
        self.assertFalse(
            decide_robust_from_margins(margins, policy="symmetric_confident")
        )
        # Same margins WOULD pass conservative (τ=0.5, φ=0.75)
        self.assertTrue(decide_robust_from_margins(margins, policy="conservative"))

    def test_mid_pos_passes_symmetric_but_not_conservative(self) -> None:
        # Softened τ=0.30 catches mid-margin Trues that conservative (0.5) rejects.
        margins = [0.35] * 12  # mean=0.35, pos_frac=1.0
        self.assertTrue(
            decide_robust_from_margins(margins, policy="symmetric_confident")
        )
        self.assertFalse(decide_robust_from_margins(margins, policy="conservative"))

    def test_high_neg_symmetric_gate_is_false(self) -> None:
        margins = [-0.5] * 11 + [0.05]
        self.assertFalse(
            decide_robust_from_margins(margins, policy="symmetric_confident")
        )

    def test_dead_zone_is_false(self) -> None:
        margins = [0.1, 0.05, -0.02, 0.08]  # mean ~0.05, |mean| < 0.30
        self.assertFalse(
            decide_robust_from_margins(margins, policy="symmetric_confident")
        )
        # Active default policy should match
        self.assertFalse(decide_robust_from_margins(margins))


class CascadeGateTests(unittest.TestCase):
    def test_reject_overconfident_stage_votes_at_active_cap(self) -> None:
        n = 20
        # 0.55 > 0.50 → reject under symmetric_confident
        over = [True] * 11 + [False] * 9
        # 0.45 <= 0.50 → accept
        ok = [True] * 9 + [False] * 11
        # 0.40 would have been rejected under conservative (0.35) but OK now
        mid = [True] * 8 + [False] * 12
        self.assertTrue(reject_overconfident_votes(over))
        self.assertFalse(reject_overconfident_votes(ok))
        self.assertFalse(reject_overconfident_votes(mid))
        # Explicit conservative cap still rejects mid
        self.assertTrue(reject_overconfident_votes(mid, max_true_rate=0.35))
        self.assertEqual(len(all_false_fallback(n)), n)
        self.assertTrue(all(v is False for v in all_false_fallback(n)))


class CascadeCpuFallbackTests(unittest.TestCase):
    def test_cpu_cascade_returns_all_false(self) -> None:
        # No CUDA / no matching val-sample model_id → stages None → all-False.
        problems = ["p1", "p2", "p3"]
        out = run_cascade("qwen3-8b:low", "low", problems)
        self.assertEqual(out, [False, False, False])
        self.assertTrue(all(type(v) is bool for v in out))


if __name__ == "__main__":
    unittest.main()
