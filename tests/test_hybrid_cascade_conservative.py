"""Smoke checks for hybrid-cascade decision policies + CF veto helpers."""

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
from counterfactual import (  # noqa: E402
    agreement_to_bool,
    extract_final_answer,
    fuse_probe_with_cf_veto,
    light_transforms,
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
    def test_active_policy_is_conservative(self) -> None:
        self.assertEqual(DECISION_POLICY, "conservative")
        self.assertTrue(CONSERVATIVE)  # non-legacy
        self.assertTrue(CONSERVATIVE_MODE)
        self.assertAlmostEqual(TRUE_MARGIN_THRESHOLD, 0.5)
        self.assertIsNone(FALSE_MARGIN_THRESHOLD)
        self.assertAlmostEqual(MIN_AGREE_FRAC, 0.75)
        self.assertAlmostEqual(MIN_POSITIVE_MARGIN_FRAC, 0.75)
        self.assertAlmostEqual(MAX_STAGE_TRUE_RATE, 0.35)
        self.assertIn("conservative", POLICY_THRESHOLDS)
        self.assertIn("symmetric_confident", POLICY_THRESHOLDS)
        self.assertAlmostEqual(CASCADE_TRUE_RATE_CAPS["conservative"], 0.35)
        self.assertAlmostEqual(CASCADE_TRUE_RATE_CAPS["symmetric_confident"], 0.50)

    def test_conservative_policy_thresholds_preserved(self) -> None:
        cfg = POLICY_THRESHOLDS["conservative"]
        self.assertAlmostEqual(float(cfg["true_margin_threshold"]), 0.5)
        self.assertIsNone(cfg["false_margin_threshold"])
        self.assertAlmostEqual(float(cfg["min_agree_frac"]), 0.75)

    def test_symmetric_thresholds_still_selectable(self) -> None:
        cfg = POLICY_THRESHOLDS["symmetric_confident"]
        self.assertAlmostEqual(float(cfg["true_margin_threshold"]), 0.30)
        self.assertAlmostEqual(float(cfg["false_margin_threshold"]), -0.30)
        self.assertAlmostEqual(float(cfg["min_agree_frac"]), 0.90)


class ConservativeProbeDecisionTests(unittest.TestCase):
    def test_legacy_mean_nonneg_is_true(self) -> None:
        self.assertTrue(
            decide_robust_from_margins([0.1, -0.05, 0.2], policy="legacy")
        )

    def test_conservative_requires_margin_and_seed_agree(self) -> None:
        # mean 0.6 but only 50% seeds >= 0 → False under conservative
        margins = [1.2, 1.0, -0.1, -0.1]
        self.assertFalse(decide_robust_from_margins(margins, policy="conservative"))
        # mean 0.6 and 100% seeds >= 0.5-ish positive → True
        strong = [0.6, 0.7, 0.55, 0.8]
        self.assertTrue(decide_robust_from_margins(strong, policy="conservative"))
        # Active default matches conservative
        self.assertTrue(decide_robust_from_margins(strong))

    def test_symmetric_true_gate(self) -> None:
        margins = [0.4] * 11 + [-0.05]  # mean≈0.36, pos_frac≈11/12
        self.assertTrue(
            decide_robust_from_margins(margins, policy="symmetric_confident")
        )

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


class CascadeGateTests(unittest.TestCase):
    def test_reject_overconfident_stage_votes_at_active_cap(self) -> None:
        n = 20
        # 0.40 > 0.35 → reject under conservative
        over = [True] * 8 + [False] * 12
        # 0.30 <= 0.35 → accept
        ok = [True] * 6 + [False] * 14
        self.assertTrue(reject_overconfident_votes(over))
        self.assertFalse(reject_overconfident_votes(ok))
        # Explicit symmetric cap allows 0.40
        self.assertFalse(reject_overconfident_votes(over, max_true_rate=0.50))
        self.assertEqual(len(all_false_fallback(n)), n)
        self.assertTrue(all(v is False for v in all_false_fallback(n)))


class CascadeCpuFallbackTests(unittest.TestCase):
    def test_cpu_cascade_returns_all_false(self) -> None:
        # No CUDA / no matching val-sample model_id → stages None → all-False.
        problems = ["p1", "p2", "p3"]
        out = run_cascade("qwen3-8b:low", "low", problems)
        self.assertEqual(out, [False, False, False])
        self.assertTrue(all(type(v) is bool for v in out))


class CounterfactualHelperTests(unittest.TestCase):
    def test_extract_answer_line(self) -> None:
        text = "Reasoning...\nANSWER: 42\n"
        self.assertEqual(extract_final_answer(text), "42")

    def test_extract_boxed(self) -> None:
        text = r"Final is \boxed{7}."
        self.assertEqual(extract_final_answer(text), "7")

    def test_extract_missing_is_none(self) -> None:
        self.assertIsNone(extract_final_answer("no numeric answer here"))

    def test_agreement_requires_all_equal(self) -> None:
        self.assertTrue(agreement_to_bool(["3", "3", "3"]))
        self.assertFalse(agreement_to_bool(["3", "4", "3"]))
        self.assertFalse(agreement_to_bool(["3", None, "3"]))

    def test_light_transforms_stable_length(self) -> None:
        variants = light_transforms("Let n = 3. Find n+1.", k=2)
        self.assertEqual(len(variants), 2)
        self.assertNotEqual(variants[0], "")

    def test_fuse_probe_cf_veto(self) -> None:
        probe = [True, True, False, True]
        cf = [True, False, True, True]
        fused = fuse_probe_with_cf_veto(probe, cf)
        self.assertEqual(fused, [True, False, False, True])
        self.assertTrue(all(type(v) is bool for v in fused))


if __name__ == "__main__":
    unittest.main()
