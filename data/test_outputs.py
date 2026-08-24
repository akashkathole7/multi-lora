#!/usr/bin/env python3
"""Fixture table for the deterministic verifier.

Runs under pytest, and also as `python data/test_outputs.py` with no pytest
installed. Every assertion here is the contract Stage 1 data generation will be
filtered against, so a change in the verifier that moves any of these verdicts
should be a deliberate, logged decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.verifier import verify  # noqa: E402

FIXTURES = ROOT / "data" / "fixtures"

NON_JSON_TEXT = (
    "Here is a plan: first we run a gate review, then we record the deliverable "
    "and the sign-off."
)


def load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 1. meridian_good: passes Meridian, fails Vantage
# --------------------------------------------------------------------------


def test_meridian_good_passes_meridian():
    result = verify(load("meridian_good.json"), "meridian")
    assert result["schema_ok"] is True, result["errors"]
    assert result["vocab_ok"] is True, result["errors"]
    assert result["ok"] is True, result["errors"]


def test_meridian_good_fails_vantage():
    result = verify(load("meridian_good.json"), "vantage")
    assert result["ok"] is False
    assert result["schema_ok"] is False
    assert result["vocab_ok"] is False


# --------------------------------------------------------------------------
# 2. meridian_broken: bad WP id and missing compliance_notes
# --------------------------------------------------------------------------


def test_meridian_broken_fails_meridian_on_schema():
    result = verify(load("meridian_broken.json"), "meridian")
    assert result["ok"] is False
    assert result["schema_ok"] is False
    joined = " ".join(result["errors"])
    assert "WP-1" in joined
    assert "compliance_notes" in joined


def test_meridian_broken_fails_vantage():
    result = verify(load("meridian_broken.json"), "vantage")
    assert result["ok"] is False


# --------------------------------------------------------------------------
# 3. meridian_crossover: schema alone is not enough
# --------------------------------------------------------------------------


def test_meridian_crossover_schema_ok_but_vocab_fails():
    result = verify(load("meridian_crossover.json"), "meridian")
    assert result["schema_ok"] is True, result["errors"]
    assert result["vocab_ok"] is False
    assert result["ok"] is False
    assert any("rival terms" in e for e in result["errors"])


# --------------------------------------------------------------------------
# 4. vantage_good: passes Vantage, fails Meridian
# --------------------------------------------------------------------------


def test_vantage_good_passes_vantage():
    result = verify(load("vantage_good.json"), "vantage")
    assert result["schema_ok"] is True, result["errors"]
    assert result["vocab_ok"] is True, result["errors"]
    assert result["ok"] is True, result["errors"]


def test_vantage_good_fails_meridian():
    result = verify(load("vantage_good.json"), "meridian")
    assert result["ok"] is False
    assert result["schema_ok"] is False
    assert result["vocab_ok"] is False


# --------------------------------------------------------------------------
# 5. vantage_broken: string sprint id and string timeline_weeks
# --------------------------------------------------------------------------


def test_vantage_broken_fails_vantage_on_schema():
    result = verify(load("vantage_broken.json"), "vantage")
    assert result["ok"] is False
    assert result["schema_ok"] is False
    joined = " ".join(result["errors"])
    assert "sprint_plan[2].sprint" in joined
    assert "timeline_weeks" in joined


# --------------------------------------------------------------------------
# 6. vantage_crossover: schema ok, Meridian voice
# --------------------------------------------------------------------------


def test_vantage_crossover_schema_ok_but_vocab_fails():
    result = verify(load("vantage_crossover.json"), "vantage")
    assert result["schema_ok"] is True, result["errors"]
    assert result["vocab_ok"] is False
    assert result["ok"] is False
    assert any("rival terms" in e for e in result["errors"])


# --------------------------------------------------------------------------
# Regressions
# --------------------------------------------------------------------------


def test_underscore_word_boundary_does_not_trip_forbidden_owner():
    """owner_role and approver_role must not match Meridian-forbidden \\bowner\\b.

    Underscore is a word character, so there is no boundary after "owner" in
    "owner_role". This is the behaviour the vocabulary check depends on.
    """
    text = load("meridian_good.json")
    assert "owner_role" in text
    assert "approver_role" in text
    result = verify(text, "meridian")
    assert result["vocab_ok"] is True, result["errors"]


def test_non_json_fails_both_tenants_with_parse_error():
    for tenant in ("meridian", "vantage"):
        result = verify(NON_JSON_TEXT, tenant)
        assert result["ok"] is False
        assert result["schema_ok"] is False
        assert result["vocab_ok"] is False
        assert any("json parse error" in e for e in result["errors"])


def test_unknown_tenant_rejected():
    for bad in ("acme", "Meridian", "", None):
        try:
            verify("{}", bad)
        except ValueError as exc:
            assert "unknown tenant" in str(exc)
        else:
            raise AssertionError(f"tenant {bad!r} should have been rejected")


# --------------------------------------------------------------------------
# plain-python runner
# --------------------------------------------------------------------------


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - test runner
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(_main())
