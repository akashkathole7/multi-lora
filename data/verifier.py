#!/usr/bin/env python3
"""Deterministic output verifier for the two-tenant multi-LoRA project.

Pure stdlib. This module must never import a model-client library; it is the
data filter, the training gate, and the final proof, so it has to be
reproducible on any machine with no network and no API key.

Two independent checks per output:

  schema_ok  the text parses as JSON and satisfies the tenant's exact schema
  vocab_ok   the raw text uses at least 2 distinct tenant-own terms and zero
             rival terms

ok = schema_ok and vocab_ok.

Vocabulary matching notes (intended behaviour, not accidents):

  * Matching runs on the raw output text, keys included, not on the parsed
    object. It only runs when json.loads succeeded.
  * Patterns are case-insensitive and use \\b word boundaries.
  * Underscore is a word character. So \\bowner\\b does NOT match the key
    "owner_role", and \\bquarters?\\b does NOT match "duration_quarters".
    Meridian's own schema keys therefore never trip Meridian's forbidden
    \\bowners?\\b rule.
  * A quote is not a word character. So the bare key "gate" DOES match
    \\bgates?\\b and the bare key "sprint" DOES match \\bsprints?\\b. A
    Meridian JSON therefore auto-fails Vantage rules through its "gate" key,
    and a Vantage JSON auto-fails Meridian rules through "sprint"/"ships"/
    "blockers"/"okrs".

Both behaviours are deliberate. The underscore boundary has a dedicated
regression test in data/test_outputs.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TENANTS = ("meridian", "vantage")

SEVERITIES = ("high", "medium", "low")

WP_ID_RE = re.compile(r"^WP-\d{2}$")
GATE_ID_RE = re.compile(r"^G[1-9]\d*$")
WEEKS_RE = re.compile(r"^\d+(-\d+)?$")

MERIDIAN_KEYS = {
    "initiative",
    "work_packages",
    "approval_chain",
    "compliance_notes",
    "risks",
    "timeline_horizon",
}
VANTAGE_KEYS = {
    "initiative",
    "okrs",
    "sprint_plan",
    "blockers",
    "success_metric",
    "timeline_weeks",
}

VOCAB = {
    "meridian": {
        "own": [
            r"\bwork packages?\b",
            r"\bgate reviews?\b",
            r"\bdeviations?\b",
            r"\bsign[- ]?offs?\b",
            r"\bdeliverables?\b",
            r"\bcontrols?\b",
            r"\bnon[- ]?conformances?\b",
        ],
        "forbidden": [
            r"\bsprints?\b",
            r"\bokrs?\b",
            r"\bship(s|ped|ping)?\b",
            r"\bowners?\b",
            r"\bblockers?\b",
            r"\biterat(e|es|ed|ing|ion|ions)\b",
            r"\bweeks?\b",
        ],
    },
    "vantage": {
        "own": [
            r"\bsprints?\b",
            r"\bokrs?\b",
            r"\bowners?\b",
            r"\bship(s|ped|ping)?\b",
            r"\bblockers?\b",
            r"\biterat(e|es|ed|ing|ion|ions)\b",
            r"\bmetrics?\b",
        ],
        "forbidden": [
            r"\bwork packages?\b",
            r"\bgate reviews?\b",
            r"\bgates?\b",
            r"\bsign[- ]?offs?\b",
            r"\bdeviations?\b",
            r"\bnon[- ]?conformances?\b",
            r"\bquarters?\b",
        ],
    },
}

_COMPILED = {
    tenant: {
        bucket: [re.compile(p, re.IGNORECASE) for p in patterns]
        for bucket, patterns in buckets.items()
    }
    for tenant, buckets in VOCAB.items()
}

MIN_OWN_TERMS = 2


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _is_str(value) -> bool:
    """Non-empty string."""
    return isinstance(value, str) and value.strip() != ""


def _is_int(value) -> bool:
    """Real int. Booleans are not ints here."""
    return isinstance(value, int) and not isinstance(value, bool)


def _type_name(value) -> str:
    return type(value).__name__


# --------------------------------------------------------------------------
# schema checks
# --------------------------------------------------------------------------


def _check_top_level(obj, allowed: set, errors: list) -> bool:
    if not isinstance(obj, dict):
        errors.append(f"top level: expected object, got {_type_name(obj)}")
        return False
    missing = sorted(allowed - set(obj))
    for key in missing:
        errors.append(f"missing required key: {key}")
    unknown = sorted(set(obj) - allowed)
    for key in unknown:
        errors.append(f"unknown top-level key: {key}")
    return True


def _check_str_field(obj, key, errors, path=None) -> None:
    path = path or key
    if key not in obj:
        return
    if not _is_str(obj[key]):
        errors.append(f"{path}: expected non-empty string, got {_type_name(obj[key])}")


def _check_list(obj, key, errors, lo=None, hi=None):
    """Return the list if present and a list of the right length, else None."""
    if key not in obj:
        return None
    value = obj[key]
    if not isinstance(value, list):
        errors.append(f"{key}: expected list, got {_type_name(value)}")
        return None
    if lo is not None and len(value) < lo:
        errors.append(f"{key}: expected at least {lo} entries, got {len(value)}")
        return None
    if hi is not None and len(value) > hi:
        errors.append(f"{key}: expected at most {hi} entries, got {len(value)}")
        return None
    return value


def _check_meridian(obj, errors: list) -> None:
    if not _check_top_level(obj, MERIDIAN_KEYS, errors):
        return

    _check_str_field(obj, "initiative", errors)
    _check_str_field(obj, "timeline_horizon", errors)

    packages = _check_list(obj, "work_packages", errors, lo=3, hi=6)
    if packages is not None:
        for i, wp in enumerate(packages):
            path = f"work_packages[{i}]"
            if not isinstance(wp, dict):
                errors.append(f"{path}: expected object, got {_type_name(wp)}")
                continue
            for key in ("id", "title", "owner_role", "duration_quarters", "deliverable"):
                if key not in wp:
                    errors.append(f"{path}: missing key {key}")
            for key in ("title", "owner_role", "deliverable"):
                _check_str_field(wp, key, errors, path=f"{path}.{key}")
            if "id" in wp:
                wp_id = wp["id"]
                expected = f"WP-{i + 1:02d}"
                if not isinstance(wp_id, str) or not WP_ID_RE.match(wp_id):
                    errors.append(f"{path}.id: expected ^WP-\\d{{2}}$, got {wp_id!r}")
                elif wp_id != expected:
                    errors.append(f"{path}.id: expected {expected}, got {wp_id!r}")
            if "duration_quarters" in wp:
                dq = wp["duration_quarters"]
                if not _is_int(dq) or dq <= 0:
                    errors.append(
                        f"{path}.duration_quarters: expected positive int, got {dq!r}"
                    )

    gates = _check_list(obj, "approval_chain", errors, lo=2, hi=4)
    if gates is not None:
        for i, gate in enumerate(gates):
            path = f"approval_chain[{i}]"
            if not isinstance(gate, dict):
                errors.append(f"{path}: expected object, got {_type_name(gate)}")
                continue
            for key in ("gate", "name", "approver_role", "criteria"):
                if key not in gate:
                    errors.append(f"{path}: missing key {key}")
            for key in ("name", "approver_role", "criteria"):
                _check_str_field(gate, key, errors, path=f"{path}.{key}")
            if "gate" in gate:
                gid = gate["gate"]
                expected = f"G{i + 1}"
                if not isinstance(gid, str) or not GATE_ID_RE.match(gid):
                    errors.append(f"{path}.gate: expected ^G[1-9]\\d*$, got {gid!r}")
                elif gid != expected:
                    errors.append(f"{path}.gate: expected {expected}, got {gid!r}")

    notes = _check_list(obj, "compliance_notes", errors, lo=1)
    if notes is not None:
        for i, note in enumerate(notes):
            if not _is_str(note):
                errors.append(
                    f"compliance_notes[{i}]: expected non-empty string, got {note!r}"
                )

    risks = _check_list(obj, "risks", errors, lo=1)
    if risks is not None:
        for i, risk in enumerate(risks):
            path = f"risks[{i}]"
            if not isinstance(risk, dict):
                errors.append(f"{path}: expected object, got {_type_name(risk)}")
                continue
            for key in ("risk", "mitigation", "severity"):
                if key not in risk:
                    errors.append(f"{path}: missing key {key}")
            for key in ("risk", "mitigation"):
                _check_str_field(risk, key, errors, path=f"{path}.{key}")
            if "severity" in risk and risk["severity"] not in SEVERITIES:
                errors.append(
                    f"{path}.severity: expected one of {'|'.join(SEVERITIES)}, "
                    f"got {risk['severity']!r}"
                )


def _check_vantage(obj, errors: list) -> None:
    if not _check_top_level(obj, VANTAGE_KEYS, errors):
        return

    _check_str_field(obj, "initiative", errors)
    _check_str_field(obj, "success_metric", errors)

    okrs = _check_list(obj, "okrs", errors, lo=1, hi=2)
    if okrs is not None:
        for i, okr in enumerate(okrs):
            path = f"okrs[{i}]"
            if not isinstance(okr, dict):
                errors.append(f"{path}: expected object, got {_type_name(okr)}")
                continue
            for key in ("objective", "key_results"):
                if key not in okr:
                    errors.append(f"{path}: missing key {key}")
            _check_str_field(okr, "objective", errors, path=f"{path}.objective")
            if "key_results" in okr:
                krs = okr["key_results"]
                if not isinstance(krs, list):
                    errors.append(
                        f"{path}.key_results: expected list, got {_type_name(krs)}"
                    )
                elif len(krs) < 1:
                    errors.append(f"{path}.key_results: expected at least 1 entry")
                else:
                    for j, kr in enumerate(krs):
                        if not _is_str(kr):
                            errors.append(
                                f"{path}.key_results[{j}]: expected non-empty string, "
                                f"got {kr!r}"
                            )

    sprints = _check_list(obj, "sprint_plan", errors, lo=3, hi=6)
    if sprints is not None:
        for i, sprint in enumerate(sprints):
            path = f"sprint_plan[{i}]"
            if not isinstance(sprint, dict):
                errors.append(f"{path}: expected object, got {_type_name(sprint)}")
                continue
            for key in ("sprint", "weeks", "focus", "owner", "ships"):
                if key not in sprint:
                    errors.append(f"{path}: missing key {key}")
            for key in ("focus", "owner", "ships"):
                _check_str_field(sprint, key, errors, path=f"{path}.{key}")
            if "sprint" in sprint:
                sid = sprint["sprint"]
                if not _is_int(sid):
                    errors.append(f"{path}.sprint: expected int, got {sid!r}")
                elif sid != i + 1:
                    errors.append(f"{path}.sprint: expected {i + 1}, got {sid!r}")
            if "weeks" in sprint:
                weeks = sprint["weeks"]
                if not isinstance(weeks, str) or not WEEKS_RE.match(weeks):
                    errors.append(
                        f"{path}.weeks: expected ^\\d+(-\\d+)?$, got {weeks!r}"
                    )

    if "blockers" not in obj:
        pass  # already reported as missing by _check_top_level
    else:
        blockers = obj["blockers"]
        if not isinstance(blockers, list):
            errors.append(f"blockers: expected list, got {_type_name(blockers)}")
        else:
            for i, blocker in enumerate(blockers):
                if not _is_str(blocker):
                    errors.append(
                        f"blockers[{i}]: expected non-empty string, got {blocker!r}"
                    )

    if "timeline_weeks" in obj:
        tw = obj["timeline_weeks"]
        if not _is_int(tw) or tw <= 0:
            errors.append(f"timeline_weeks: expected positive int, got {tw!r}")


_SCHEMA_CHECKS = {"meridian": _check_meridian, "vantage": _check_vantage}


# --------------------------------------------------------------------------
# vocabulary check
# --------------------------------------------------------------------------


def _check_vocab(text: str, tenant: str, errors: list) -> bool:
    patterns = _COMPILED[tenant]
    own_hits = [p.pattern for p in patterns["own"] if p.search(text)]
    bad_hits = [p.pattern for p in patterns["forbidden"] if p.search(text)]

    ok = True
    if len(own_hits) < MIN_OWN_TERMS:
        ok = False
        errors.append(
            f"vocab: {tenant} needs >= {MIN_OWN_TERMS} distinct own terms, "
            f"found {len(own_hits)} ({', '.join(own_hits) or 'none'})"
        )
    if bad_hits:
        ok = False
        errors.append(
            f"vocab: rival terms present for {tenant}: {', '.join(bad_hits)}"
        )
    return ok


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


def verify(text: str, tenant: str) -> dict:
    """Check one model output against one tenant contract.

    Returns {"ok", "schema_ok", "vocab_ok", "errors"}.
    Raises ValueError on an unknown tenant name.
    """
    if not isinstance(tenant, str) or tenant not in TENANTS:
        raise ValueError(
            f"unknown tenant {tenant!r}: expected one of {', '.join(TENANTS)}"
        )

    errors: list = []

    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        errors.append(f"json parse error: {exc}")
        return {"ok": False, "schema_ok": False, "vocab_ok": False, "errors": errors}

    schema_errors: list = []
    _SCHEMA_CHECKS[tenant](obj, schema_errors)
    schema_ok = not schema_errors
    errors.extend(schema_errors)

    vocab_ok = _check_vocab(text, tenant, errors)

    return {
        "ok": schema_ok and vocab_ok,
        "schema_ok": schema_ok,
        "vocab_ok": vocab_ok,
        "errors": errors,
    }


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

NON_JSON_TEXT = (
    "Here is a plan: first we run a gate review, then we record the deliverable "
    "and the sign-off."
)

# case label, fixture file (or None), tenant, expected schema_ok, vocab_ok, ok
SELF_TEST_CASES = [
    ("meridian_good", "meridian_good.json", "meridian", True, True, True),
    ("meridian_good x-tenant", "meridian_good.json", "vantage", False, False, False),
    ("meridian_broken", "meridian_broken.json", "meridian", False, True, False),
    ("meridian_broken x-tenant", "meridian_broken.json", "vantage", False, False, False),
    ("meridian_crossover", "meridian_crossover.json", "meridian", True, False, False),
    ("vantage_good", "vantage_good.json", "vantage", True, True, True),
    ("vantage_good x-tenant", "vantage_good.json", "meridian", False, False, False),
    ("vantage_broken", "vantage_broken.json", "vantage", False, True, False),
    ("vantage_crossover", "vantage_crossover.json", "vantage", True, False, False),
    ("regression non-json/mer", None, "meridian", False, False, False),
    ("regression non-json/van", None, "vantage", False, False, False),
]


def _fmt(schema_ok, vocab_ok, ok) -> str:
    return f"schema={int(schema_ok)} vocab={int(vocab_ok)} ok={int(ok)}"


def run_self_test(stream=sys.stdout) -> int:
    """Run the fixture table plus the regressions. Return a process exit code."""
    rows = []
    all_match = True

    for label, filename, tenant, exp_schema, exp_vocab, exp_ok in SELF_TEST_CASES:
        text = (
            NON_JSON_TEXT
            if filename is None
            else (FIXTURES_DIR / filename).read_text(encoding="utf-8")
        )
        result = verify(text, tenant)
        expected = _fmt(exp_schema, exp_vocab, exp_ok)
        actual = _fmt(result["schema_ok"], result["vocab_ok"], result["ok"])
        match = expected == actual
        all_match = all_match and match
        rows.append((label, tenant, expected, actual, "MATCH" if match else "MISMATCH"))

    # regression: underscore word boundary. meridian_good carries the keys
    # owner_role and approver_role; \bowner\b must not fire on them.
    good = (FIXTURES_DIR / "meridian_good.json").read_text(encoding="utf-8")
    underscore_ok = (
        "owner_role" in good
        and "approver_role" in good
        and verify(good, "meridian")["vocab_ok"] is True
    )
    all_match = all_match and underscore_ok
    rows.append(
        (
            "regression underscore \\bowner\\b",
            "meridian",
            "vocab=1 on owner_role",
            f"vocab={int(underscore_ok)} on owner_role",
            "MATCH" if underscore_ok else "MISMATCH",
        )
    )

    # regression: unknown tenant name is rejected.
    try:
        verify("{}", "acme")
        tenant_ok = False
        detail = "no error raised"
    except ValueError:
        tenant_ok = True
        detail = "ValueError raised"
    all_match = all_match and tenant_ok
    rows.append(
        (
            "regression unknown tenant",
            "acme",
            "ValueError raised",
            detail,
            "MATCH" if tenant_ok else "MISMATCH",
        )
    )

    widths = [max(len(str(r[i])) for r in rows + [("CASE", "TENANT", "EXPECTED", "ACTUAL", "RESULT")]) for i in range(5)]
    header = ("CASE", "TENANT", "EXPECTED", "ACTUAL", "RESULT")
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))
    print(line, file=stream)
    print("-" * len(line), file=stream)
    for row in rows:
        print(
            "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)), file=stream
        )
    print("-" * len(line), file=stream)
    print(
        f"{len(rows)} cases, "
        f"{sum(1 for r in rows if r[4] == 'MATCH')} match, "
        f"{sum(1 for r in rows if r[4] != 'MATCH')} mismatch",
        file=stream,
    )
    print("SELF-TEST: " + ("PASS" if all_match else "FAIL"), file=stream)
    return 0 if all_match else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="verifier.py",
        description=(
            "Deterministic schema + vocabulary checker for Meridian Industrial "
            "and Vantage Cloud execution plans."
        ),
    )
    parser.add_argument(
        "--tenant",
        choices=TENANTS,
        help="tenant contract to check the files against",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in fixture table and regressions, then exit",
    )
    parser.add_argument("files", nargs="*", help="files containing one JSON output each")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    if not args.files:
        parser.error("no files given (use --self-test to run the built-in table)")
    if not args.tenant:
        parser.error("--tenant is required when checking files")

    failures = 0
    for path in args.files:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"FAIL {path}")
            print(f"       read error: {exc}")
            failures += 1
            continue
        result = verify(text, args.tenant)
        print(f"{'PASS' if result['ok'] else 'FAIL'} {path}")
        if not result["ok"]:
            failures += 1
            for err in result["errors"]:
                print(f"       {err}")

    print(f"{len(args.files) - failures}/{len(args.files)} passed as {args.tenant}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
