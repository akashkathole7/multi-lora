#!/usr/bin/env python3
"""B2: is the off-diagonal zero guaranteed by construction?

Reviewer claims: Meridian and Vantage required top-level key sets are disjoint,
and the verifier rejects both missing and unknown keys, so any output passing
one schema必fails the other. Off-diagonal cells therefore carry no information.
This script builds minimal schema-valid objects with deliberately NEUTRAL prose
and runs each through BOTH contracts.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data.verifier import verify

meridian_neutral = {
    "initiative": "Make the numbers better next period.",
    "work_packages": [
        {"id": f"WP-0{i}", "title": f"Part {i}", "owner_role": "Team Lead",
         "duration_quarters": 1, "deliverable": f"Result {i}"} for i in (1, 2, 3)],
    "approval_chain": [
        {"gate": "G1", "name": "First check", "approver_role": "Manager", "criteria": "Looks fine"},
        {"gate": "G2", "name": "Second check", "approver_role": "Director", "criteria": "Also fine"}],
    "compliance_notes": ["Standard policy applies."],
    "risks": [{"risk": "It slips", "mitigation": "Start early", "severity": "low"}],
    "timeline_horizon": "next planning period",
}
vantage_neutral = {
    "initiative": "Make the numbers better next period.",
    "okrs": [{"objective": "Improve outcomes", "key_results": ["Numbers go up"]}],
    "sprint_plan": [
        {"sprint": i, "weeks": f"{2*i-1}-{2*i}", "focus": f"Part {i}",
         "owner": "Team Lead", "ships": f"Result {i}"} for i in (1, 2, 3)],
    "blockers": [],
    "success_metric": "Numbers improved",
    "timeline_weeks": 12,
}

out = []
for label, obj in [("Meridian-schema JSON, neutral prose", meridian_neutral),
                   ("Vantage-schema JSON, neutral prose", vantage_neutral)]:
    out.append(label)
    for tenant in ("meridian", "vantage"):
        r = verify(json.dumps(obj), tenant)
        out.append(f"  vs {tenant.upper():8} contract: ok={r['ok']}  schema_ok={r['schema_ok']}  vocab_ok={r['vocab_ok']}")
        for e in r["errors"][:6]:
            out.append(f"    - {e[:110]}")
    out.append("")

out.append("Key-set analysis:")
mk = {"initiative","work_packages","approval_chain","compliance_notes","risks","timeline_horizon"}
vk = {"initiative","okrs","sprint_plan","blockers","success_metric","timeline_weeks"}
out.append(f"  meridian required keys: {sorted(mk)}")
out.append(f"  vantage  required keys: {sorted(vk)}")
out.append(f"  intersection: {sorted(mk & vk)}")
out.append(f"  disjoint apart from 'initiative': {mk & vk == {'initiative'}}")
out.append("  verifier rejects unknown top-level keys -> a JSON with meridian's keys")
out.append("  always carries keys unknown to vantage and lacks keys vantage requires.")
out.append("  Conclusion test: can ANY single JSON object pass both schemas? It would")
out.append("  need both key sets simultaneously; each side rejects the other's keys as")
out.append("  unknown -> structurally impossible.")

text = "\n".join(out)
print(text)
(Path(__file__).parent / "evidence" / "b2_tautology.txt").write_text(text + "\n")
