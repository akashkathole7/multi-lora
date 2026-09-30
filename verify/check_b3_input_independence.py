#!/usr/bin/env python3
"""B3: is the metric independent of the input goal?

(a) API shape: verify(text, tenant) takes no goal argument at all.
(b) The reviewer's cake test: a schema-valid Meridian plan about baking a cake.
(c) How many of MIN_OWN_TERMS are satisfiable by schema keys alone, per tenant.
"""
import inspect, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import data.verifier as V
from data.verifier import verify

out = []
sig = str(inspect.signature(verify))
out.append(f"(a) verify signature: verify{sig}")
out.append("    The input goal is not a parameter; no comparison to the goal is possible.")
out.append("")

cake = {
    "initiative": "Bake a three-layer chocolate cake for the office party",
    "work_packages": [
        {"id": "WP-01", "title": "Buy flour, eggs and chocolate", "owner_role": "Baker",
         "duration_quarters": 1, "deliverable": "Ingredients on the counter"},
        {"id": "WP-02", "title": "Mix the batter and bake the layers", "owner_role": "Baker",
         "duration_quarters": 1, "deliverable": "Three baked sponge layers"},
        {"id": "WP-03", "title": "Frost and decorate", "owner_role": "Decorator",
         "duration_quarters": 1, "deliverable": "Finished cake, gate review of icing quality passed"}],
    "approval_chain": [
        {"gate": "G1", "name": "Taste gate review", "approver_role": "Head Chef",
         "criteria": "Batter tastes right, sign-off given"},
        {"gate": "G2", "name": "Final sign-off", "approver_role": "Party Host",
         "criteria": "Cake looks presentable"}],
    "compliance_notes": ["Kitchen hygiene controls apply while baking."],
    "risks": [{"risk": "Cake collapses", "mitigation": "Follow the recipe exactly", "severity": "medium"}],
    "timeline_horizon": "one afternoon",
}
r = verify(json.dumps(cake), "meridian")
out.append(f"(b) Cake plan vs MERIDIAN contract -> ok={r['ok']}  schema_ok={r['schema_ok']}  vocab_ok={r['vocab_ok']}  errors={r['errors']}")
out.append("    The plan is about baking a cake; no goal is consulted.")
out.append("")

out.append(f"(c) MIN_OWN_TERMS = {V.MIN_OWN_TERMS}. Own-term hits from an empty-prose skeleton")
out.append("    (all string values single dots, keys only):")
import re
def key_only(obj_keys_json, tenant):
    hits = set()
    for pat in V.VOCAB[tenant]["own"]:
        if re.search(pat, obj_keys_json, re.IGNORECASE):
            hits.add(pat)
    return hits
mer_skel = json.dumps({"initiative":".","work_packages":[{"id":"WP-01","title":".","owner_role":".","duration_quarters":1,"deliverable":"."}],"approval_chain":[{"gate":"G1","name":".","approver_role":".","criteria":"."}],"compliance_notes":["."],"risks":[{"risk":".","mitigation":".","severity":"low"}],"timeline_horizon":"."})
van_skel = json.dumps({"initiative":".","okrs":[{"objective":".","key_results":["."]}],"sprint_plan":[{"sprint":1,"weeks":"1-2","focus":".","owner":".","ships":"."}],"blockers":[],"success_metric":".","timeline_weeks":12})
mh = key_only(mer_skel, "meridian"); vh = key_only(van_skel, "vantage")
out.append(f"    meridian: {len(mh)} own-term pattern(s) from keys alone: {sorted(mh)}")
out.append(f"    vantage:  {len(vh)} own-term pattern(s) from keys alone: {sorted(vh)}")
out.append(f"    -> meridian needs >= {V.MIN_OWN_TERMS - len(mh)} more from prose; vantage needs >= {max(0, V.MIN_OWN_TERMS - len(vh))} more from prose.")

text = "\n".join(out)
print(text)
(Path(__file__).parent / "evidence" / "b3_input_independence.txt").write_text(text + "\n")
