#!/usr/bin/env python3
"""B7: does the headlined tokens/sec figure understate decode overhead vs ITL?

Reads bench/logs/matrix_summary_endpoint_session1.json (no recomputation from
raw; the summary is the published artifact under test) and prints itl_p50_s per
arm, the interleaved/base ratio, and the headlined tokens/sec delta beside it.
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
d = json.load(open(ROOT / "bench/logs/matrix_summary_endpoint_session1.json"))
rows = d["rows"]

out = ["itl_p50_s per arm:"]
for arm, r in rows.items():
    out.append(f"  {arm:22} itl_p50_s={r['itl_p50_s']:.6f}  itl_p95_s={r['itl_p95_s']:.6f}  tokens_per_sec_p50={r['tokens_per_sec_p50']}")

b = rows["base-only"]; i = rows["two-lora-interleaved"]
itl_ratio = i["itl_p50_s"] / b["itl_p50_s"]
tps_delta = (i["tokens_per_sec_p50"] / b["tokens_per_sec_p50"] - 1) * 100
out.append("")
out.append(f"ITL p50 interleaved / base = {i['itl_p50_s']:.6f} / {b['itl_p50_s']:.6f} = {itl_ratio:.4f}  ->  +{(itl_ratio-1)*100:.1f}% slower decode")
out.append(f"Headlined metric: tokens_per_sec_p50 {b['tokens_per_sec_p50']} -> {i['tokens_per_sec_p50']}  =  {tps_delta:.2f}%")
out.append("")
out.append("Definition check (from the summary file itself):")
out.append(f"  itl_definition: {d.get('itl_definition','(absent)')[:120]}")
out.append("  tokens_per_sec per request = output_tokens / e2e_s -> includes TTFT in the")
out.append("  denominator, so longer generations amortize prefill and shrink the delta.")
out.append("")
out.append("Reviewer's numbers: 0.011319 vs 0.013628 -> +20.4%. Mine above.")
out.append("Caveat carried from the repo's own RESULTS (b): the arms decode different")
out.append("output lengths (378 vs 512 mean tokens), so even the ITL delta is not fully")
out.append("attributable to LoRA without a matched-length rerun.")

text = "\n".join(out)
print(text)
(Path(__file__).parent / "evidence" / "b7_itl.txt").write_text(text + "\n")
