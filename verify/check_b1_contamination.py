#!/usr/bin/env python3
"""B1: sealed-set contamination check, computed from data/generated/full/goals.jsonl.

Reviewer claims: 960 goals -> 226 distinct strings; 147/160 sealed goals (91.9%)
appear verbatim in the train split (ids 1-700). Dev (701-800): 94/100.
This script computes the same quantities independently from the committed file.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

rows = [json.loads(l) for l in open(ROOT / "data/generated/full/goals.jsonl")]
by_id = {r["goal_id"]: r["goal"] for r in rows}
train = [by_id[i] for i in range(1, 701)]
dev = [by_id[i] for i in range(701, 801)]
sealed = [by_id[i] for i in range(801, 961)]
train_set = set(train)

import data.generate as g
templates = list(g.GOAL_TEMPLATES)
slotless = [t for t in templates if "{" not in t]

out = []
out.append(f"total goals in file:                {len(rows)}")
out.append(f"templates in GOAL_TEMPLATES:        {len(templates)}")
out.append(f"templates with no slots:            {len(slotless)}")
out.append(f"distinct goal strings, all 960:     {len(set(by_id.values()))}")
out.append(f"distinct in train (1-700):          {len(set(train))}")
out.append(f"distinct in dev (701-800):          {len(set(dev))}")
out.append(f"distinct in sealed (801-960):       {len(set(sealed))}")
n_sealed_in_train = sum(1 for s in sealed if s in train_set)
n_dev_in_train = sum(1 for s in dev if s in train_set)
out.append(f"sealed goals verbatim in train:     {n_sealed_in_train}/160 ({100*n_sealed_in_train/160:.1f}%)")
out.append(f"dev goals verbatim in train:        {n_dev_in_train}/100 ({100*n_dev_in_train/100:.1f}%)")

n_temp = len(templates)
out.append("")
out.append(f"template-index mapping: goal i uses shuffled_pool[(i-1) % {n_temp}],")
out.append(f"so goal 801 shares a template with goal {(800 % n_temp) + 1},")
out.append(f"goal 802 with goal {(801 % n_temp) + 1}, etc. (same shuffle, same seed).")
out.append("")
out.append("5 example sealed goals that appear verbatim in the training split:")
shown = 0
for i in range(801, 961):
    if by_id[i] in train_set and shown < 5:
        first_train_id = next(j for j in range(1, 701) if by_id[j] == by_id[i])
        out.append(f"  sealed id {i} == train id {first_train_id}: {by_id[i][:80]}")
        shown += 1

text = "\n".join(out)
print(text)
(Path(__file__).parent / "evidence" / "b1_contamination.txt").write_text(text + "\n")
