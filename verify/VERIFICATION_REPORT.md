# Verification report: external review findings B1, B2, B3, B7

Independent reproduction of four findings from the external critical review,
using only this repository's committed code, data, and logs. No GPU, no cloud,
no network. Every number below was printed by a script in `verify/`, with raw
output in `verify/evidence/`. Phase 1 was read-only: no existing repository
file was modified.

Path check: every file named in the verification brief exists at the stated
path. No substitutions were needed.

---

## B1 — Sealed-set contamination: **CONFIRMED** (one detail corrected)

Script: `verify/check_b1_contamination.py` → `verify/evidence/b1_contamination.txt`

| Quantity | Reviewer | This verification |
| --- | ---: | ---: |
| Distinct goal strings in all 960 | 226 | **226** |
| Sealed goals verbatim in train | 147/160 (91.9%) | **147/160 (91.9%)** |
| Dev goals verbatim in train | 94/100 (94%) | **94/100 (94.0%)** |
| Goal templates | 25, 6 slotless | **40, 13 slotless** |

Mechanism confirmed in source (`data/generate.py`, `stage_goals`): template =
`pool[(i-1) % 40]` over one seed-shuffled pool, so goal 801 reuses goal 1's
template, 802 reuses 2's, and slotless templates render byte-identical text.
Five verbatim examples are listed in the evidence file (e.g. sealed id 801 ==
train id 1, word for word).

The reviewer's template count (25/6) is wrong — it is 40/13 — but this changes
nothing: their three headline numbers reproduce exactly.

**Meaning for the headline claim:** `RESULTS.md`'s "160 goals held out of
training" is true of the *output* pairs only. 91.9% of sealed *inputs* were
seen, verbatim, during training. The sealed matrix measures recall on seen
prompts far more than generalization.

---

## B2 — Tautological off-diagonal: **CONFIRMED** (and slightly sharpened)

Script: `verify/check_b2_tautology.py` → `verify/evidence/b2_tautology.txt`

The two contracts' required top-level key sets share only `initiative`, and
`data/verifier.py` rejects both missing and unknown keys. Verified by
construction: a schema-valid Meridian object fails Vantage with 5 missing-key
plus unknown-key errors, and vice versa. No single JSON object can pass both
schemas, so `P(passes rival | passes own) = 0` structurally. The 0/160
off-diagonal cells were guaranteed before any adapter was trained.

Two nuances my run adds:
- A neutral-prose Vantage-schema object passes its own full contract
  (schema + vocabulary) with **no tenant prose at all** — its schema keys alone
  supply 5 own-vocabulary matches.
- A neutral-prose Meridian object does *not* pass its own contract (keys give
  only 1 of 2 required own terms), so Meridian's vocab gate does add a small
  real check; the reviewer's example of a passing neutral Meridian object did
  not reproduce as printed, though their structural conclusion stands fully.

**Meaning for the headline claim:** the off-diagonal zeros carry no information
about the adapters; the matrix's informative content is the diagonal
("emits own valid schema") plus a weak base row.

---

## B3 — Metric is input-independent: **CONFIRMED**

Script: `verify/check_b3_input_independence.py` → `verify/evidence/b3_input_independence.txt`

- API shape: `verify(text, tenant)` — the input goal is not a parameter, so
  goal-conditioning is impossible by design, not merely unimplemented.
- The reviewer's cake test reproduces: a schema-valid Meridian plan for baking
  a chocolate cake returns `ok=True, errors=[]`.
- Key-only vocabulary floor: with all prose reduced to ".", schema keys alone
  supply **1 of 2** required own terms for Meridian and **5 of 2** for Vantage
  — i.e. Vantage's vocabulary requirement is fully satisfied by its own schema.

**Meaning for the headline claim:** a degenerate model emitting one memorized
valid plan for every input would score 160/160. The metric cannot detect
whether the model read the goal.

---

## B7 — Understated decode overhead: **CONFIRMED**

Script: `verify/check_b7_itl.py` → `verify/evidence/b7_itl.txt`

| Metric | base-only | two-lora-interleaved | delta |
| --- | ---: | ---: | ---: |
| itl_p50_s | 0.011319 | 0.013628 | **+20.4%** |
| tokens_per_sec_p50 (headlined) | 69.451 | 62.495 | −10.02% |

Reviewer's figures (0.011319, 0.013628, +20.4%) match mine exactly. Both
metrics were published in the summary JSON and in RESULTS' full table — nothing
was hidden — but the narrative headlined the metric that includes TTFT in its
denominator, which halves the apparent decode tax. The clean decode metric is
ITL: **+20.4%**, with the repo's own caveat that unequal output lengths
(378 vs 512 mean tokens) mean even this is not fully attributable to LoRA
without a matched-length rerun.

---

## Verdict summary

| Finding | Verdict | Reviewer's number | Mine |
| --- | --- | --- | --- |
| B1 contamination | CONFIRMED | 147/160 (91.9%) | 147/160 (91.9%) — templates are 40, not 25 |
| B2 tautology | CONFIRMED | structural 0 | structural 0 (Vantage vocab also key-satisfied) |
| B3 input-independence | CONFIRMED | cake passes | cake passes; verify() has no goal argument |
| B7 ITL vs tok/s | CONFIRMED | +20.4% vs −10.0% | +20.4% vs −10.02% |

Note on scope: the source review document was truncated in transmission — its
section 4 ("Reproducing the findings") was never seen, and material after P2.2
is missing. Findings B1–B12 headers were visible; only B1, B2, B3, B7 were in
scope for this verification.

What survives untouched: provenance discipline, the pre-registration mechanism
itself, the deterministic verifier's reproducibility, the honest-mistakes
change log, cost control, and the swap/TTFT measurements as *system* numbers.
What does not survive: the generalization claim, the off-diagonal as evidence,
and task competence beyond schema emission.

---

## Phase 2 — proposed remediation, NOT implemented (priority order)

1. **Template-held-out + paraphrase + out-of-domain sealed set, with a
   contamination report per eval run** — fixes B1. **FREE** (scripting) to
   build; scoring it against the adapters **NEEDS GPU** (~1 endpoint session).
2. **Goal-conditional scoring** (grounding/specificity gates behind the
   existing deterministic pass) with the cake as a permanent must-fail
   fixture — fixes B3. **FREE** to build; re-scoring **NEEDS GPU**.
3. **Report schema-pass and vocab-pass separately; state off-diagonal
   attainability in the table; add a shared-schema style arm** — fixes B2.
   Separation-of-columns and the attainability statement are **FREE**
   (honest restatement is valid remediation); the shared-schema arm
   **NEEDS RETRAINING** (new data + two adapters) — and note B2 cannot be
   fully "fixed" without redesigning the tenant schemas; restating the claim
   honestly is the free remediation.
4. **Promote ITL to the headline cost metric and rerun the benchmark at
   matched output length** (`min_tokens = max_tokens`) — fixes B7 and the
   length confound. Restating headline: **FREE**. Matched-length rerun:
   **NEEDS GPU** (~30 min endpoint time).
5. **Prompted-base and constrained-decoding (guided-JSON) baseline arms** —
   addresses the necessity question (reviewer's B5). **NEEDS GPU**
   (one endpoint session covers all arms).
6. Vocabulary check on string values only (or raise MIN_OWN_TERMS and exclude
   keys) so schema keys stop self-satisfying the style gate — **FREE**, but
   changes historical comparability; must be reported as a metric version bump.

Phase 1 complete. No repository files were modified.
Awaiting approval for Phase 2.
