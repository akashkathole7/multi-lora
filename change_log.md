# Change log

## Entry 1 — Stage 0: deterministic verifier, fixtures, pipeline skeleton

**Date:** 2026-08-24

**What.** Built `data/verifier.py`, a pure-stdlib checker that validates a JSON execution
plan against one of two tenant contracts (Meridian Industrial, Vantage Cloud) on two
independent axes: schema and vocabulary. Added six fixture cases with fixed expected
verdicts and a test file that asserts every one of them, including both cross-tenant
directions and the underscore word-boundary regression. Added `data/generate.py`, a
four-stage pipeline (goals, outputs, filter, package). Added a grep guardrail that fails
if a model-client import ever appears in the verifier or in `eval/`.

**Why.** The whole project rests on being able to say, deterministically, whether an
output belongs to tenant A or tenant B. That judgement is used three times: to filter
generated training data, to gate what reaches training, and to measure the adapters at
the end. If it depends on a model call, it is not evidence. Building it first, before any
data exists, means the data is generated against a contract that already exists rather
than a contract fitted to whatever the model happened to produce.

**Problem it solves.** A schema check on its own accepts a Meridian-shaped plan written
in Vantage voice. That output is structurally valid and stylistically wrong, and it would
teach the adapter the wrong house style with no visible error. Fixture 3
(`meridian_crossover.json`) is exactly that case, held as a permanent regression: valid
schema, `schema_ok=True`, and rejected on vocabulary.

**Expected impact.** A filter that rejects both malformed and off-voice outputs before
they reach training, plus a fixed measurement for later stages that does not drift
between runs or machines.

**Measured impact.** Verifier self-test: 13/13 expected verdicts matched, exit 0.
`data/test_outputs.py`: 12/12 passed, under both the plain runner and pytest 9.1.1.
Dry run over 20 hand-authored outputs: 20 in, 17 kept, 3 rejected, 15.00% rejection rate.
All three rejects were the planted ones and each was caught for its planted reason —
a non-consecutive gate id (`G1` then `G3`), a string `timeline_weeks` (`"10"`), and a
Vantage plan carrying Meridian vocabulary. No clean row was rejected. Guardrail grep:
PASS on the shipped tree, and confirmed to exit 1 against a planted `import anthropic`
under `eval/`.

**Evidence.** `data/logs/verifier_selftest.log`, `data/logs/dryrun.log`,
`data/logs/guardrail_grep.log`, `data/generated/dryrun/filter_summary.json`.

**Not done in this stage.** No model has been called. The `outputs` stage is built but
unrun: `ANTHROPIC_API_KEY` is absent, and the stage exits 2 with a one-line message
naming the missing key. Stage 1 is blocked on that key.

**Open item for Stage 1.** `data/generate.py` sends `temperature=1.0` as specified. The
bundled claude-api reference states that sampling parameters are rejected with a 400 on
`claude-sonnet-5` and the rest of the current model family. This is flagged with a
`# CHECK:` comment at the constant and needs one live call to confirm before a full
generation run.
