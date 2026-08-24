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
generation run. Resolved by entry 2.

## Entry 2 — generate.py: stop sending temperature by default

**Date:** 2026-08-24

**What.** Removed the hardcoded `temperature=1.0` from the `outputs` stage API call.
Temperature is now sent only when `--temperature` is passed explicitly; the default
request carries no sampling parameters. Closes the `# CHECK:` from entry 1.

**Why.** The current claude-api reference states the Claude 5 family rejects explicit
sampling parameters (`temperature`/`top_p`/`top_k`) with a 400. The default generator is
`claude-sonnet-5`, so the previous default would have failed every call in the first real
generation run.

**Problem it solves.** Stage 1 would have opened with 100% API errors and zero data.

**Expected impact.** None on outputs (1.0 was the API default anyway); removes a
guaranteed 400 against Claude 5 models. Older models that accept temperature can still
get it via the new flag.

**Measured impact.** `--help` exits 0; `filter` stage re-run on the dry-run fixtures
reproduces 17 kept / 3 rejected with identical reject reasons. No live `outputs` call yet
(still blocked on `ANTHROPIC_API_KEY`), so the 400-avoidance itself is confirmed only
against the API reference, not a live response.

**Evidence.** `data/generate.py` (diff in this commit); claude-api reference note in
entry 1.

## Entry 3 — Pre-Stage-1 tooling: measurement stack, self-testable offline

**Date:** 2026-08-24

*(Naming note: the commit for this entry, `e266b46`, says "Stage 1" in its title. That
label is wrong — Stage 1 in the project plan is the data pilot, which is still blocked on
an API key. This entry is tooling built ahead of need while blocked. The commit message is
left as-is rather than rewriting history.)*

**What.** Built the measurement side of the project before the thing it measures.
`tools/mock_openai_server.py` is a stdlib mock of a vLLM OpenAI-compatible server with
per-request LoRA selection, configurable TTFT and inter-token delay, and a configurable
extra delay on the first request to each non-base model that stands in for a cold adapter
load. Against it run five tools: `eval/separation.py` (per-arm confusion matrix over both
tenant contracts, with sealed-set enforcement), `bench/swap_time.py` (cold vs warm adapter
swap, isolated by controlling request order), `bench/run_matrix.py` (four-arm serving
benchmark: arm definitions, a 17-field metric row, and a fallback stdlib load driver),
`bench/economics.py` (memory and cost per tenant per month, arithmetic only) and
`scripts/make_report.py` (raw logs to `reports/iter_NN.md`, citing the source file for
every number). `tests/test_tooling.py` exercises all five end to end.

**Why.** The headline claim of this project is a number: two adapters served from one GPU
give tenant-correct output with a stated latency and cost. A number is only evidence if
the instrument that produced it was verifiable before the result existed. Building the
harness after the adapters exist means the first time it runs is also the first time
anyone would notice it is wrong, and by then there is a result to be attached to. Building
it against a mock whose right answer is fixed by construction means the harness is either
right or visibly broken, today, with nothing riding on the outcome.

**Problem it solves.** Three specific failure modes. First, a benchmark harness that
reports a plausible number because of a bug in the harness — the mock's fixed responses
mean the correct confusion matrix (100/0, 0/100, 0/0) and the correct cold-swap value
(the injected delay) are known in advance, so a wrong harness cannot look right. Second,
summaries computed from memory and reported without a trace — every tool now writes raw
per-request JSONL first and computes its summary by reading that file back off disk, and
every summary carries the path of the log it came from. Third, a headline number tuned
against the set it is measured on — `--sealed` refuses to run unless the goals file
matches `eval/SEALED.sha256`, and refuses a second sealed run without `--allow-rerun`.

**Expected impact.** At Stage 3 the tools point at the Azure scoring URI by changing
`--endpoint` and nothing else. Adapter names are `--served-names`, so a rename is a flag,
not a code edit.

**Measured impact.** `tests/test_tooling.py`: 8/8 passed under the plain runner and under
pytest 9.1.1 (`8 passed in 11.09s`). Separation against the mock: meridian arm 100.0%
(6/6) on Meridian rules and 0.0% (0/6) on Vantage, vantage arm the exact reverse, base arm
0.0% (0/6) on both, 0 errors, raw log holding 18 rows = 3 arms x 6 goals. Swap time
against a 400ms injected cold penalty: cold TTFT 0.4815s, warm-adapter p50 0.0816s,
baseline p50 0.0815s, `cold_swap_estimate` 0.3999s (0.12ms from the injected 0.400s) and
`warm_swap_estimate` 0.0001s. `run_matrix.py` on arms base-only and two-lora-interleaved,
6 requests each at concurrency 2: all 17 metric fields populated on both rows, 0 errors,
12 raw rows. Economics at N=20: 320.00 GB for 20 full fine-tunes against 17.60 GB for one
base plus 20 adapters, 18.18x. Guardrail grep: PASS, `eval/` and `data/verifier.py` still
free of model-client imports. All six scripts answer `--help` with exit 0.

**Evidence.** `data/logs/tooling_selftest.log`, `data/logs/guardrail_grep.log`,
`reports/iter_00.md`, and the `_selftest` logs under `eval/logs/` and `bench/logs/`.

**Not done in this stage.** No model has been called and no adapter has been trained.
Every number in `reports/iter_00.md` and every `_selftest` log came from the mock server
and is synthetic by construction; the report says so on its first line. No sealed goal set
exists yet — `eval/SEALED.sha256` is deliberately absent and is written once, at Stage 3,
against goals held out of training. `bench/run_matrix.py` is a stub in the sense that
matters: the arms and the metric row are final, the load driver is a fallback.

**Open item for Stage 2.** `bench/economics.py` defaults `ADAPTER_GB` to 0.08, an estimate
for a rank-16 LoRA on an 8B model across 7 target modules, not a measurement. Measure the
on-disk adapter size after training and re-run with `--adapter-gb <measured>`; the whole
table moves with it. The label in the generated markdown says ESTIMATE until it does.

**Open item for Stage 3.** `bench/run_matrix.py` drives load with a stdlib fallback. The
preferred generators are NVIDIA genai-perf and vLLM's `benchmarks/benchmark_serving.py`,
both of which report the same metric names and one of which produces a true per-token-pair
ITL distribution that the fallback cannot. Four `# CHECK:` comments in the file mark where
the swap goes. The arms and the metric row do not change when it happens.
