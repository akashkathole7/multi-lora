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

## Entry 4 — Stage 2 draft: training and serving stack, verified against docs, unrun

**Date:** 2026-08-25

**What.** Drafted the two halves of the project that have never had code:
training and serving. `train/` gets two routes to the same artifact — a rank-16
LoRA adapter per tenant in Hugging Face PEFT layout. Route A is
`train_lora.py` on the NeMo Framework 2.x API; route B is `train_lora_hf.py` on
HF PEFT + TRL `SFTTrainer`, openly stated as the fallback. `convert_to_hf.py`
wraps NeMo's documented `hf-peft` exporter and adds a `--verify` mode that
checks any adapter directory against the training config. Both routes read the
same `config_<tenant>.yaml`. `serve/azure/` gets a BYOC vLLM container, the
three `az ml` YAMLs, and `deploy.sh` — zero to scoring URI to smoke test to
teardown, behind a mandatory cost guardrail. `serve/spark/launch.sh` is the
on-prem DGX Spark path with the same flags. `ARCHITECTURE.md` explains the whole
thing to a non-specialist reader. README gains Training and Serving sections and
a consolidated **Unverified items (# CHECK list)**.

**Why.** Everything before this entry was measurement without a subject. The
tooling from entry 3 can measure an endpoint, and there was no endpoint; the
verifier can gate training data, and there was no training. This entry writes
the subject down. It is drafted rather than run because nothing here can run
today: there is no GPU on this machine, no Azure CLI, no adapters, and Stage 1
data generation is still blocked on `ANTHROPIC_API_KEY`.

**Problem it solves.** Three, and they are all the same shape — a plausible
guess that fails late and expensively.

First, invented flags. An 8B model on a billing A100 is a bad place to discover
that `--max-lora-rank` is spelled differently or that a YAML key does not exist.
So every vLLM flag, NeMo API call, TRL/PEFT argument and `az ml` schema key was
checked against official documentation, and anything the docs did not settle got
a `# CHECK:` at the exact line plus an entry in README rather than a confident
guess. Two of those checks changed the code: `vllm serve` **rejects** `--model`
and needs the model positionally, and NeMo calls the LoRA rank `dim`, not `r`.

Second, silent divergence between the two training routes. A fallback that
trains on different hyperparameters is not a fallback, it is a second
experiment. Both routes read one config file, and check 6 in the check log
asserts they resolve to the same rank, alpha, dropout, seed, epochs, learning
rate, sequence length and effective global batch.

Third, the budget. The whole project is ₹10,000 and one hour of the target SKU
is about ₹320. `deploy.sh` prints the SKU, the rate, and the projected session
cost, then refuses to create anything billable until the operator types
`yes-bill`; it refuses outright in a non-interactive shell. `teardown` deletes
the deployment and endpoint and then **lists the endpoints to confirm the name
is gone**, because a delete that returned zero is not evidence.

**Expected impact.** Stage 3 runs `deploy.sh`, points the entry-3 tools at the
scoring URI it prints, and gets real numbers. The `# CHECK` list is the agenda
for that session: each item is resolved by observation and recorded here.

**Measured impact.** Syntax, parse and consistency checks only — no model was
called, no GPU was touched, no Azure resource was created. 3 new `.py` files
plus every tracked `.py` compile under `py_compile`. 4 `.sh` files pass
`bash -n`. 5 `.yaml` files parse under PyYAML 6.0.3. 11 schema spot-checks pass
on the Azure YAMLs, including `request_timeout_ms <= 180000` (the documented
maximum), probes carrying no `path`/`port` key (they do not exist on
`ProbeSettings`; the routes live in `inference_config`), and
`MAX_CPU_LORAS >= MAX_LORAS` (vLLM's documented rule). 13 config checks confirm
the hyperparameters match the spec. 18 cross-route checks confirm both trainers
resolve identically. The cost guardrail was exercised three ways: refuses
non-interactively, aborts on a wrong phrase with nothing created, proceeds only
on `yes-bill`. Guardrail grep still PASSes. Existing suite: **20 passed in
11.36s**, unchanged.

Doc verification, all 2026-08-24. Confirmed: vLLM `--enable-lora`,
`--max-lora-rank`, `--max-loras`, `--max-cpu-loras`, `--lora-modules`,
`--served-model-name`, `--gpu-memory-utilization`, `--max-num-seqs`, `/health`,
and image `vllm/vllm-openai:v0.27.1` (Docker Hub, pushed 2026-08-11, upstream
`ENTRYPOINT ["vllm", "serve"]`). Azure ML endpoint/deployment/environment YAML
schemas including `inference_config.{liveness,readiness,scoring}_route.{path,port}`.
NeMo 2.x `llm.peft.LoRA(dim=…)`, the `linear_qkv`/`linear_proj`/`linear_fc1`/
`linear_fc2` module names, `FineTuningDataModule`'s `{"input","output"}` rows,
`MegatronMixedPrecision(precision="bf16-mixed")`, `CosineAnnealingScheduler`,
`llm.export_ckpt(target='hf-peft')`, and that NeMo 2.0 replaced YAML config with
Python. TRL `SFTConfig`/`SFTTrainer` (note: `max_length`, not `max_seq_length`)
and PEFT `LoraConfig`. NIM `NIM_PEFT_SOURCE`. Full list with URLs is in
README.md under "What *was* verified".

**Evidence.** `data/logs/stack_draft_checks.log` (every check above, with its
output), `README.md` "Unverified items (# CHECK list)", `versions.lock`
(regenerated 2026-08-25 with a dated note; PyYAML 6.0.3 is the only addition).

**Not done in this stage.** Nothing here has been executed against real
hardware. No adapter has been trained, no endpoint created, no Azure resource
provisioned, no model called. Every file in `train/` and `serve/` is a draft
whose correctness rests on documentation plus syntax checking. The numbers in
`ARCHITECTURE.md` marked `~` — the ~84 MB adapter size in particular — are
arithmetic over published layer dimensions, not measurements, and the document
says so where they appear.

**Open item for Stage 3 — the routing question.** Azure's docs do not state what
literal path a BYOC container receives when a client POSTs to a scoring URI
ending in `/score`, nor whether a client may address `/v1/chat/completions` on it
directly. `environment.yaml` sets `scoring_route.path: /v1/chat/completions`,
which is the documented BYOC pattern, and `deploy.sh` probes both URLs against
the live endpoint and prints which answered. Record the answer here. It matters
beyond tidiness: `eval/separation.py` and `bench/*.py` build their URL by
appending `/v1/chat/completions`, so they cannot construct a bare `/score` and
would need a local rewrite proxy if `/score` turns out to be the only route.

**Open item for Stage 3 — the NeMo class name.** `config_*.yaml` names
`Llama31NemotronNano8BConfig` and it is a guess; NVIDIA publishes an
`import_ckpt` example only for the Ultra 253B variant. `train_lora.py` refuses to
run on a name it cannot resolve and prints the candidates visible in the
container. Read the real name off `dir(llm)` and record it here. Related and
unresolved: whether `seed_everything` actually reaches Megatron's data sampler
and parallel RNG, which decides whether "seed 1234" means reproducible or merely
reproducible-ish.

**Open item for Stage 3 — the chat template.** Route A flattens chat rows into
NeMo's `{"input","output"}` shape and relies on NeMo to template them; how it
does that is undocumented. If the training-time prompt shape differs from the
serving-time one, the separation number is quietly low with no visible error.
Inspect the tokenised first batch before committing to a 3-epoch run. Route B
does not have this problem — TRL applies the tokenizer's own chat template to
conversational rows — which is the one respect in which the fallback is better
than the primary.

**Note on this entry.** The working session behind it was interrupted partway
and resumed; the partial files on disk were finished rather than rewritten. No
effect on the result, recorded because the commit spans one session in the log
and two in reality.

## Entry 5 — generate.py: a second provider, so generation is paid from Azure credit

**Date:** 2026-08-25

**What.** The `outputs` stage of `data/generate.py` gained a provider switch.
`--provider anthropic` is the default and is byte-for-byte the old path; the
anthropic SDK is still imported lazily and the request is unchanged.
`--provider openai` speaks any OpenAI-compatible chat-completions endpoint over
stdlib `urllib.request` — no new pip package — with `--base-url`,
`--api-key-env` and `--azure-api-version`. With `--azure-api-version` set the
request goes Azure OpenAI style, `POST {base}/openai/deployments/{model}/chat/
completions?api-version=VER` with an `api-key` header; without it, plain OpenAI
style, `POST {base}/chat/completions` with `Authorization: Bearer`. Both
providers send the identical payload: same system prompt, one user message,
`max_tokens` 2000, and no sampling parameters unless `--temperature` is passed.
429 and 5xx are retried (3 attempts, fixed 2s then 4s); anything else, or an
exhausted retry, is written as a row carrying an `error` field and counted in
the stage summary rather than aborting the run. New self-test
`tests/test_generate_openai_provider.py` drives the whole thing against
`tools/mock_openai_server.py`.

**Why.** Money. The project budget is a ₹10,000 Azure free-trial credit and
there is no `ANTHROPIC_API_KEY` in this environment, so the `outputs` stage —
built in entry 1, corrected in entry 2 — has never been run and Stage 1 has been
blocked on a key nobody is going to buy. Azure hosts cheap OpenAI-compatible
models (`gpt-4o-mini`), billed against the same credit that pays for the A100
later. Adding a second transport unblocks generation without adding a
dependency, a key, or a second budget.

**Problem it solves.** A blocked pipeline, and the temptation to unblock it by
lowering the standard of evidence. The generator does not have to be trusted:
every row it produces goes through `data/verifier.py`, which is deterministic,
has no model client and no network, so swapping in a cheaper model changes cost,
latency and rejection rate but cannot change what reaches training. That is why
this is a budget decision and not a scientific one, and it is the property the
new self-test checks — the mock's Meridian plan filed under Vantage is rejected
3/3, so "6 kept" is not the filter waving data through.

Two smaller ones. The key is read from the environment only, exactly as the
entry-3 tools do it, so an Azure key never reaches shell history or a CI log;
the single exception is a loopback `--base-url`, where a missing key is allowed
because that is the mock and it has no auth. And a 429 in the middle of a
200-goal run used to be a lost run; it is now a lost row with a reason attached.

**Expected impact.** Stage 1 becomes runnable the moment an Azure deployment
exists: one `--base-url`, one `--azure-api-version`, one env var. Nothing
downstream changes — filter, package and every number they feed are indifferent
to which provider wrote the row, and `provider` is now recorded on each row so a
mixed file stays traceable.

**Measured impact.** No model was called and no credit was spent; every request
in this entry went to the mock on 127.0.0.1. New self-test: **5/5 passed** under
the plain runner and under pytest. The `outputs` stage against the mock over 3
goals from `data/fixtures/dryrun_goals.jsonl` produced **6 rows (3 goals x 2
tenants), every one non-empty (2261 chars), 0 errors** — run once with a key
present, exercising the `Authorization` header, and once with the env var absent
on loopback, exercising the keyless allowance. Piped through the filter stage,
each tenant's own mock plan passed its own contract: **6 in, 6 kept, 0
rejected**, 3/3 per tenant. The cross direction on the same data: the Meridian
plan filed under `vantage` was rejected 3/3 on `schema+vocab`. A missing key for
a non-loopback endpoint exits 2 and writes nothing; `--provider openai` with no
`--base-url` exits 2. A route the mock does not serve produced 6 rows, 6 errors,
exit 0 — a failing endpoint costs rows, not the run. The retry path was checked
out of band because the real schedule sleeps 2s then 4s: permanent 503 and
permanent 429 each take exactly 3 attempts before recording the error, recovery
on attempt 3 returns the text, and an HTTP 400 is not retried. Full suite:
**25 passed in 11.68s** (was 20). Guardrail grep: PASS — `data/verifier.py` and
`eval/` are still model-free, and the new HTTP code lives in `data/generate.py`,
which the guardrail deliberately does not cover.

Doc verification, 2026-08-25, against learn.microsoft.com. Confirmed: the Azure
OpenAI deployment route `POST https://YOUR_RESOURCE_NAME.openai.azure.com/openai/
deployments/YOUR_DEPLOYMENT_NAME/chat/completions?api-version=YYYY-MM-DD`; that
key auth uses the `api-key` header and Entra ID uses `Authorization: Bearer`;
that `api-version` is a query parameter; and the newer v1 route
(`{endpoint}/openai/v1/`, no `api-version`, deployment name in the body's
`model` field) which the plain-OpenAI mode reaches by `--base-url` alone. Four
`# CHECK:` comments mark what the docs did not settle, listed in README under
"Unverified items".

**Evidence.** `data/logs/generate_provider_selftest.log` (plain runner, full
suite, guardrail, and the out-of-band retry check), `README.md` "Data
generation" and its new "Unverified items" subsection.

**Not done in this stage.** Still no model called and no adapter trained. The
Azure path has never touched a real endpoint: no Azure OpenAI resource exists
yet, no deployment has been created, and the whole provider is proved only
against a mock that answers on loopback with fixed bodies. Nothing here measures
generation quality, rejection rate against a real model, or cost per row —
those are Stage 1 numbers and Stage 1 has not run.

**Open item for Stage 1.** Four unresolved items, all in README's Unverified
table and all resolved by one live call: whether the deployment name equals the
model id (the code reuses `--model` for both and a mismatch is a 404), which
`api-version` string the resource accepts, whether the serverless Foundry
Models route (`/models/...`, `Authorization: Bearer`) is needed instead of
`/openai/deployments/`, and whether `max_tokens` or `max_completion_tokens` is
the right field for the chosen deployment. Record the answers here, and record
the real rejection rate next to them — a cheap generator is only proved cheap
once the fraction of its output the verifier throws away is known.

## Entry 6 — generate.py: reasoning-model request fields; generator is live

**Date:** 2026-08-25

**What.** Two flags on the `outputs` stage: `--max-tokens-param` picks between
`max_tokens` and `max_completion_tokens`, and `--reasoning-effort` sends
`reasoning_effort` when set. Chosen generator: Azure OpenAI `gpt-5-mini`
(GlobalStandard, 100K TPM, South Central US) with `max_completion_tokens` and
`reasoning_effort: minimal`.

**Why.** The subscription's Azure OpenAI quota allows no current non-reasoning
small model: `gpt-4o-mini` has TPM quota but its only version (2024-07-18) is in
Deprecating state and refuses new deployments; the gpt-5.x family has zero TPM
quota except `gpt-5-mini` GlobalStandard (500K default). `gpt-5-mini` is a
reasoning model and rejects `max_tokens`.

**Problem it solves.** Without the field switch every call to the only
deployable generator would 400. `reasoning_effort: minimal` keeps billed
reasoning tokens near zero for a task that needs none.

**Expected impact.** Working generation at ~$0.25/M input, $2.00/M output;
full-dataset projection ~$3 against the $200 credit.

**Measured impact.** Smoke test (1 goal x 2 tenants) against the live
deployment: 2/2 responses, both passed the deterministic verifier, 0 errors.
Existing provider self-tests: 5/5 green after the patch.

**Evidence.** `data/logs/pilot_outputs.log` (smoke lines at top),
`tests/test_generate_openai_provider.py`, README Unverified-items row marked
RESOLVED.

## Entry 7 — Stage 1 pilot accepted; --concurrency for the scale run

**Date:** 2026-08-25

**What.** Ran the 80-goal pilot (160 calls, gpt-5-mini, reasoning_effort minimal).
Accepted the generator prompts unchanged. Added `--concurrency` to the `outputs`
stage (thread pool, order-preserving, default 1 = old behavior) and switched
progress prints to flush.

**Why.** Pilot gate is rejection <15%; measured 7.50% (160 in, 148 kept: meridian
71/80, vantage 77/80; 3 parse — includes 2 API errors — 1 schema, 8 vocab). The
dominant reject is instructive: Meridian outputs writing the standalone word
"owner"/"owners" in prose (rival term) while the schema's own `owner_role` key is
legal — the exact schema-vs-voice leakage the vocabulary check exists to catch.
A prompt change would need a fresh pilot per the one-change-per-iteration rule;
projected clean yield without it (800 x 0.89 = 712 per tenant, need 600) makes
that spend unnecessary. Sequential generation at ~6.5 s/call would put the
1,440-call scale run at ~2.6 h; concurrency 8 brings it to ~20 min inside a
100K-TPM deployment limit.

**Problem it solves.** Scale-run wall time; otherwise none — the pilot passed.

**Expected impact.** Full train/dev outputs (goals 81–800) in ~20 min for ~$1.10;
goals 1–80 reuse the pilot outputs byte-for-byte (goals stage is prefix-stable
for a fixed seed, verified). Goals 801–960 are reserved as the sealed set and
never receive generated outputs.

**Measured impact.** Pilot numbers above, parsed into `reports/iter_01.md`.
Provider self-tests 5/5 after the concurrency patch.

**Evidence.** `data/logs/pilot_outputs.log`, `data/logs/pilot_filter.log`,
`data/generated/pilot/filter_summary.json`, `reports/iter_01.md`.

## Entry 8 — Stage 1 complete: dataset generated, splits frozen, sealed set hashed

**Date:** 2026-08-25

**What.** Generated outputs for goals 81–800 (1,440 calls, concurrency 8, 8 API
errors), merged with the 160 pilot rows (goals 1–80 are byte-identical between
the pilot and the seed-1234 960-goal run). Filtered all 1,600 rows. Split:
train = goals 1–700, dev = goals 701–800 (`eval/dev_goals.jsonl`; their generated
outputs are excluded from training), sealed = goals 801–960
(`eval/sealed_goals.jsonl`; never had outputs generated at all). Packaged
train-range kept rows to chat JSONL. Wrote `eval/SEALED.sha256`.

**Why.** Stage 1 exit criteria: ≥600 verifier-clean pairs per tenant, 160 sealed
goals, frozen splits, rejection <15%.

**Problem it solves.** The dev matrix (Stage 2) and sealed matrix (Stage 4) must
measure generalization, so their goals cannot appear in training data.

**Expected impact.** Training inputs for both adapters at the exact paths
`train/config_*.yaml` expects.

**Measured impact.** Full-set rejection 4.69% (1,600 in, 1,525 kept; meridian
760/800, vantage 765/800; 19 parse — includes 8 API errors — 3 schema, 53
vocab). Train-range clean pairs: meridian 663, vantage 667 (target ≥600).
Every packaged row verified to carry system message "detailed thinking off".
Sealed sha256 d54319eb…47fb15.

**Evidence.** `data/logs/full_outputs.log`, `data/logs/full_filter.log`,
`data/generated/full/filter_summary.json`, `eval/SEALED.sha256`.

## Entry 9 — Training route switch: NeMo (route A) -> HF PEFT + TRL (route B)

**Date:** 2026-08-25

**What.** Abandoned the NeMo-API training route after one attempt and one
diagnostic; adapter training now runs `train/train_lora_hf.py` (route B) inside
the same `nvcr.io/nvidia/nemo:26.08` container, reading the same
`train/config_<tenant>.yaml`. Cluster idle-scale-down raised to 30 min for the
duration of the training session so back-to-back jobs reuse the pulled image.

**Why.** Job `upbeat_map_g80sxygzbh` (nemo-lora-meridian) failed with
`No module named 'nemo'`. Diagnostic job `funny_ball_xpnt0wxsxt` on the same
image shows the 26.08 container's default interpreter (`/opt/venv/bin/python`,
3.12.3) ships `megatron.bridge` but neither `nemo` nor `lightning`: NVIDIA's
26.x NeMo-FW images have moved the training stack to Megatron-Bridge. Source
trees for NeMo sit in `/opt` uninstalled. Rewriting `train_lora.py` against the
Megatron-Bridge API two days before the deadline, with no way to verify the API
offline, is exactly the "NeMo fights the environment" case the plan reserved the
fallback for.

**Problem it solves.** Training was blocked; route B unblocks it with identical
hyperparameters (single shared config; both scripts print the resolved values).

**Expected impact.** A rank-16 HF PEFT adapter per tenant, directly in the
layout vLLM's `--lora-modules` reads — the conversion step (`convert_to_hf.py`)
drops out of the critical path entirely. The trade, stated openly: the delivered
adapters are not trained through the NeMo library itself, though they run in
NVIDIA's NeMo-FW container and serve identically. `train_lora.py` stays in the
repo as the route-A entry point with this entry referenced in its header.

**Measured impact.** Route A attempt: 1 failed job (~25 min billed, ~$1.50,
dominated by the image pull). Diagnostic: ~3 min, ~$0.20. Route B outcome lands
in the next entry.

**Evidence.** Azure ML jobs `upbeat_map_g80sxygzbh` (failed, std_log shows the
resolved config then the import failure), `funny_ball_xpnt0wxsxt` (the
importable-modules table), scratchpad log copy committed at
`train/logs/container_diagnostic_20260825.txt`.
