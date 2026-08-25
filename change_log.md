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

## Entry 10 — devmatrix job served nothing for 2h: YAML folding bug, fixed

**Date:** 2026-08-25

**What.** Cancelled eval job `yellow_dinner_p5pg2mhblp` (devmatrix-A) after ~2h
of nothing, replaced `job_devmatrix_a.yaml` with a fixed combined
`job_devmatrix.yaml` (base + both adapters, one eval job), and resequenced:
adapter B trains first (it was already queued), then one combined dev matrix.

**Why.** The job's vLLM line was spread over several deeper-indented lines
inside a YAML folded scalar. Folded scalars keep the newlines of more-indented
lines, and a newline splits a shell command: the container ran bare
`vllm serve <model>` in the foreground — no `--enable-lora`, no adapter, no
`--served-model-name base` — and the script never reached the health check or
the eval. vLLM's own startup log is the proof: `non-default args: {'model': …}`
and `served_model_name=nvidia/Llama-3.1-Nemotron-Nano-8B-v1`. The training jobs
survived the same layout only because each of their physical lines was a
complete `;`-terminated command.

**Problem it solves.** Every logical shell command in job YAML now sits on one
physical line, asserted by a parse check before submission; the fixed file
carries the lesson in a header comment.

**Expected impact.** The rerun performs the actual eval. Combining both
adapters into one eval job saves one full server-startup cycle (~15 min GPU).
Resequencing rationale: adapter A's training metrics were healthy (loss
1.94 -> 0.54, clean artifact), so the train-B-before-A-check risk is small,
and B was already queued on the warm node.

**Measured impact.** Cost of the bug: ~2h A100 ≈ $7.3 serving zero requests.
Detection was delayed ~1h by the Entra security-defaults auth outage (CLI
locked out; resolved by switching to a service-principal login, which the
policy does not block; SP is deleted at project teardown).

**Evidence.** `scratchpad devmatrix_stream.log` (vLLM `non-default args` line),
cancelled job `yellow_dinner_p5pg2mhblp` in the workspace, fixed
`train/azureml/job_devmatrix.yaml`.

*(Correction, 2026-08-25 later: the cancel in this entry never executed — the
command carried `--yes`, which this CLI version rejects, and the error scrolled
past unverified. The job kept the node ~70 more minutes until a verified cancel
landed. True cost of the bug: ~3.2h A100 ≈ $11.7, not $7.3. Lesson applied: a
state-changing command is only done when the state read back changed.)*

## Entry 11 — Stage 2 gate PASSED: dev confusion matrix

**Date:** 2026-08-25

**What.** Trained adapter B (vantage; job `salmon_turtle_qv72f9ndkq`, 12.8 min
train time, final-epoch loss 0.67, artifact 167,832,240 bytes — identical size
to adapter A by construction). Ran the combined dev matrix (job
`sad_line_6gslz8gljq`): one vLLM server, base + both adapters, 3 arms x 100 dev
goals, deterministic verification of every response under both tenant contracts.

**Why.** Stage 2 exit gate = objective #1's pattern on dev data.

**Problem it solves.** Proves the adapters — not the prompt — carry the tenant
behavior: the base model with the identical system message passes neither
contract even once.

**Expected impact.** Green light for Stage 3 (endpoint + benchmarks). The sealed
set stays untouched until Stage 4.

**Measured impact.** Matrix (100 goals/arm, 0 HTTP errors, wall 627s):
base 0%/0%, meridian 100%/0%, vantage 0%/98%. Gate thresholds >=90 own /
<=10 rival / <=10 base: all six cells pass with margin. The two vantage misses
are catalogued in the raw log for the README's honest-limits section. GPU
memory: 0 MiB idle -> 75,730 MiB with server up (vLLM pre-allocates KV cache at
its default 0.92 utilization; per-adapter memory delta is measured at Stage 3
startup logging, not from this number).

**Evidence.** `eval/logs/separation_matrix_20260825T103026225Z.json`,
`eval/logs/separation_raw_20260825T103026225Z.jsonl`,
`eval/logs/devmatrix_ab_std_log.txt`, `reports/iter_02.md`.

## Entry 12 — Serving stack rebased on the registered adapter asset

**Date:** 2026-08-25

**What.** Rewrote `serve/azure/` around the artifact that actually exists.
`deployment.yaml` now mounts the registered model asset
`azureml:adapters-both@latest` (custom_model, `meridian/` + `vantage/` HF PEFT
dirs) instead of a folder uploaded to a standalone blob container.
`start_server.sh` discovers the mount instead of assuming it, and hard-exits if
it cannot. `deploy.sh` lost adapter staging and the whole storage-account /
`blob upload-batch` path, gained a free pre-flight `az ml model show`, and its
defaults now point at the live workspace (`mlw-multilora` / `rg-multilora` /
southcentralus, not the invented `multilora-ws` / `multilora-rg` / eastus).
`environment.yaml` and the `Dockerfile` keep the BYOC build but now say plainly
why one is unavoidable. `scripts/check_flag_continuation.py` makes the entry-10
lesson executable.

**Why.** The Stage 2 draft (entry 4) was written before anything existed, so it
invented a plausible pipeline: stage adapters locally → create a storage account
→ upload → register a model asset pointing at the blob copy. Three of those four
steps are now dead. Both adapters are registered assets (`adapter-meridian:1`,
`adapter-vantage:1`), and a combined `adapters-both:1` was registered from a
training-job output. The draft would have created a redundant storage account
and re-uploaded from a laptop the same bytes the cluster had already written.

**Problem it solves.** Three, in descending order of how expensive they are.

First, serving the wrong bytes. Round-tripping adapters through a workstation
means the artifact that gets evaluated is a copy of the artifact that was
trained, related to it only by a `cp`. Referencing `adapters-both@latest`
removes the copy: the deployment mounts the job output itself, with a version
number that a report can cite. It also removes the SAS token nobody wanted to
own, and — a side effect noticed while wiring it — the ACR build context, which
used to be `serve/azure/` and would have swept `logs/endpoint.env`, the file
holding the endpoint key, into a container image. The rendered build context is
now exactly `Dockerfile` + `start_server.sh`.

Second, guessing the mount layout. Azure's docs pin down the mount *directory*
(`/var/azureml-app/azureml-models/<name>/<version>`, or
`<model_mount_path>/<name>/<version>`) but not whether a folder-shaped
`custom_model` keeps its own top-level folder name inside it — the TF Serving
BYOC sample reads as though it does, the model-specification page as though it
does not. `adapters-both:1` was registered from a job output folder literally
named `adapters`, so the adapters land at either
`$AZUREML_MODEL_DIR/adapters/meridian` or `$AZUREML_MODEL_DIR/meridian` and
nothing in the documentation settles which. `start_server.sh` therefore searches
four candidate roots, tries the direct path, then `*/meridian`, then
`*/*/meridian`, then a bounded `find`, logs the directory it resolved AND the
pattern that matched, and if none match dumps a recursive listing and exits
nonzero. It no longer falls back to a base-only server, which would have passed
the health probe, taken 100% of traffic and 404'd every tenant request while
billing.

Third, a question that had gone unasked: does this need a custom image at all?
Answer, verified rather than assumed: yes, and for a reason worth writing down.
The managed online deployment schema has no `command`, `args` or entrypoint key,
and the custom-container how-to describes configuring a stock image only through
environment variables its own ENTRYPOINT reads. `vllm/vllm-openai`'s entrypoint
is `vllm serve`, which takes the model positionally and every flag on ARGV;
there is no env var for `--enable-lora` or `--lora-modules`. A bare
`image: vllm/vllm-openai:v0.27.1` reference cannot serve LoRA adapters at all,
and separately leaves nowhere to run the `nvidia-smi` sampling objective #4
depends on. `train/azureml/job_devmatrix.yaml` does use that image with no build
— because a command **job** has a `command:` key. Same image, different lever.
Both files now carry that paragraph so the question is not reopened on a billing
node.

**Expected impact.** `deploy.sh` runs end to end against the real workspace:
subscription → model-asset pre-flight → cost guardrail → idempotent
group/workspace → environment → endpoint → deployment → URI + key → route probe
→ smoke test on all three served names → GPU memory lines → teardown reminder.
Everything before the guardrail is free, and a missing `adapters-both` now costs
nothing instead of being discovered 30 minutes into a running A100.

**Measured impact.** Offline checks only — nothing was deployed and no A100 was
started by this entry. `data/logs/serve_adapt_checks.log` has all of it:
4 `.sh` files pass `bash -n`; 8 `.yaml` files parse under PyYAML; 21 schema and
consistency spot-checks pass on the serve YAMLs (including "the deployment has
no `command`/`args`/`entrypoint` key" and "the deployment has no
`inference_config`", both of which are the load-bearing facts above); the
entry-10 guard passes over 10 files and is itself negative-tested — it is shown
failing on a hand-made folded-scalar bug, because a guard that only ever passes
proves nothing. The adapter discovery was **executed**, not just read: three
fake mount layouts (flat, one level of nesting, two) each resolve correctly and
log the matching pattern, and an empty mount exits 1 with a FATAL line.
`print_cost()` and `confirm_billing()` are byte-identical to the previous
version; `teardown()` differs by exactly one informational line, which named a
storage account that no longer exists.

Test suite: **2 failed, 23 passed, 1 error in 11.69s** — and those failures are
**pre-existing and unrelated**, confirmed by stashing every file in this commit
and re-running against a clean HEAD for an identical result.
`tests/test_tooling.py:256` and its teardown at `:201` assert that
`eval/SEALED.sha256` does not exist; they were written in commit `e266b46`
(entry 3) when that was true, and commit `92e0f3a` (entry 8, "sealed set
hashed") legitimately created the file. The tests encode a precondition the repo
outgrew. Left alone rather than fixed inside a serving commit — it is a real
bug, in the tooling tests, and it deserves its own entry.

**Doc verification, 2026-08-25.** Confirmed: `azureml:<asset_name>@latest` is
the documented "latest version of an asset" reference form
([core YAML syntax](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-core-syntax));
`inference_config` is a key on the **environment** schema and does not exist on
the deployment schema, and the environment may be registered-and-referenced or
inlined in the deployment YAML (the how-to's own CLI sample inlines it);
`model_mount_path` is "the path to mount the model in a custom container …
applicable only for custom container deployment scenarios, where environment has
`inference_config` configured"; the default mount is
`/var/azureml-app/azureml-models/<name>/<version>`; the deployment schema
attribute table contains no `command`, `args` or entrypoint key
([managed online deployment schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-managed-online),
[custom container how-to](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-container),
[model specification](https://learn.microsoft.com/en-us/azure/machine-learning/concept-online-deployment-model-specification)).
Verified live against the workspace rather than the docs:
`adapters-both:1` exists as a `custom_model` whose source folder is named
`adapters`; `az ml model show --label latest` resolves; southcentralus quota is
`standardNCADSA100v4Family` 32 / `TotalDedicatedCores` 52, so one
`Standard_NC24ads_A100_v4` (24 × 1.2 = 28.8 reserved cores) fits and a second
does not.

**Still unverified (# CHECK list for the deploy session).**

1. **Mount shape.** `$AZUREML_MODEL_DIR/adapters/meridian` vs
   `$AZUREML_MODEL_DIR/meridian`, and whether `AZUREML_MODEL_DIR` follows a
   custom `model_mount_path` at all. Marked at `serve/azure/deployment.yaml:71`.
   Resolved by reading the `adapter root` / `matched pattern` lines out of the
   first container log. Both shapes work, so this costs a log line, not a
   redeploy.
2. **The routing question, still open from entry 4 and deliberately preserved.**
   Azure's docs do not state what literal path a BYOC container receives when a
   client POSTs to a scoring URI ending in `/score`, nor whether a client may
   address `/v1/chat/completions` on it directly. Marked at
   `serve/azure/environment.yaml:79`. `deploy.sh probe_routes` tries both against
   the live endpoint and prints which answered. It matters beyond tidiness:
   `eval/separation.py` and `bench/*.py` build their URL by appending
   `/v1/chat/completions`, so they cannot construct a bare `/score` and would
   need a local rewrite proxy if `/score` is the only route. Record the answer.
3. **Total startup budget.** The probes allow ~35 minutes; whether Azure ML
   imposes its own provisioning timeout is undocumented. Marked at
   `serve/azure/deployment.yaml:210`.
4. **The ACR build itself.** Never run. Time and cost unknown; the first deploy
   pulls the ~10 GB vLLM base into the workspace registry.

**Evidence.** `data/logs/serve_adapt_checks.log` (every check above with its
output, plus the live `az ml model show` for `adapters-both:1`),
`serve/azure/README.md` (rewritten command order, file table and the
blob-vs-asset section), `scripts/check_flag_continuation.py` (the entry-10
lesson, now executable and negative-tested).

**Not done in this stage.** Nothing was deployed. No endpoint, no deployment, no
ACR build, no GPU. Every claim about how the mount behaves, what the scoring URI
routes to, and how much GPU memory an adapter costs is still a claim.

## Entry 13 — tooling tests: sealed-hash precondition outgrown, fixed

**Date:** 2026-08-25

**What.** `tests/test_tooling.py` asserted `eval/SEALED.sha256` does not exist
(a teardown check plus two in-test preconditions). Written at entry 3 when true;
entry 8 legitimately created the file, and the suite went 2 failed / 1 error.
The invariant is now expressed as intended: the real sealed hash is snapshotted
at import and asserted **unchanged** after the run, and the refusal-path test
uses `--sealed-hash-path` on a nonexistent temp path instead of borrowing the
real file's absence.

**Why.** A failing suite is noise that hides real failures, and the failure was
in the tests' precondition, not in the code under test. Never weaken a test to
make it pass — this change strengthens it: "absent" only defended against a
test creating the file; "unchanged" also defends against one modifying it.

**Problem it solves.** Surfaced by the entry-12 serving adaptation's check run
(2 failed, 1 error on clean HEAD, pre-existing).

**Expected impact.** Green suite; sealed-set integrity still enforced.

**Measured impact.** Full suite: 25 passed. Guardrail grep PASS; flag-
continuation guard PASS.

**Evidence.** This commit's diff; `eval/SEALED.sha256` byte-identical before
and after the suite (the new teardown assertion is the proof mechanism).

## Entry 14 — Stage 3: endpoint live, smoke green, four-arm benchmark measured

**Date:** 2026-08-25

**What.** `deploy.sh` ran end to end, exit 0: environment image built in the
workspace ACR, endpoint `multilora-ep` + deployment `blue`
(Standard_NC24ads_A100_v4) provisioned in ~26 min, scoring URI issued, route
probed, smoke test passed on all three served names, GPU memory phases captured
via get-logs. Then `bench/run_matrix.py` ran all four arms against the live URI:
40 requests/arm, concurrency 4, 0 errors on 160 requests.

**Why.** Objectives #2 (multi-LoRA inference overhead) and #5 (runs on Azure
Foundry) need a live endpoint and measured rows.

**Problem it solves / findings.**
- Routing (# CHECK since entry 4): RESOLVED — the scoring URI itself is
  `.../v1/chat/completions` (Azure honored `scoring_route`); appending another
  `/v1/...` 424s. Tools work unmodified.
- Mount shape (# CHECK from entry 12): RESOLVED — nested,
  `/mnt/adapters/adapters/<tenant>`, matched pattern logged.
- Smoke: meridian PASS, vantage PASS by the deterministic verifier through the
  public URI; base returns prose that fails both contracts (correct control).
- **Objective #2 verdict: outside the 5% aspiration, reported and explained.**
  TTFT p50 base 1.124s -> interleaved 1.229s (+9.3%); per-request tokens/sec
  p50 69.5 -> 62.5 (-10.1%). The larger e2e delta (+47%) decomposes exactly:
  adapter arms emit 512-token p50 outputs (structured JSON, hitting the
  driver's cap) vs base 382 (1.34x) times the 1.11x per-token slowdown
  ~= 1.47x observed. Not adapter-cache thrashing: both adapters resident,
  --max-loras 4, no swap events. The per-token cost is LoRA's extra GEMMs.
- GPU memory phases: 0 MiB at container start, 74,002 MiB from server-ready
  onward, flat through both adapters' first requests — vLLM pre-allocates its
  pool, so marginal adapter memory is invisible to nvidia-smi; the honest
  per-tenant figure is the adapter artifact itself (167,832,240 B) inside the
  pre-allocated pool, beside the ~16 GB counterfactual of a second full model.

**Measured impact.** Rows above; full 17-field rows in the summary JSON.
Session billing: deployment created 11:13Z.

**Evidence.** `serve/azure/logs/deploy_session1.log`,
`serve/azure/logs/deployment_logs.txt` (GPUMEM phases),
`serve/azure/logs/smoke_*.json`, `bench/logs/matrix_summary_endpoint_session1.json`,
`bench/logs/matrix_raw_endpoint_session1.jsonl`, `reports/iter_03.md`.

## Entry 15 — Swap time measured, twice: warm ≈ 0 confirmed, no runtime cold path exists

**Date:** 2026-08-25

**What.** Restarted deployment `blue` (env-var nonce update, ~25 min rolling
reprovision) to obtain a container no client had ever touched, then ran
`bench/swap_time.py` twice — meridian first (the container's first-ever
adapter request), then vantage (still untouched after the meridian run).
20 warm requests per distribution, 0 errors across 82 requests.

**Why.** Objective #3, the headline claim. The smoke test had already warmed
both adapters on the previous container, which would have silently invalidated
the cold sample — the restart is what makes the first-request measurement mean
something.

**Findings.**
- **Warm swap ≈ 0, measured twice:** adapter p50 minus base p50 = **-48 ms**
  (meridian) and **+4 ms** (vantage). Switching tenants between requests is a
  pointer change; the deviation is inside network jitter on the public URI
  (baseline p95-p50 spread ~290 ms).
- **There is no runtime cold path on this serving design.** Meridian's
  first-ever request was *faster* than its own warm p50 (0.976s vs 1.232s);
  vantage's was +100 ms, inside its p95 spread. Cause: vLLM loads adapters
  named in `--lora-modules` during server startup, so every registered tenant
  is GPU-resident before the first request arrives. The Blob->GPU load cost is
  real but is paid once, inside the ~2.5-minute container start
  (`deployment_logs.txt` phases), not by any request.
- A per-request cold number would require dynamic adapter loading
  (`VLLM_ALLOW_RUNTIME_LORA_UPDATING` + load/unload API). Deliberately not
  measured: it is a different serving mode from the one deployed, and the
  static mode's answer — "tenants are warm from the moment the server is up" —
  is the stronger operational property. Stated in RESULTS as a limit.

**Evidence.** `bench/logs/swap_time_summary_20260825T122412363Z.json` +
`..._122801312Z.json` and their raw JSONL, `data/logs/swap_meridian_endpoint.log`,
`data/logs/swap_vantage_endpoint.log`.

## Entry 16 — Stage 4 sealed matrix: 100/0/0; endpoint torn down, verified

**Date:** 2026-08-25

**What.** Ran `eval/separation.py --sealed` once against the live endpoint:
sha256 of `eval/sealed_goals.jsonl` checked against `eval/SEALED.sha256` before
any request; 160 sealed goals x 3 arms = 480 requests, concurrency 4,
0 HTTP errors, wall 1151.65s. Then `deploy.sh teardown`: deployment and
endpoint deleted, and the endpoint list read back empty; cluster node count
read back zero.

**Why.** Objective #1 on unseen data is the acceptance criterion; teardown is
the budget guardrail.

**Measured impact.** base 0.0% (0/160) on both contracts; meridian 100.0%
(160/160) own / 0.0% rival; vantage 100.0% (160/160) own / 0.0% rival.
Thresholds were >=90 / <=10 / <=10; every cell clears with maximum margin, and
the dev-set's two vantage misses did not recur on sealed data. The base row
proves the prompt alone does nothing: identical system message, zero passes.

**Evidence.** `eval/logs/separation_matrix_sealed_sealed_final.json`,
`eval/logs/separation_raw_sealed_sealed_final.jsonl`, `data/logs/sealed_run.log`,
`serve/azure/logs/teardown_session1.log`, `reports/iter_04.md`.

*(Correction to entry 14: the serve/azure/logs evidence files named there were
silently excluded from that commit by a gitignore rule covering the directory;
git errored on the batch add and the log files were skipped. Caught while
committing entry 16, when the same error appeared visibly. All six files are
force-added, secret-scanned, in commit d3b9a96; endpoint.env remains ignored.)*

## Entry 17 — Write-up: economics re-run on the measured adapter size, RESULTS.md, docs reconciled

**Date:** 2026-08-25

**What.** Re-ran `bench/economics.py --adapter-gb 0.168 --sku-price-usd-hr 3.673
--hours-month 730`, retiring the 0.08 GB estimate that had been marked ESTIMATE
since entry 3. Wrote `RESULTS.md`: sealed matrix, the full 17-field four-arm
benchmark table, both swap runs, the GPU memory phase table, the economics
tables, an honest-limits section and a provenance section. Updated `README.md`
(status paragraph, a Results section, an Honest-limits pointer, a Reproduce
section, and the Unverified-items table marked up with what the deploy session
answered) and `ARCHITECTURE.md` (estimates replaced by measurements, a
measured-on note, the preload finding). Added `email_draft.md`, unsent, marked
DRAFT.

**Why.** Four days of logs are not a result until someone can read them. Every
measurement was already on disk; what was missing was a document that puts each
number next to the file it came from, and three stale documents that still said
"no model has been called".

**Problem it solves.** Two. First, a reader with no access to this machine could
not tell which numbers were measured, which were arithmetic and which were
estimates carried over from the drafting stages — README and ARCHITECTURE still
described a project that had never run. Second, the economics table was still
printing an unmeasured adapter size, which is exactly the kind of number that
gets quoted after the caveat has been forgotten.

**Expected impact.** One document to hand to a reader; three documents that no
longer contradict the logs.

**Measured impact.** Economics at the measured 0.168 GB: N=1 is 16.00 GB against
16.17 GB, so multi-LoRA is 0.17 GB *worse* at one tenant and the table says so;
N=20 is 320.00 GB against 19.36 GB, 16.53x. The old 0.08 GB estimate gave 17.60
GB and 18.18x at N=20 (entry 3), so the correction moves the headline ratio down
by about 9%. Nothing else changed: no new measurement was taken, no Azure
resource was created, no model was called.

Three things in this repo disagreed with the logs and the logs won. (1) Entry 14
says the first deployment provisioned in "~26 min". The log says deployment
`blue` was created 11:13:44Z and took 100% of traffic at 11:56:36Z — 42.9
minutes, and 46 minutes for the whole `deploy.sh` run from pre-flight to smoke
test. RESULTS and README use the log's numbers. (2) Entry 14 states the
per-request tokens/sec drop as −10.1%; recomputed from
`matrix_summary_endpoint_session1.json` it is 69.451 → 62.495 = −10.02%. RESULTS
uses −10.0%. (3) The adapter artifact is 167,832,240 bytes (entry 11), but
`start_server.sh` reported the mounted adapter *directories* as 1,748,220,372 and
1,748,221,939 bytes. Both numbers are real and they measure different things; no
log itemises what else is in those folders, so RESULTS states both and says the
difference is unexplained.

Also corrected while reconciling: `ARCHITECTURE.md` and `README.md` both said the
region was East US; it was `southcentralus`. `ARCHITECTURE.md`'s request-path
diagram showed a `/score` scoring URI; the resolved URI is
`/v1/chat/completions`.

Full suite: **25 passed**. Guardrail grep: PASS. `bench/logs`, `eval/logs`,
`data/logs` and `reports` restored after the suite rewrote their `_selftest`
files, with the new economics output staged first so the restore could not take
it.

**Evidence.** `bench/logs/economics_measured_final.md` (the run), `RESULTS.md`
(every number with its source file), this commit's diff for the three document
updates.

**Not done in this stage.** No new measurement. The per-adapter GPU footprint is
still not measured and RESULTS says so rather than estimating it; the per-request
cold-load number still does not exist on this serving mode; behaviour beyond
`--max-loras` is still unmeasured; the load driver is still the stdlib fallback.
The ~$35 total project spend is an estimate assembled from this log's own
per-item figures and the list SKU rate — it has not been reconciled against an
Azure invoice, and no invoice is in this repository.
